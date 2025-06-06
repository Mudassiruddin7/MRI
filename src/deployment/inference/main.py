import os
import datetime
import io
import logging
import time
import tempfile 

from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


try:
    import inference_preprocessing_utils as pp_utils
    import ml_inference_utils as ml_utils
    logger.info("Successfully imported local utility modules.")
except ImportError as e:
     logger.error(f"Failed to import utility modules (ensure inference_preprocessing_utils.py and ml_inference_utils.py are in the same directory or PYTHONPATH is set): {e}", exc_info=True)
     pp_utils = None
     ml_utils = None
     import sys
     sys.exit("Fatal: Could not load required utility modules.")


SEG_MODEL_URI = os.environ.get('SEG_MODEL_URI')
CLASS_MODEL_URI = os.environ.get('CLASS_MODEL_URI')
RESULT_BUCKET_NAME = os.environ.get('RESULT_BUCKET_NAME')
UPLOAD_BUCKET_NAME = os.environ.get('UPLOAD_BUCKET_NAME')
PORT = int(os.environ.get('PORT', 8080))

app = Flask(__name__)
storage_client = None
segmentation_model = None 
classification_model = None 
result_bucket = None
upload_bucket = None


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

@tf.keras.utils.register_keras_serializable()
def combo_loss(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f)
    dsc = dice_loss(y_true_f, y_pred_f) 
    return bce + dsc

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = tf.where(union == 0, 1.0, (2. * intersection + epsilon) / (union + epsilon))
    return dice

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = tf.where(union == 0, 1.0, (intersection + epsilon) / (union + epsilon))
    return iou

@tf.keras.utils.register_keras_serializable()
def channel_avg_pool(x):
    """Calculates the average across the channel axis."""
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable()
def channel_max_pool(x):
    """Calculates the maximum across the channel axis."""
    return tf.reduce_max(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable()
def stack_features_lambda_fn(x_list):
    """Stacks a list of tensors along axis 1."""
    return tf.stack(x_list, axis=1)

@tf.keras.utils.register_keras_serializable()
def reduce_sum_lambda_fn(x):
    """Reduces the sum of a tensor along axis 1."""
    return tf.reduce_sum(x, axis=1)


def load_models():
    """Loads both segmentation and classification models from GCS."""
    global segmentation_model, classification_model
    models_loaded = True

    all_custom_objects = {
        "dice_loss": dice_loss, "combo_loss": combo_loss,
        "dice_coefficient": dice_coefficient, "iou_metric": iou_metric,
        "channel_avg_pool": channel_avg_pool,
        "channel_max_pool": channel_max_pool,
        'stack_features_lambda_fn': stack_features_lambda_fn,
        'reduce_sum_lambda_fn': reduce_sum_lambda_fn
    }
    logger.info(f"Using custom objects for loading: {list(all_custom_objects.keys())}")

    if not SEG_MODEL_URI: logger.error("SEG_MODEL_URI env var not set."); models_loaded = False
    elif not SEG_MODEL_URI.startswith("gs://"): logger.error(f"Invalid SEG_MODEL_URI: {SEG_MODEL_URI}"); models_loaded = False
    else:
        logger.info(f"Loading segmentation model from: {SEG_MODEL_URI}")
        try:
            if not tf.io.gfile.exists(SEG_MODEL_URI): logger.error(f"Seg model not found: {SEG_MODEL_URI}"); models_loaded = False
            else:
                segmentation_model = tf.keras.models.load_model(SEG_MODEL_URI, custom_objects=all_custom_objects)
                logger.info("Segmentation model loaded successfully."); logger.info(f"  Input Shape: {segmentation_model.input_shape}")
        except Exception as e: logger.error(f"Error loading seg model: {e}", exc_info=True); segmentation_model = None; models_loaded = False

    if not CLASS_MODEL_URI: logger.error("CLASS_MODEL_URI env var not set."); models_loaded = False
    elif not CLASS_MODEL_URI.startswith("gs://"): logger.error(f"Invalid CLASS_MODEL_URI: {CLASS_MODEL_URI}"); models_loaded = False
    else:
        logger.info(f"Loading classification model from: {CLASS_MODEL_URI}")
        try:
            if not tf.io.gfile.exists(CLASS_MODEL_URI): logger.error(f"Class model not found: {CLASS_MODEL_URI}"); models_loaded = False
            else:
                classification_model = tf.keras.models.load_model(CLASS_MODEL_URI, custom_objects=all_custom_objects)
                logger.info("Classification model loaded successfully.")
        except Exception as e: logger.error(f"Error loading class model: {e}", exc_info=True); classification_model = None; models_loaded = False

    return models_loaded

def initialize_gcs():
    """Initializes GCS client and result/upload bucket objects."""
    global storage_client, result_bucket, upload_bucket
    try:
        storage_client = storage.Client()
        logger.info("GCS Client initialized.")
        if pp_utils: pp_utils.initialize_gcs_client(storage_client) # Pass client to utils
        else: logger.error("Preprocessing utils not loaded."); return False

        if RESULT_BUCKET_NAME: result_bucket = storage_client.bucket(RESULT_BUCKET_NAME); logger.info(f"Result bucket set to: {RESULT_BUCKET_NAME}")
        else: logger.error("RESULT_BUCKET_NAME missing."); return False

        if UPLOAD_BUCKET_NAME: upload_bucket = storage_client.bucket(UPLOAD_BUCKET_NAME); logger.info(f"Upload bucket set to: {UPLOAD_BUCKET_NAME}")
        else: logger.warning("UPLOAD_BUCKET_NAME not set, /save_corrections will fail."); upload_bucket = None

        return True
    except Exception as e: logger.error(f"Error initializing GCS: {e}", exc_info=True); return False

INITIALIZED = False
def initialize_app():
    global INITIALIZED
    if not INITIALIZED:
        logger.info("Starting application initialization...")
        if not pp_utils or not ml_utils: logger.error("Utility modules failed to import."); return False
        gcs_ok = initialize_gcs(); models_ok = load_models()
        if gcs_ok and models_ok: INITIALIZED = True; logger.info("Application initialization successful.")
        else: logger.error("Application initialization failed (GCS or Model loading).")
    return INITIALIZED

@app.route('/health', methods=['GET'])
def health_check():
    if not INITIALIZED: return "Initialization failed", 503
    if not storage_client: return "GCS Client not initialized", 503
    if not segmentation_model: return "Segmentation model not loaded", 503
    if not classification_model: return "Classification model not loaded", 503
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handles MHA or Image Sequence input for 2-stage prediction."""
    start_time = time.time()
    request_data = request.get_json()
    request_id_from_caller = request_data.get('request_id') if request_data else None
    request_id = request_id_from_caller if request_id_from_caller else datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    log_prefix = f"[RequestID: {request_id}]"
    logger.info(f"{log_prefix} Prediction request received.")

    if not INITIALIZED: return jsonify({"error": "Service not initialized", "request_id": request_id}), 503
    if not pp_utils or not ml_utils: return jsonify({"error": "Internal server error: utilities missing", "request_id": request_id}), 500
    if not segmentation_model: return jsonify({"error": "Segmentation model not loaded", "request_id": request_id}), 500
    if not classification_model: return jsonify({"error": "Classification model not loaded", "request_id": request_id}), 500
    if not storage_client or not result_bucket: return jsonify({"error": "GCS not configured", "request_id": request_id}), 500
    if not request_data: return jsonify({"error": "Request must be JSON", "request_id": request_id}), 400

    #  determine input type and preprocess 
    mha_gcs_uri = request_data.get('mha_gcs_uri')
    image_sequence_gcs_prefix = request_data.get('image_sequence_gcs_prefix')
    slice_stack = None # holds the (5, h_crop, w_crop, 1) numpy array
    input_type = "Unknown"
    primary_input_uri_log = "N/A"

    try:
        preprocess_start = time.time()
        if mha_gcs_uri and mha_gcs_uri.startswith("gs://"):
            input_type = "MHA"
            primary_input_uri_log = mha_gcs_uri
            logger.info(f"{log_prefix} Processing MHA input: {mha_gcs_uri}")
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_mha_path = os.path.join(tmpdir, f"{request_id}.mha")
                try:
                    bucket_name, blob_name = mha_gcs_uri.replace("gs://", "").split("/", 1)
                    input_bucket = upload_bucket if upload_bucket else storage_client.bucket(bucket_name)
                    blob = input_bucket.blob(blob_name)
                    logger.info(f"{log_prefix} Downloading MHA blob: {blob_name} from bucket {input_bucket.name}")
                    blob.download_to_filename(temp_mha_path)
                    slice_stack = pp_utils.load_and_preprocess_mha_for_inference(temp_mha_path)
                except Exception as download_err:
                     logger.error(f"{log_prefix} Failed to download or access MHA from {mha_gcs_uri}: {download_err}", exc_info=True)
                     raise download_err

            if slice_stack is not None and upload_bucket: 
                logger.info(f"{log_prefix} Saving {slice_stack.shape[0]} processed slices from MHA as JPG sequence...")
                save_errors = 0
                for i in range(slice_stack.shape[0]):
                    try:
                        slice_np_float32 = slice_stack[i, ..., 0] # get slice (h, w) remove channel dim
                        slice_uint8 = (slice_np_float32 * 255.0).clip(0, 255).astype(np.uint8)

                        slice_filename = f"Image{i+1:05d}.jpg"
                        blob_path = f"input_sequences/{request_id}/{slice_filename}"

                        slice_blob = upload_bucket.blob(blob_path)

                        pil_img = Image.fromarray(slice_uint8, mode='L')
                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format='JPEG')
                        img_byte_arr.seek(0) # rewind buffer, didnt work without it

                        slice_blob.upload_from_file(img_byte_arr, content_type='image/jpeg')
                        logger.debug(f"{log_prefix} Saved slice {i+1} to gs://{UPLOAD_BUCKET_NAME}/{blob_path}")

                    except Exception as save_err:
                        save_errors += 1
                        logger.error(f"{log_prefix} Failed to save slice {i+1} from MHA to JPG: {save_err}", exc_info=True)

                if save_errors == 0:
                    logger.info(f"{log_prefix} Successfully saved all {slice_stack.shape[0]} slices as JPG sequence.")
                else:
                    logger.warning(f"{log_prefix} Encountered {save_errors} errors while saving slices as JPG sequence.")
            elif slice_stack is not None and not upload_bucket:
                 logger.warning(f"{log_prefix} Cannot save MHA-derived sequence, UPLOAD_BUCKET_NAME not configured.")
        
        elif image_sequence_gcs_prefix and image_sequence_gcs_prefix.startswith("gs://"):
            input_type = "Sequence"
            primary_input_uri_log = image_sequence_gcs_prefix
            logger.info(f"{log_prefix} Processing image sequence input from prefix: {image_sequence_gcs_prefix}")
            slice_stack = pp_utils.load_and_preprocess_sequence_for_inference(image_sequence_gcs_prefix)

        else:
            logger.warning(f"{log_prefix} Invalid input payload.")
            return jsonify({"error": "Invalid input payload", "request_id": request_id}), 400

        preprocess_time = time.time() - preprocess_start
        if slice_stack is None:
            logger.error(f"{log_prefix} Preprocessing failed for input type {input_type}.")
            return jsonify({"error": f"Preprocessing failed for {input_type}", "request_id": request_id}), 500
        logger.info(f"{log_prefix} Preprocessing successful ({preprocess_time:.4f}s). Slice stack shape: {slice_stack.shape}")

        #  1. segmentation 
        middle_slice_index = pp_utils.MIDDLE_SLICE_OFFSET
        middle_slice_input_batch = np.expand_dims(slice_stack[middle_slice_index], axis=0)
        logger.info(f"{log_prefix} Running segmentation on middle slice (index {middle_slice_index}). Shape: {middle_slice_input_batch.shape}")
        seg_start = time.time()
        mask = ml_utils.generate_mask(middle_slice_input_batch, segmentation_model)
        seg_time = time.time() - seg_start
        if mask is None:
             logger.error(f"{log_prefix} Segmentation failed.")
             return jsonify({"error": "Segmentation failed", "request_id": request_id}), 500
        logger.info(f"{log_prefix} Segmentation complete ({seg_time:.4f}s). Mask shape: {mask.shape}")

        #  2. bounding box extraction 
        box_start = time.time()
        boxes = ml_utils.find_bounding_boxes_contours(mask)
        box_time = time.time() - box_start
        logger.info(f"{log_prefix} Found {len(boxes)} bounding boxes ({box_time:.4f}s).")

        #  3. classification (if boxes found) 
        classification_results = []
        class_preproc_time = 0
        class_predict_time = 0
        if boxes:
            class_preproc_start = time.time()
            list_of_patch_sets = ml_utils.classification_preprocessing_5slices(slice_stack, boxes)
            class_preproc_time = time.time() - class_preproc_start
            logger.info(f"{log_prefix} Classification preprocessing complete ({class_preproc_time:.4f}s). Found {len(list_of_patch_sets)} valid patch sets.")
            if list_of_patch_sets:
                class_predict_start = time.time()
                classification_results = ml_utils.predict_label_5slices(list_of_patch_sets, classification_model)
                class_predict_time = time.time() - class_predict_start
                logger.info(f"{log_prefix} Classification prediction complete ({class_predict_time:.4f}s). Results: {classification_results}")
            else: logger.info(f"{log_prefix} No valid patch sets generated, skipping classification.")
        else: logger.info(f"{log_prefix} No bounding boxes found, skipping classification.")

        #  4. create and upload overlay image 
        overlay_gcs_uri = None
        overlay_upload_time = 0
        try:
            overlay_start = time.time()
            middle_slice_for_overlay = slice_stack[middle_slice_index, ..., 0]
            logger.info(f"{log_prefix} Creating overlay image from slice shape: {middle_slice_for_overlay.shape}")
            overlay_img_np = ml_utils.create_overlay_image(middle_slice_for_overlay, boxes) # Call the new util function

            if overlay_img_np is not None:
                overlay_filename = f"overlay_{request_id}.png"
                overlay_blob_path = f"output_overlays/{overlay_filename}"
                overlay_blob = result_bucket.blob(overlay_blob_path)
                overlay_gcs_uri = f"gs://{RESULT_BUCKET_NAME}/{overlay_blob_path}"
                logger.info(f"{log_prefix} Uploading overlay image to {overlay_gcs_uri}")
                overlay_image_pil = Image.fromarray(cv2.cvtColor(overlay_img_np, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                overlay_image_pil.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                overlay_blob.upload_from_string(img_byte_arr, content_type='image/png')
                overlay_upload_time = time.time() - overlay_start
                logger.info(f"{log_prefix} Overlay image uploaded ({overlay_upload_time:.4f}s).")
            else: logger.error(f"{log_prefix} Failed to create overlay image.")
        except Exception as overlay_err: logger.error(f"{log_prefix} Error during overlay creation/upload: {overlay_err}", exc_info=True); overlay_gcs_uri = None

        #  5. upload segmentation mask 
        mask_upload_time = 0
        mask_gcs_uri = None
        try:
            mask_upload_start = time.time()
            result_mask_filename = f"mask_{request_id}.png"
            result_blob_path = f"output_masks/{result_mask_filename}"
            result_blob = result_bucket.blob(result_blob_path)
            mask_gcs_uri = f"gs://{RESULT_BUCKET_NAME}/{result_blob_path}"
            logger.info(f"{log_prefix} Uploading result mask to {mask_gcs_uri}")
            mask_image_pil = Image.fromarray(mask * 255, mode='L')
            img_byte_arr = io.BytesIO()
            mask_image_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            result_blob.upload_from_string(img_byte_arr, content_type='image/png')
            mask_upload_time = time.time() - mask_upload_start
            logger.info(f"{log_prefix} Result mask uploaded ({mask_upload_time:.4f}s).")
        except Exception as mask_upload_err: logger.error(f"{log_prefix} Error uploading mask image: {mask_upload_err}", exc_info=True); mask_gcs_uri = None

        #  6. log success and return combined results 
        total_time = time.time() - start_time
        success_log = {
            "message": "Prediction successful", "request_id": request_id,
            "input_type": input_type, "input_uri_processed": primary_input_uri_log,
            "output_mask_gcs_uri": mask_gcs_uri, "output_overlay_gcs_uri": overlay_gcs_uri,
            "num_boxes_found": len(boxes), "num_classifications_run": len(classification_results),
            "classification_scores": classification_results, "total_latency_sec": round(total_time, 4),
            "preprocess_latency_sec": round(preprocess_time, 4), "segmentation_latency_sec": round(seg_time, 4),
            "classification_total_latency_sec": round(class_preproc_time + class_predict_time, 4),
            "mask_upload_latency_sec": round(mask_upload_time, 4), "overlay_creation_upload_latency_sec": round(overlay_upload_time, 4),
        }
        logger.info(success_log)
        return jsonify({
            "mask_gcs_uri": mask_gcs_uri, "overlay_gcs_uri": overlay_gcs_uri,
            "classification_results": classification_results, "bounding_boxes": boxes,
            "request_id": request_id
        }), 200

    except Exception as e:
        total_time = time.time() - start_time
        error_log = {
            "message": "Prediction failed", "request_id": request_id,
            "input_type": input_type, "input_uri_processed": primary_input_uri_log,
            "total_latency_sec": round(total_time, 4),
            "error_type": type(e).__name__, "error_details": str(e),
        }
        logger.error(error_log, exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e), "request_id": request_id}), 500
    
@app.route('/reload-models', methods=['POST'])
def trigger_model_reload():
    """
    Endpoint to be called by Eventarc to trigger a reload of ML models.
    Assumes new models are at the same GCS paths defined by env variables.
    """

    
    logger.info("Received request to reload models via /reload-models endpoint...")
    

    reload_successful = load_models()
    
    if reload_successful:
        logger.info("Models reloaded successfully based on current GCS URIs.")
        return jsonify({"message": "Models reloaded successfully"}), 200
    else:
        logger.error("Failed to reload one or more models. Check previous logs for details.")
        return jsonify({"error": "Failed to reload models"}), 500

#  run initialization 
initialize_app()

if __name__ == '__main__':
    if not INITIALIZED:
         print("Initialization failed, exiting.")
         exit(1)
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False) # Disable reloader for stability
