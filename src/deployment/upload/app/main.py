import os
import datetime
import io
import logging
import re
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from google.cloud import storage

import requests
import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

PROJECT_ID = os.environ.get('PROJECT_ID')
UPLOAD_BUCKET_NAME = os.environ.get('UPLOAD_BUCKET_NAME')
BACKEND_SERVICE_URL = os.environ.get('BACKEND_SERVICE_URL')
FRONTEND_SERVICE_ACCOUNT_EMAIL = os.environ.get('FRONTEND_SERVICE_ACCOUNT_EMAIL')
try:
    RESULT_BUCKET_NAME = os.environ.get('RESULT_BUCKET_NAME')
    if RESULT_BUCKET_NAME is None:
        raise ValueError("RESULT_BUCKET_NAME environment variable not set")
except:
    RESULT_BUCKET_NAME = UPLOAD_BUCKET_NAME

config_missing = False
if not PROJECT_ID:
    logging.error("FATAL: PROJECT_ID environment variable not set.")
    config_missing = True
if not UPLOAD_BUCKET_NAME:
    logging.error("FATAL: UPLOAD_BUCKET_NAME environment variable not set.")
    config_missing = True
if not BACKEND_SERVICE_URL:
    logging.error("FATAL: BACKEND_SERVICE_URL environment variable not set.")
    config_missing = True
if not RESULT_BUCKET_NAME:
    logging.error("FATAL: RESULT_BUCKET_NAME environment variable not set.")
    config_missing = True # Checked here

if not FRONTEND_SERVICE_ACCOUNT_EMAIL:
    logging.error("FATAL: FRONTEND_SERVICE_ACCOUNT_EMAIL environment variable not set. Cannot generate signed URLs.")
    config_missing = True
elif '@' not in FRONTEND_SERVICE_ACCOUNT_EMAIL or '.iam.gserviceaccount.com' not in FRONTEND_SERVICE_ACCOUNT_EMAIL:
     logging.error(f"FATAL: FRONTEND_SERVICE_ACCOUNT_EMAIL ({FRONTEND_SERVICE_ACCOUNT_EMAIL}) does not look like a valid service account email.")
     config_missing = True
if config_missing: raise ValueError("One or more required environment variables are missing or invalid.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

storage_client = None
upload_bucket = None
try:
    storage_client = storage.Client(project=PROJECT_ID)
    upload_bucket = storage_client.bucket(UPLOAD_BUCKET_NAME)
    logger.info(f"GCS Client initialized. Upload bucket: {UPLOAD_BUCKET_NAME}")
except Exception as e:
    logger.error(f"Fatal: Failed to initialize GCS client or bucket: {e}", exc_info=True)

RELEVANT_CLASS_NAMES = [
    'L1-L2', 'L1-L2_LDH', 'L2-L3', 'L2-L3_LDH', 'L3-L4', 'L3-L4_LDH',
    'L4-L5', 'L4-L5_LDH', 'L5-S1', 'L5-S1_LDH'
]

def get_id_token(audience_url):
    """Fetches OIDC ID token for the service account Cloud Run is running as."""
    try:
        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience_url)
        logging.debug(f"Successfully fetched ID token for audience: {audience_url}")
        return id_token
    except Exception as e:
        logging.error(f"Could not fetch ID token for audience {audience_url}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to obtain authentication token for backend service: {e}")

def extract_sequence_number_from_filename(filename):
    """Extracts trailing numbers from a filename before the extension."""
    match = re.search(r'(\d+)\.[^.]+$', filename)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

def generate_signed_url_with_token(bucket_name, blob_name, service_account_email, expiration_minutes=15):
    """Generates a v4 signed URL using refreshed token and SA email."""
    log_prefix = "[generate_signed_url]"
    if not storage_client:
        logging.error(f"{log_prefix} Storage client not initialized.")
        return None
    if not service_account_email:
        logging.error(f"{log_prefix} Service account email not provided.")
        return None

    try:
        logging.info(f"{log_prefix} Generating signed URL for gs://{bucket_name}/{blob_name}")
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        expiration_time = datetime.timedelta(minutes=expiration_minutes)
        credentials, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        logging.debug(f"{log_prefix} Credentials refreshed.")
        if not credentials.token: raise ValueError("Credentials token is missing after refresh.")
        logging.info(f"{log_prefix} Using SA email '{service_account_email}' and refreshed token for signing.")
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
            service_account_email=service_account_email,
            access_token=credentials.token
        )
        logging.info(f"{log_prefix} Generated signed URL successfully.")
        return signed_url
    except Exception as e:
        logging.error(f"{log_prefix} Failed to generate signed URL: {e}", exc_info=True)
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles upload, calls backend, generates signed URLs for mask & overlay."""
    if not storage_client or not upload_bucket:
         flash("Error: Storage service not configured correctly.", "error")
         return redirect(url_for('index'))

    uploaded_files = request.files.getlist("file")
    if not uploaded_files or not uploaded_files[0].filename:
        flash('No file selected for uploading.', "error")
        return redirect(url_for('index'))

    request_id_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S%f")
    log_prefix = f"[RequestID: {request_id_ts}]"
    logger.info(f"{log_prefix} Received {len(uploaded_files)} file(s).")

    input_type = "Unknown"
    backend_payload = {"request_id": request_id_ts}
    primary_gcs_uri_log = None

    try:
       
        if len(uploaded_files) == 1:
            file = uploaded_files[0]
            original_filename = file.filename
            _, file_extension = os.path.splitext(original_filename)
            file_extension = file_extension.lower()
            if file_extension == ".mha":
                input_type = "MHA"
                input_blob_name = f"input_mha/{request_id_ts}{file_extension}"
                input_gcs_uri = f"gs://{UPLOAD_BUCKET_NAME}/{input_blob_name}"
                primary_gcs_uri_log = input_gcs_uri
                blob = upload_bucket.blob(input_blob_name)
                blob.upload_from_file(file, content_type=file.content_type or 'application/octet-stream')
                backend_payload["mha_gcs_uri"] = input_gcs_uri
            elif file_extension in ['.png', '.jpg', '.jpeg']: 
                input_type = "SingleImage"
                gcs_prefix = f"input_sequences/{request_id_ts}/"
                input_blob_name = f"{gcs_prefix}{original_filename}"
                primary_gcs_uri_log = f"gs://{UPLOAD_BUCKET_NAME}/{gcs_prefix}"
                blob = upload_bucket.blob(input_blob_name)
                blob.upload_from_file(file, content_type=file.content_type or 'image/png')
                backend_payload["image_sequence_gcs_prefix"] = primary_gcs_uri_log
            else:
                flash(f"Unsupported file type: {original_filename}.", "error")
                return redirect(url_for('index'))
        elif len(uploaded_files) > 1:
            input_type = "Sequence"
            gcs_prefix = f"input_sequences/{request_id_ts}/"
            primary_gcs_uri_log = f"gs://{UPLOAD_BUCKET_NAME}/{gcs_prefix}"
            files_with_seq = []
            for file in uploaded_files: seq_num = extract_sequence_number_from_filename(file.filename)
            files_with_seq.append({'file': file, 'seq': seq_num})
            files_with_seq.sort(key=lambda item: item['seq'])
            for item in files_with_seq: file = item['file']
            original_filename = file.filename
            _, file_extension = os.path.splitext(original_filename)
            input_blob_name = f"{gcs_prefix}{original_filename}"
            blob = upload_bucket.blob(input_blob_name)
            file.seek(0)
            blob.upload_from_file(file, content_type=file.content_type or 'image/png')
            backend_payload["image_sequence_gcs_prefix"] = primary_gcs_uri_log
        else:
            flash('No files received.', "error")
            return redirect(url_for('index'))
        logging.info(f"{log_prefix} Upload successful for {input_type}.")

        predict_endpoint = f"{BACKEND_SERVICE_URL}/predict"
        auth_token = get_id_token(BACKEND_SERVICE_URL)
        headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        logging.info(f"{log_prefix} Calling backend service ({predict_endpoint}) with payload: {backend_payload}")
        response = requests.post(predict_endpoint, headers=headers, json=backend_payload, timeout=600)
        response.raise_for_status()
        logging.info(f"{log_prefix} Backend service responded with status: {response.status_code}")
        result_data = response.json()
        mask_gcs_uri = result_data.get("mask_gcs_uri")
        overlay_gcs_uri = result_data.get("overlay_gcs_uri")
        classification_results = result_data.get("classification_results")
        backend_request_id = result_data.get("request_id", request_id_ts)
        bounding_boxes = result_data.get("bounding_boxes", [])

        mask_signed_url = None
        overlay_signed_url = None
        if mask_gcs_uri and mask_gcs_uri.startswith("gs://"):
            parts = mask_gcs_uri.replace("gs://", "").split("/", 1)
            if len(parts) == 2: mask_signed_url = generate_signed_url_with_token(parts[0], parts[1], FRONTEND_SERVICE_ACCOUNT_EMAIL)
        if overlay_gcs_uri and overlay_gcs_uri.startswith("gs://"):
            parts = overlay_gcs_uri.replace("gs://", "").split("/", 1)
            if len(parts) == 2: overlay_signed_url = generate_signed_url_with_token(parts[0], parts[1], FRONTEND_SERVICE_ACCOUNT_EMAIL)

        # --- Render Result Page ---
        flash(f"Processing successful! Request ID: {backend_request_id}", "success")
        return render_template('result.html',
                               request_id=backend_request_id,
                               input_uri_info=primary_gcs_uri_log,
                               mask_uri=mask_gcs_uri,
                               overlay_uri=overlay_gcs_uri,
                               mask_signed_url=mask_signed_url,
                               overlay_signed_url=overlay_signed_url,
                               classification_results=classification_results,
                               bounding_boxes=bounding_boxes,
                               class_names=RELEVANT_CLASS_NAMES
                              )

    except requests.exceptions.Timeout:
        logging.error(f"{log_prefix} Timeout calling backend service.", exc_info=True)
        flash(f"Error: Prediction service timed out (Request ID: {request_id_ts}).", "error")
        return redirect(url_for('index'))
    except requests.exceptions.RequestException as http_err:
        logging.error(f"{log_prefix} HTTP error calling backend: {http_err}", exc_info=True)
        error_details = "Unknown backend error"
        try:
            if http_err.response is not None:
                 error_details = http_err.response.json().get("error", http_err.response.text)
            else:
                 error_details = str(http_err) 
        except (ValueError, AttributeError, TypeError): 
             error_details = str(http_err) 
        flash(f"Error calling prediction service: {error_details} (Request ID: {request_id_ts})", "error")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"{log_prefix} Error processing upload: {e}", exc_info=True)
        flash(f"An unexpected error occurred: {e} (Request ID: {request_id_ts})", "error")
        return redirect(url_for('index'))


@app.route('/save_accepted_predictions', methods=['POST'])
def save_accepted_predictions():
    if not storage_client:
        logger.error("Save accepted predictions failed: Storage client not configured")
        return jsonify({"error": "Storage client not configured"}), 500
    if not RESULT_BUCKET_NAME:
        logger.error("Save accepted predictions failed: RESULT_BUCKET_NAME not configured")
        return jsonify({"error": "Result bucket configuration missing"}), 500

    result_bucket_client = storage_client.bucket(RESULT_BUCKET_NAME)

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        request_id = data.get('request_id')
        initial_boxes = data.get('initial_boxes')
        classification_scores = data.get('classification_scores')

        if not request_id or initial_boxes is None or classification_scores is None:
            return jsonify({"error": "Missing 'request_id', 'initial_boxes', or 'classification_scores'"}), 400

        if len(initial_boxes) != len(classification_scores):
            return jsonify({"error": "Mismatch between number of boxes and scores"}), 400

        log_prefix = f"[SaveAcceptedPredictions RequestID: {request_id}]"
        logger.info(f"{log_prefix} Received {len(initial_boxes)} initial predictions for saving.")

        if not initial_boxes: 
            logger.info(f"{log_prefix} No initial predictions provided to save.")
            return jsonify({"message": "No initial predictions to save.", "accepted_label_gcs_path": None}), 200

        overlay_image_blob_name = f"output_overlays/overlay_{request_id}.png"
        overlay_blob = result_bucket_client.blob(overlay_image_blob_name)

        image_width = None
        image_height = None
        try:
            logger.info(f"{log_prefix} Reading dimensions from overlay image: gs://{RESULT_BUCKET_NAME}/{overlay_image_blob_name}")
            image_bytes = overlay_blob.download_as_bytes()
            overlay_image_obj = Image.open(io.BytesIO(image_bytes)) 
            image_width, image_height = overlay_image_obj.size
            logger.info(f"{log_prefix} Overlay image dimensions: {image_width}x{image_height}")
            overlay_image_obj.close()
        except Exception as img_err:
            logger.error(f"{log_prefix} Failed to read overlay image dimensions from GCS: {img_err}", exc_info=True)
            return jsonify({"error": f"Could not read overlay image for request {request_id} to normalize labels"}), 500

        if not image_width or not image_height or image_width == 0 or image_height == 0: # Added zero check
             logger.error(f"{log_prefix} Invalid image dimensions obtained ({image_width}x{image_height}).")
             return jsonify({"error": "Failed to get valid image dimensions for normalization"}), 500

        #  convert predictions to yolo format with 0/1 labels 
        output_lines = []
        for i, box_coords in enumerate(initial_boxes):
            score = classification_scores[i]

            if not isinstance(box_coords, list) or len(box_coords) != 4:
                logger.warning(f"{log_prefix} Skipping invalid box data: {box_coords}")
                continue
            if not isinstance(score, (float, int)): # score should be a number
                logger.warning(f"{log_prefix} Skipping box with invalid score data: {score}")
                continue

            x, y, w, h = box_coords # pixel coordinates [x_min, y_min, width, height]

            binary_label = 1 if score >= 0.5 else 0

            x_center = float(x) + float(w) / 2.0
            y_center = float(y) + float(h) / 2.0

            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = float(w) / image_width
            height_norm = float(h) / image_height

            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            output_line = f"{binary_label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            output_lines.append(output_line)

        if not output_lines:
            logger.warning(f"{log_prefix} No valid predictions found to save after processing.")
            return jsonify({"message": "No valid predictions to save after processing.", "accepted_label_gcs_path": None}), 200

        output_content_string = "\n".join(output_lines)
        accepted_label_blob_name = f"accepted_predictions/accepted_{request_id}.txt" # New path
        accepted_blob = result_bucket_client.blob(accepted_label_blob_name)
        target_gcs_path = f"gs://{RESULT_BUCKET_NAME}/{accepted_label_blob_name}"

        logger.info(f"{log_prefix} Saving {len(output_lines)} accepted predictions to {target_gcs_path}")
        accepted_blob.upload_from_string(output_content_string, content_type='text/plain')

        logger.info(f"{log_prefix} Accepted predictions saved successfully.")
        return jsonify({"message": "Accepted predictions saved successfully", "accepted_label_gcs_path": target_gcs_path}), 200

    except Exception as e:
        request_id_log = 'unknown'
        try:
            if request and request.is_json:
                request_id_log = request.get_json(silent=True).get('request_id', 'unknown_in_exception')
        except Exception:
            pass
        log_prefix = f"[SaveAcceptedPredictions RequestID: {request_id_log}]" # Use a more specific prefix
        logger.error(f"{log_prefix} Error saving accepted predictions: {e}", exc_info=True)
        return jsonify({"error": "Failed to save accepted predictions", "details": str(e)}), 500

@app.route('/save_corrections', methods=['POST'])
def save_corrections():
    if not storage_client:
        logger.error("Save corrections failed: Storage client not configured")
        return jsonify({"error": "Storage client not configured"}), 500
    if not UPLOAD_BUCKET_NAME: 
         logger.warning("UPLOAD_BUCKET_NAME not set in upload service.")
    if not RESULT_BUCKET_NAME:
         logger.error("Save corrections failed: RESULT_BUCKET_NAME not configured")
         return jsonify({"error": "Result bucket configuration missing"}), 500

    result_bucket_client = storage_client.bucket(RESULT_BUCKET_NAME)

    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Invalid JSON payload"}), 400

        request_id = data.get('request_id')
        # list of {"box": [x,y,w,h], "label": "classname"}
        corrected_annotations = data.get('corrected_annotations')

        if not request_id or corrected_annotations is None:
            return jsonify({"error": "Missing 'request_id' or 'corrected_annotations'"}), 400

        log_prefix = f"[SaveCorrections RequestID: {request_id}]"
        logger.info(f"{log_prefix} Received {len(corrected_annotations)} corrected annotations for YOLO conversion.")

        overlay_image_blob_name = f"output_overlays/overlay_{request_id}.png"
        overlay_blob = result_bucket_client.blob(overlay_image_blob_name)

        image_width = None
        image_height = None
        try:
            logger.info(f"{log_prefix} Reading dimensions from overlay image: gs://{RESULT_BUCKET_NAME}/{overlay_image_blob_name}")
            image_bytes = overlay_blob.download_as_bytes()
            overlay_image = Image.open(io.BytesIO(image_bytes))
            image_width, image_height = overlay_image.size
            logger.info(f"{log_prefix} Overlay image dimensions: {image_width}x{image_height}")
            overlay_image.close() 
        except Exception as img_err:
            logger.error(f"{log_prefix} Failed to read overlay image dimensions from GCS: {img_err}", exc_info=True)
            return jsonify({"error": f"Could not read overlay image for request {request_id} to normalize labels"}), 500

        if not image_width or not image_height:
             logger.error(f"{log_prefix} Invalid image dimensions obtained.")
             return jsonify({"error": "Failed to get valid image dimensions for normalization"}), 500

        class_name_to_index = {name: idx for idx, name in enumerate(RELEVANT_CLASS_NAMES)}
        logger.debug(f"{log_prefix} Class map: {class_name_to_index}")

        yolo_lines = []
        skipped_labels = 0
        for annotation in corrected_annotations:
            box = annotation.get("box")
            label = annotation.get("label")

            if not box or len(box) != 4 or label is None:
                logger.warning(f"{log_prefix} Skipping invalid annotation data: {annotation}")
                continue

            class_index = class_name_to_index.get(label)
            if class_index is None:
                logger.warning(f"{log_prefix} Skipping annotation with unrecognized label: '{label}'")
                skipped_labels += 1
                continue

            x, y, w, h = box

            x_center = x + w / 2.0
            y_center = y + h / 2.0

            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = w / image_width
            height_norm = h / image_height

            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))

            yolo_line = f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)

        if skipped_labels > 0:
             logger.warning(f"{log_prefix} Skipped {skipped_labels} annotations due to unrecognized labels.")
        if not yolo_lines:
             logger.warning(f"{log_prefix} No valid annotations found to save after conversion.")
  


        yolo_content_string = "\n".join(yolo_lines)
        yolo_label_blob_name = f"corrections/overlay_{request_id}.txt"
        yolo_blob = result_bucket_client.blob(yolo_label_blob_name)
        target_gcs_path = f"gs://{RESULT_BUCKET_NAME}/{yolo_label_blob_name}"

        logger.info(f"{log_prefix} Saving {len(yolo_lines)} YOLO formatted labels to {target_gcs_path}")
        yolo_blob.upload_from_string(yolo_content_string, content_type='text/plain')

        logger.info(f"{log_prefix} YOLOv8 format corrections saved successfully.")
        return jsonify({"message": "Corrections saved successfully in YOLOv8 format", "yolo_label_gcs_path": target_gcs_path}), 200

    except Exception as e:
        request_id_log = 'unknown'
        try:
             if request and request.is_json:
                  request_id_log = request.get_json(silent=True).get('request_id', 'unknown')
        except Exception:
             pass 
        log_prefix = f"[SaveCorrections RequestID: {request_id_log}]"
        logger.error(f"{log_prefix} Error saving corrections in YOLO format: {e}", exc_info=True)
        return jsonify({"error": "Failed to save corrections in YOLO format", "details": str(e)}), 500


if __name__ == '__main__':
    initialize_app() 
    if not INITIALIZED:
         print("FATAL: Application initialization failed. Exiting.")
         exit(1)
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)

