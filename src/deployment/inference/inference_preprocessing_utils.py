import os
import io
import logging
import re
import SimpleITK as sitk
import numpy as np
from PIL import Image
from skimage.transform import resize
from google.cloud import storage
from preprocessing_utils import normalize_image


PRE_CROP_TARGET_SHAPE = (512, 512)
CROP_Y_START = 96
CROP_X_START = 48
CROP_X_END_OFFSET = -48
FINAL_TARGET_SHAPE = (
    PRE_CROP_TARGET_SHAPE[0] - CROP_Y_START,
    PRE_CROP_TARGET_SHAPE[1] - CROP_X_START - abs(CROP_X_END_OFFSET)
)

NUM_SLICES_FOR_CLASSIFICATION = 5
MIDDLE_SLICE_OFFSET = NUM_SLICES_FOR_CLASSIFICATION // 2

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

storage_client = None

def initialize_gcs_client(client=None):
    """Initializes GCS client for this module."""
    global storage_client
    if client:
        logger.debug("Using GCS client passed from caller for inference utils.")
        storage_client = client
    elif not storage_client:
        logger.info("Initializing GCS client within inference_preprocessing_utils.")
        try:
            storage_client = storage.Client()
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)
            storage_client = None
    if not storage_client:
         logger.warning("GCS client is not available for inference utils.")


def preprocess_and_crop_slice_for_inference(slice_2d_np, pre_crop_shape=PRE_CROP_TARGET_SHAPE, final_shape=FINAL_TARGET_SHAPE):
    """
    Resizes, normalizes (using imported function), and crops a single 2D numpy slice.
    Returns a float32 numpy array in range [0, 1] with final_shape.
    """
    
    if slice_2d_np.ndim != 2:
        raise ValueError(f"preprocess_and_crop_slice_for_inference expects a 2D numpy array, got shape {slice_2d_np.shape}")

    logger.debug(f"Resizing slice from {slice_2d_np.shape} to {pre_crop_shape}")
    resized_slice = resize(
        slice_2d_np,
        pre_crop_shape, 
        order=3, # bicubic
        preserve_range=True,
        anti_aliasing=True
    )

    logger.debug("Normalizing slice...")
    normalized_slice = normalize_image(resized_slice)

    logger.debug(f"Cropping normalized slice from {CROP_Y_START}:, {CROP_X_START}:{CROP_X_END_OFFSET}")
    cropped_slice = normalized_slice[CROP_Y_START:, CROP_X_START:CROP_X_END_OFFSET]

    if cropped_slice.shape != final_shape:
        logger.warning(f"Cropped slice shape {cropped_slice.shape} does not match expected final shape {final_shape}. Check params.")

    logger.debug(f"Final slice shape: {cropped_slice.shape}, dtype: {cropped_slice.dtype}")
    return cropped_slice.astype(np.float32)

def load_and_preprocess_mha_for_inference(mha_file_path, target_shape=FINAL_TARGET_SHAPE, num_slices=NUM_SLICES_FOR_CLASSIFICATION):
    """
    Loads MHA from path, handles orientation, extracts middle N slices,
    and preprocesses them using preprocess_and_crop_slice_for_inference.

    Args:
        mha_file_path (str): Path to the .mha file.
        target_shape (tuple): Final target (H, W) AFTER cropping (e.g., 416x416).
        num_slices (int): Number of slices to extract around the middle.

    Returns:
        np.ndarray: Stack of preprocessed slices (num_slices, H, W, 1) as float32 [0,1].
                    Returns None if loading or processing fails.
    """
    logger.info(f"Loading MHA for inference from path: {mha_file_path}")
    if not isinstance(mha_file_path, str) or not os.path.exists(mha_file_path):
         logger.error(f"MHA file path is invalid or does not exist: {mha_file_path}")
         return None

    try:
        image = sitk.ReadImage(mha_file_path)
        image_array = sitk.GetArrayFromImage(image)
        direction = image.GetDirection()
        logger.info(f"MHA loaded. Initial shape: {image_array.shape}, Direction: {direction}")

        third_column = (direction[2], direction[5], direction[8])
        if np.allclose(third_column, (0, 0, 1), atol=1e-3):
            logger.info("Detected (x, y, z) orientation. Transposing and rotating...")
            image_array = np.transpose(image_array, (2, 1, 0))
            image_array = np.rot90(image_array, k=1, axes=(1, 2))
        logger.info(f"Shape after orientation handling: {image_array.shape}")

        total_slices = image_array.shape[0]
        if total_slices < num_slices:
            logger.error(f"MHA file has only {total_slices} slices, but {num_slices} are needed.")
            return None

        mid_idx = total_slices // 2
        start_idx = mid_idx - (num_slices // 2)
        end_idx = start_idx + num_slices
        start_idx = max(0, start_idx)
        end_idx = min(total_slices, end_idx)
        start_idx = max(0, end_idx - num_slices)

        logger.info(f"Extracting slices {start_idx} to {end_idx-1} from {total_slices} total slices.")
        slice_stack_3d = image_array[start_idx:end_idx, :, :]

        preprocessed_slices = []
        for i in range(slice_stack_3d.shape[0]):
            preprocessed_slice = preprocess_and_crop_slice_for_inference(slice_stack_3d[i, :, :], final_shape=target_shape)
            preprocessed_slices.append(preprocessed_slice)

        if len(preprocessed_slices) != num_slices:
             logger.warning(f"Expected {num_slices} preprocessed slices, but got {len(preprocessed_slices)}.")
             return None

        final_stack = np.stack(preprocessed_slices, axis=0)
        final_stack = np.expand_dims(final_stack, axis=-1)
        logger.info(f"MHA inference preprocessing complete. Final stack shape: {final_stack.shape}")
        return final_stack.astype(np.float32)

    except Exception as e:
        logger.error(f"Error processing MHA file '{mha_file_path}' for inference: {e}", exc_info=True)
        return None

def extract_sequence_number(filename):
    """Extracts trailing numbers from a filename before the extension."""
    match = re.search(r'(\d+)\.[^.]+$', filename)
    if match:
        try: return int(match.group(1))
        except ValueError: return None
    return None

def load_and_preprocess_sequence_for_inference(gcs_prefix_uri, target_shape=FINAL_TARGET_SHAPE, num_slices=NUM_SLICES_FOR_CLASSIFICATION):
    """
    Lists files in GCS prefix, sorts, selects middle N, downloads,
    and preprocesses them using preprocess_and_crop_slice_for_inference.

    Args:
        gcs_prefix_uri (str): GCS URI prefix (gs://bucket/path/folder/). Must end with '/'.
        target_shape (tuple): Final target (H, W) AFTER cropping (e.g., 416x416).
        num_slices (int): Number of slices required (e.g., 5).

    Returns:
        np.ndarray: Stack of preprocessed slices (num_slices, H, W, 1) as float32 [0,1].
                    Returns None if loading or processing fails.
    """
    logger.info(f"Loading PNG/JPG sequence for inference from GCS prefix: {gcs_prefix_uri}")
    if not gcs_prefix_uri or not gcs_prefix_uri.startswith("gs://") or not gcs_prefix_uri.endswith('/'):
        logger.error(f"Invalid GCS prefix URI format: {gcs_prefix_uri}")
        return None
    if not storage_client:
        logger.error("GCS client not initialized. Cannot process sequence.")
        return None

    try:
        bucket_name, prefix = gcs_prefix_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        logger.info(f"Listing blobs in bucket '{bucket_name}' with prefix '{prefix}'")

        blobs_with_seq = []
        all_blobs = list(bucket.list_blobs(prefix=prefix))
        logger.info(f"Found {len(all_blobs)} total blobs under prefix.")
        for blob in all_blobs:
            if blob.name.endswith('/') or blob.name == prefix: continue
            filename = os.path.basename(blob.name)
            seq_num = extract_sequence_number(filename)
            if seq_num is not None:
                blobs_with_seq.append({'blob': blob, 'seq': seq_num, 'name': filename})
            else:
                logger.warning(f"Could not extract sequence number from filename: {filename}. Skipping.")

        # check, sort, select middle n
        if len(blobs_with_seq) < num_slices:
            logger.error(f"Found only {len(blobs_with_seq)} sequence files, but {num_slices} required.")
            return None
        blobs_with_seq.sort(key=lambda item: item['seq'])
        logger.info(f"Sorted {len(blobs_with_seq)} sequence files.")

        total_valid_slices = len(blobs_with_seq)
        mid_idx_sorted = total_valid_slices // 2
        start_idx_sorted = mid_idx_sorted - (num_slices // 2)
        end_idx_sorted = start_idx_sorted + num_slices
        start_idx_sorted = max(0, start_idx_sorted)
        end_idx_sorted = min(total_valid_slices, end_idx_sorted)
        start_idx_sorted = max(0, end_idx_sorted - num_slices)
        selected_blobs_info = blobs_with_seq[start_idx_sorted:end_idx_sorted]

        if len(selected_blobs_info) != num_slices:
            logger.error(f"Could not select exactly {num_slices} middle slices. Found {len(selected_blobs_info)}.")
            return None
        logger.info(f"Selected middle {num_slices} slices (Indices {start_idx_sorted} to {end_idx_sorted-1}):")
        for item in selected_blobs_info: logger.info(f"  - {item['name']} (Seq: {item['seq']})")

        # download and process selected blobs
        preprocessed_slices = []
        for i, item in enumerate(selected_blobs_info):
            blob = item['blob']
            try:
                logger.debug(f"Downloading selected file: {blob.name}")
                image_bytes = blob.download_as_bytes()
                img = Image.open(io.BytesIO(image_bytes)).convert('L') # grayscale
                img_np = np.array(img)

                preprocessed_slice = preprocess_and_crop_slice_for_inference(img_np, final_shape=target_shape)
                preprocessed_slices.append(preprocessed_slice)
            except Exception as download_err:
                 logger.error(f"Failed to download or process selected file {blob.name}: {download_err}", exc_info=True)
                 return None

        final_stack = np.stack(preprocessed_slices, axis=0)
        final_stack = np.expand_dims(final_stack, axis=-1)
        logger.info(f"Sequence inference preprocessing complete. Final stack shape: {final_stack.shape}")
        return final_stack.astype(np.float32)

    except Exception as e:
        logger.error(f"Error processing PNG/JPG sequence from prefix {gcs_prefix_uri}: {e}", exc_info=True)
        return None

def preprocess_single_slice_for_segmentation_inference(image_bytes, target_shape=FINAL_TARGET_SHAPE):
    """
    Preprocesses a single image slice (bytes) for segmentation inference,
    including cropping. Returns shape (1, H_crop, W_crop, 1).
    """
    logger.info("Preprocessing single slice for segmentation inference...")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # grayscale
        img_np = np.array(img)

        preprocessed_cropped_slice = preprocess_and_crop_slice_for_inference(img_np, final_shape=target_shape)

        final_slice = np.expand_dims(preprocessed_cropped_slice, axis=-1)
        final_slice_batch = np.expand_dims(final_slice, axis=0)

        logger.info(f"Single slice inference preprocessing complete. Final shape: {final_slice_batch.shape}")
        return final_slice_batch.astype(np.float32)

    except Exception as e:
        logger.error(f"Error processing single slice for inference: {e}", exc_info=True)
        return None

