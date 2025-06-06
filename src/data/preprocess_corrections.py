import argparse
import os
import io
import re
import logging
import tempfile
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LDH_MAP = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 10:1, 11: 0}
MIDDLE_SLICE_COUNT = 5

def parse_gcs_uri(gcs_uri):
    if not gcs_uri.startswith("gs://"): raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    parts = gcs_uri.replace("gs://", "").split("/", 1); bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith('/') and '.' not in prefix.split('/')[-1]: prefix += '/'
    return bucket_name, prefix

def list_gcs_files(storage_client, bucket_name, prefix, pattern=None):
    logger.info(f"Listing files in gs://{bucket_name}/{prefix} with pattern '{pattern if pattern else '*'}'")
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter='/')
    matching_files = []
    for blob in blobs:
        if blob.name.endswith('/'): continue
        relative_path = blob.name[len(prefix):] if blob.name.startswith(prefix) else blob.name
        if '/' in relative_path: continue
        filename = os.path.basename(blob.name)
        if pattern:
            if re.match(pattern, filename): matching_files.append(blob)
        else: matching_files.append(blob)
    logger.info(f"Found {len(matching_files)} matching files directly under prefix '{prefix}'.")
    return matching_files

def extract_sequence_number(filename):
    match = re.search(r'(\d+)\.[^.]+$', filename); return int(match.group(1)) if match else None

def download_and_preprocess_image(blob):
    try:
        logger.debug(f"Downloading {blob.name}..."); img_bytes = blob.download_as_bytes()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L'); img_np = np.array(img_pil, dtype=np.uint8)
        logger.debug(f"Preprocessing {blob.name} (shape: {img_np.shape}, dtype: {img_np.dtype})")
        equalized = cv2.equalizeHist(img_np); normalized = equalized.astype(np.float32) / 255.0
        logger.debug(f"Finished preprocessing {blob.name}"); return normalized
    except Exception as e:
        logger.error(f"Failed to download or preprocess image {blob.name}: {e}", exc_info=True); return None

def process_label_files( 
    corrections_gcs_dir: str,
    accepted_predictions_gcs_dir: str, 
    original_sequences_gcs_dir: str,
    output_classification_data_gcs_path: str,
    patch_height: int,
    patch_width: int
):
    """
    Reads correction and accepted prediction label files, prioritizes corrections,
    finds original images, extracts patches, saves individual patches as JPEGs,
    appends to labels.csv, and moves successfully processed txt files to an archive location.
    """
    start_time = time.time()
    logger.info("Starting label file preprocessing (Outputting JPEGs, Appending to CSV)...")
    storage_client = storage.Client()

    try:
        corr_bucket_name, corr_prefix = parse_gcs_uri(corrections_gcs_dir)
        acc_bucket_name, acc_prefix = parse_gcs_uri(accepted_predictions_gcs_dir)
        orig_bucket_name, orig_parent_prefix = parse_gcs_uri(original_sequences_gcs_dir)
        out_bucket_name, out_prefix = parse_gcs_uri(output_classification_data_gcs_path)
    except ValueError as e: logger.error(f"Configuration Error: {e}"); raise

    corr_bucket = storage_client.bucket(corr_bucket_name)
    acc_bucket = storage_client.bucket(acc_bucket_name)
    orig_bucket = storage_client.bucket(orig_bucket_name)
    out_bucket = storage_client.bucket(out_bucket_name)

    if not out_prefix.endswith('/'): out_prefix += '/'
    output_patches_prefix = os.path.join(out_prefix, "patches/").replace("\\", "/") 
    output_labels_csv_path = os.path.join(out_prefix, "labels.csv").replace("\\", "/") 

    archive_prefix_corr = os.path.join(os.path.dirname(corr_prefix.rstrip('/')), "processed/").replace("\\","/")
    archive_prefix_acc = os.path.join(os.path.dirname(acc_prefix.rstrip('/')), "processed/").replace("\\","/")

    logger.info(f"Reading corrections from: gs://{corr_bucket_name}/{corr_prefix}")
    logger.info(f"Reading accepted predictions from: gs://{acc_bucket_name}/{acc_prefix}")
    logger.info(f"Reading original images from parent: gs://{orig_bucket_name}/{orig_parent_prefix}")
    logger.info(f"Writing individual JPEG patches to: gs://{out_bucket_name}/{output_patches_prefix}")
    logger.info(f"Appending to labels CSV at: gs://{out_bucket_name}/{output_labels_csv_path}")
    logger.info(f"Processed correction TXT files will be moved to: gs://{corr_bucket_name}/{archive_prefix_corr}")
    logger.info(f"Processed accepted TXT files will be moved to: gs://{acc_bucket_name}/{archive_prefix_acc}")

    existing_labels_df = pd.DataFrame(columns=["filename", "label"])
    labels_blob = out_bucket.blob(output_labels_csv_path.replace(f"gs://{out_bucket_name}/", "", 1))
    if labels_blob.exists():
        try:
            logger.info(f"Found existing labels.csv at {output_labels_csv_path}. Attempting to load.")
            csv_data = labels_blob.download_as_text()
            existing_labels_df = pd.read_csv(io.StringIO(csv_data))
            if not existing_labels_df.empty and not ("filename" in existing_labels_df.columns and "label" in existing_labels_df.columns):
                logger.warning("Existing labels.csv has incorrect columns. Starting fresh.")
                existing_labels_df = pd.DataFrame(columns=["filename", "label"])
            logger.info(f"Loaded {len(existing_labels_df)} existing entries from labels.csv.")
        except Exception as e:
            logger.warning(f"Could not load or parse existing labels.csv: {e}. Starting with an empty DataFrame.")
            existing_labels_df = pd.DataFrame(columns=["filename", "label"])
    else:
        logger.info(f"No existing labels.csv found at {output_labels_csv_path}. A new one will be created.")

    files_to_process_map = {}
    correction_txt_blobs = list_gcs_files(storage_client, corr_bucket_name, corr_prefix, pattern=r"overlay_.*\.txt")
    logger.info(f"Found {len(correction_txt_blobs)} candidate correction .txt files.")
    corrected_request_ids = set()
    for blob in correction_txt_blobs:
        match = re.match(r"overlay_(.*)\.txt", os.path.basename(blob.name))
        if match:
            request_id = match.group(1)
            files_to_process_map[request_id] = (blob, 'correction')
            corrected_request_ids.add(request_id)

    accepted_txt_blobs = list_gcs_files(storage_client, acc_bucket_name, acc_prefix, pattern=r"accepted_.*\.txt")
    logger.info(f"Found {len(accepted_txt_blobs)} candidate accepted prediction .txt files.")
    for blob in accepted_txt_blobs:
        match = re.match(r"accepted_(.*)\.txt", os.path.basename(blob.name))
        if match:
            request_id = match.group(1)
            if request_id not in corrected_request_ids:
                if request_id not in files_to_process_map:
                    files_to_process_map[request_id] = (blob, 'accepted')
            else:
                logger.info(f"Request ID {request_id} from accepted file {blob.name} is superseded by a correction. Skipping.")
    all_files_to_process_info = []
    for req_id, (blob, file_type) in files_to_process_map.items():
        all_files_to_process_info.append((blob, file_type, req_id))
    logger.info(f"Total unique label files to process (after prioritization): {len(all_files_to_process_info)}")

    new_metadata_for_this_run = [] 
    processed_files_count = 0
    processed_jpgs_count = 0

    for txt_blob, file_type, request_id in all_files_to_process_info:
        file_processed_successfully = True
        boxes_processed_for_this_file = 0
        metadata_for_this_file_batch = []
        current_file_lines = [] 
        file_read_error_occurred = False 

        log_prefix = f"[RequestID: {request_id}] [{file_type.upper()}]"
        logger.info(f"--- Processing {file_type} file: {txt_blob.name} ---")

        #  1. find and load original sequence images 
        try:
            original_sequence_prefix = os.path.join(orig_parent_prefix, f"{request_id}/").replace("\\", "/")
            original_image_blobs = list_gcs_files(storage_client, orig_bucket_name, original_sequence_prefix, pattern=r".*\.(png|jpg|jpeg)$")
            if len(original_image_blobs) < MIDDLE_SLICE_COUNT: logger.warning(f"{log_prefix} Insufficient images ({len(original_image_blobs)}). Skipping file."); file_processed_successfully = False; continue
            img_name_num_map = {}; has_unparsable_name = False
            for b in original_image_blobs:
                 num = extract_sequence_number(os.path.basename(b.name));
                 if num is not None: img_name_num_map[b.name] = num
                 else: logger.warning(f"{log_prefix} Cannot extract num from {b.name}"); has_unparsable_name = True; break
            if has_unparsable_name: logger.warning(f"{log_prefix} Cannot sort images reliably. Skipping file."); file_processed_successfully = False; continue
            original_image_blobs.sort(key=lambda b: img_name_num_map[b.name])
            mid_index = len(original_image_blobs) // 2; start_index = max(0, mid_index - MIDDLE_SLICE_COUNT // 2)
            end_index = start_index + MIDDLE_SLICE_COUNT
            if end_index > len(original_image_blobs): logger.warning(f"{log_prefix} Cannot select middle slices. Skipping file."); file_processed_successfully = False; continue
            selected_blobs = original_image_blobs[start_index:end_index]
            if len(selected_blobs) != MIDDLE_SLICE_COUNT: logger.warning(f"{log_prefix} Incorrect middle slice count. Skipping file."); file_processed_successfully = False; continue
            original_images_processed = [download_and_preprocess_image(blob) for blob in selected_blobs]
            valid_processed_images = [img for img in original_images_processed if img is not None]
            if len(valid_processed_images) != MIDDLE_SLICE_COUNT: logger.warning(f"{log_prefix} Failed image processing. Skipping file."); file_processed_successfully = False; continue
            original_images_processed = valid_processed_images
            logger.info(f"{log_prefix} Loaded and preprocessed {MIDDLE_SLICE_COUNT} original images.")
        except Exception as img_load_err:
             logger.error(f"{log_prefix} Error finding/loading original images for {txt_blob.name}: {img_load_err}", exc_info=True)
             file_processed_successfully = False

        #  2. process label file contents 
        if file_processed_successfully:
            try:
                label_file_content = txt_blob.download_as_text()
                current_file_lines = [line for line in label_file_content.strip().split('\n') if line.strip()]
                logger.info(f"{log_prefix} Found {len(current_file_lines)} boxes in {txt_blob.name}.")
                if not current_file_lines: boxes_processed_for_this_file = 0

                for line_index, line in enumerate(current_file_lines):
                    box_processed_successfully = True
                    metadata_for_this_box = []
                    try:
                        parts = line.split();
                        if len(parts) != 5: logger.warning(f"{log_prefix} Malformed line {line_index + 1}. Skipping."); continue
                        raw_label_part = parts[0]; binary_label = None
                        if file_type == 'correction':
                            try: class_index = int(raw_label_part)
                            except ValueError: logger.warning(f"{log_prefix} Invalid class_index '{raw_label_part}' in correction. Skipping."); continue
                            binary_label = LDH_MAP.get(class_index)
                            if binary_label is None: logger.warning(f"{log_prefix} Class index {class_index} invalid for LDH_MAP. Skipping."); continue
                        elif file_type == 'accepted':
                            try: binary_label = int(raw_label_part)
                            except ValueError: logger.warning(f"{log_prefix} Invalid binary_label '{raw_label_part}' in accepted. Skipping."); continue
                            if binary_label not in [0, 1]: logger.warning(f"{log_prefix} Invalid binary label value {binary_label}. Skipping."); continue
                        else: logger.error(f"{log_prefix} Unknown file_type '{file_type}'. Skipping."); continue
                        x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                        if not (0.0 <= x_center_norm <= 1.0 and 0.0 <= y_center_norm <= 1.0 and 0.0 < width_norm <= 1.0 and 0.0 < height_norm <= 1.0): logger.warning(f"{log_prefix} Invalid coords. Skipping."); continue

                        patch_set_id = f"{request_id}_box{line_index}"
                        extracted_patches = []
                        extraction_failed = False

                        #  3. extract patches 
                        for img_idx, img_np in enumerate(original_images_processed):
                            h_img, w_img = img_np.shape; x_center_px = x_center_norm * w_img; y_center_px = y_center_norm * h_img
                            width_px = width_norm * w_img; height_px = height_norm * h_img
                            x1 = max(0, int(round(x_center_px - width_px / 2.0))); y1 = max(0, int(round(y_center_px - height_px / 2.0)))
                            x2 = min(w_img, int(round(x_center_px + width_px / 2.0))); y2 = min(h_img, int(round(y_center_px + height_px / 2.0)))
                            if x1 >= x2 or y1 >= y2: extraction_failed = True; logger.warning(f"{log_prefix} Invalid patch pixel dimensions. Skipping set."); break
                            patch = img_np[y1:y2, x1:x2]
                            if patch.size == 0: extraction_failed = True; logger.warning(f"{log_prefix} Zero-size patch. Skipping set."); break
                            resized_patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
                            if resized_patch.ndim == 2: resized_patch = np.expand_dims(resized_patch, axis=-1)
                            extracted_patches.append(resized_patch.astype(np.float32))

                        if extraction_failed or len(extracted_patches) != MIDDLE_SLICE_COUNT:
                            logger.warning(f"{log_prefix} Failed patch extraction for box {line_index+1}. Skipping this box.")
                            box_processed_successfully = False; continue

                        #  4. save individual patches as jpgs 
                        saved_jpg_count_for_box = 0
                        for img_idx, patch_float32 in enumerate(extracted_patches):
                            jpg_filename = f"{patch_set_id}_IMG{img_idx + 1}.jpg"
                            try:
                                output_blob_fullname = os.path.join(output_patches_prefix, jpg_filename).replace("\\", "/")
                                jpg_blob_obj = out_bucket.blob(output_blob_fullname)
                                patch_uint8 = (patch_float32.squeeze() * 255.0).clip(0, 255).astype(np.uint8)
                                pil_img = Image.fromarray(patch_uint8, mode='L'); img_byte_arr = io.BytesIO(); pil_img.save(img_byte_arr, format='JPEG'); img_byte_arr.seek(0)
                                jpg_blob_obj.upload_from_file(img_byte_arr, content_type='image/jpeg')
                                logger.debug(f"{log_prefix} Saved patch {jpg_filename} to gs://{out_bucket_name}/{output_blob_fullname}")
                                metadata_for_this_box.append({"filename": os.path.join("patches", jpg_filename).replace("\\", "/"), "label": binary_label}) 
                                saved_jpg_count_for_box += 1
                            except Exception as jpg_save_err:
                                logger.error(f"{log_prefix} Failed to save patch {jpg_filename}: {jpg_save_err}", exc_info=True)
                                box_processed_successfully = False; break

                        if box_processed_successfully and saved_jpg_count_for_box == MIDDLE_SLICE_COUNT:
                             metadata_for_this_file_batch.extend(metadata_for_this_box)
                             processed_jpgs_count += MIDDLE_SLICE_COUNT
                             boxes_processed_for_this_file += 1
                        else:
                             logger.warning(f"{log_prefix} Failed to save all JPGs for box {line_index+1}. Discarding metadata for this box.")

                    except Exception as line_err:
                        logger.error(f"{log_prefix} Failed processing line {line_index + 1} ('{line.strip()}'): {line_err}", exc_info=True)
            except Exception as file_read_err:
                 logger.error(f"{log_prefix} Failed reading/processing contents of {txt_blob.name}: {file_read_err}", exc_info=True)
                 file_processed_successfully = False

        #  move file after processing attempt 
        if file_processed_successfully and (boxes_processed_for_this_file > 0 or (isinstance(current_file_lines, list) and len(current_file_lines) == 0 and not file_read_error_occurred)):
            new_metadata_for_this_run.extend(metadata_for_this_file_batch) # Add this file's good data
            try:
                archive_destination_prefix = archive_prefix_corr if file_type == 'correction' else archive_prefix_acc
                destination_blob_name = os.path.join(archive_destination_prefix, os.path.basename(txt_blob.name)).replace("\\","/")
                source_bucket_for_move = txt_blob.bucket
                logger.info(f"{log_prefix} Moving successfully processed file {txt_blob.name} to gs://{source_bucket_for_move.name}/{destination_blob_name}")
                source_bucket_for_move.rename_blob(txt_blob, new_name=destination_blob_name)
                processed_files_count += 1
            except Exception as move_err:
                logger.error(f"{log_prefix} FAILED to move processed file {txt_blob.name} to archive: {move_err}", exc_info=True)
                
        else:
            logger.warning(f"{log_prefix} File {txt_blob.name} processing encountered errors or yielded no processable boxes. Leaving in input directory.")

    #  5. combine with existing and save final labels csv 
    if not new_metadata_for_this_run and existing_labels_df.empty:
        logger.warning("No new data processed and no existing labels.csv. Creating empty labels CSV.")
        final_labels_df = pd.DataFrame(columns=["filename", "label"])
    elif not new_metadata_for_this_run:
        logger.info("No new data processed in this run. Saving existing labels.csv.")
        final_labels_df = existing_labels_df
    else:
        new_labels_df = pd.DataFrame(new_metadata_for_this_run)
        if not existing_labels_df.empty:
            logger.info(f"Appending {len(new_labels_df)} new entries to {len(existing_labels_df)} existing entries.")
            final_labels_df = pd.concat([existing_labels_df, new_labels_df], ignore_index=True)
        else:
            logger.info(f"Creating new labels.csv with {len(new_labels_df)} entries.")
            final_labels_df = new_labels_df
        final_labels_df.drop_duplicates(subset=['filename'], keep='last', inplace=True)
        final_labels_df = final_labels_df[["filename", "label"]]

    logger.info(f"Saving final labels CSV with {len(final_labels_df)} entries to gs://{out_bucket_name}/{output_labels_csv_path}")
    try:
        csv_blob_obj = out_bucket.blob(output_labels_csv_path.replace(f"gs://{out_bucket_name}/", "", 1))
        csv_content = final_labels_df.to_csv(index=False); csv_blob_obj.upload_from_string(csv_content, content_type='text/csv')
    except Exception as csv_err: logger.error(f"Failed to upload labels CSV: {csv_err}", exc_info=True); raise

    end_time = time.time()
    logger.info(f"Label file preprocessing finished. Processed and moved {processed_files_count} label files, generating {processed_jpgs_count} JPEG patches in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess correction and accepted label data, save JPEGs, archive processed files.")
    parser.add_argument("--corrections-gcs-dir", required=True, help="GCS directory containing overlay_{request_id}.txt files.")
    parser.add_argument("--accepted-predictions-gcs-dir", required=True, help="GCS directory containing accepted_{request_id}.txt files.")
    parser.add_argument("--original-sequences-gcs-dir", required=True, help="GCS directory containing original input sequences parent folder.")
    parser.add_argument("--output-classification-data-gcs-path", required=True, help="GCS directory path to write processed patches and labels.csv.")
    parser.add_argument("--patch-height", type=int, default=30)
    parser.add_argument("--patch-width", type=int, default=50)

    args = parser.parse_args()

    process_label_files( 
        corrections_gcs_dir=args.corrections_gcs_dir,
        accepted_predictions_gcs_dir=args.accepted_predictions_gcs_dir,
        original_sequences_gcs_dir=args.original_sequences_gcs_dir,
        output_classification_data_gcs_path=args.output_classification_data_gcs_path,
        patch_height=args.patch_height,
        patch_width=args.patch_width
    )
