import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import aiplatform as aip
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PATCH_HEIGHT = 30
PATCH_WIDTH = 50
NUM_INPUTS = 5 
INPUT_SHAPE = (PATCH_HEIGHT, PATCH_WIDTH, 1) 
AUTOTUNE = tf.data.AUTOTUNE

@tf.keras.utils.register_keras_serializable()
def stack_features_lambda_fn(x_list):
    """Stacks a list of tensors along axis 1."""
    return tf.stack(x_list, axis=1)

@tf.keras.utils.register_keras_serializable()
def reduce_sum_lambda_fn(x):
    """Reduces the sum of a tensor along axis 1."""
    return tf.reduce_sum(x, axis=1)

def read_image_paths_and_labels_from_gcs(gcs_dir_path):
    """
    Reads labels.csv and lists corresponding jpg files from a GCS directory.
    """
    labels_csv_path = os.path.join(gcs_dir_path, "labels.csv").replace("\\","/")
    logger.info(f"Attempting to load metadata from: {labels_csv_path}")

    if not tf.io.gfile.exists(labels_csv_path):
        logger.warning(f"Labels CSV not found at: {labels_csv_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        with tf.io.gfile.GFile(labels_csv_path, 'r') as f:
            labels_df = pd.read_csv(f)

        if labels_df.empty:
             logger.warning(f"Labels CSV at {labels_csv_path} is empty.")
             return pd.DataFrame()

        labels_df.columns = labels_df.columns.str.lower()
        if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
            raise ValueError(f"'filename' or 'label' column missing in {labels_csv_path}.")
        logger.info(f"Loaded {len(labels_df)} rows from {labels_csv_path}")

        label_dict = dict(zip(labels_df['filename'], labels_df['label']))

        image_files = []
        for pattern in ["*.jpg", "patches/*.jpg"]:
            found_files = tf.io.gfile.glob(os.path.join(gcs_dir_path, pattern))
            if found_files:
                image_files.extend(found_files)
                logger.info(f"Found {len(found_files)} jpg files matching pattern '{pattern}' in {gcs_dir_path}.")
        
        logger.info(f"Found {len(image_files)} total jpg files in and under {gcs_dir_path}.")

        if not image_files:
            logger.warning(f"No .jpg files found in {gcs_dir_path} or its patches/ subdirectory.")
            return pd.DataFrame()

        id_groups = {}
        for img_path in image_files:
            img_file = os.path.basename(img_path)
            try:
                # expects format like "id_anything_imgx.jpg" or "id_imgx.jpg"
                parts = img_file.split('_')
                if len(parts) < 2:
                    logger.debug(f"Skipping file with insufficient parts for ID extraction: {img_file}")
                    continue

                img_part_str = parts[-1] # should be "imgx.jpg"
                id_num = '_'.join(parts[:-1])

                img_idx_str = img_part_str.split('.')[0].replace('IMG', '')
                img_idx = int(img_idx_str)

                if not (1 <= img_idx <= NUM_INPUTS): continue 

                if id_num not in id_groups: id_groups[id_num] = {}
                id_groups[id_num][f"IMG{img_idx}"] = img_path

            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping file with unexpected format: {img_file}. Error: {e}")
                continue

        data = []
        required_imgs = [f"IMG{i}" for i in range(1, NUM_INPUTS + 1)]
        for id_num, images_dict in id_groups.items():
            if all(key in images_dict for key in required_imgs):
                label_key_filename = os.path.basename(images_dict["IMG1"])
                label = label_dict.get(label_key_filename)

                if label is not None:
                    sorted_image_paths = [images_dict[f"IMG{i}"] for i in range(1, NUM_INPUTS + 1)]
                    row = {'ID': id_num, 'Label': label}
                    row.update({f"IMG{i}": path for i, path in enumerate(sorted_image_paths, 1)})
                    data.append(row)

        df_out = pd.DataFrame(data)
        if not df_out.empty:
            path_cols = [f"IMG{i}" for i in range(1, NUM_INPUTS + 1)]
            initial_len = len(df_out)
            df_out.dropna(subset=path_cols + ['Label'], inplace=True)
            if len(df_out) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df_out)} rows from {gcs_dir_path} due to missing paths or labels.")

        logger.info(f"Returning DataFrame with {len(df_out)} samples for path {gcs_dir_path}")
        return df_out

    except Exception as e:
        logger.error(f"Failed reading data from {gcs_dir_path}: {e}", exc_info=True)
        return pd.DataFrame()

def load_and_process_image(path_tensor):
    """Loads and preprocesses single JPG image from GCS path tensor."""
    global PATCH_HEIGHT, PATCH_WIDTH, INPUT_SHAPE
    try:
        image = tf.io.read_file(path_tensor)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [PATCH_HEIGHT, PATCH_WIDTH]) 
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape(INPUT_SHAPE) 
        return image
    except Exception as e:
        tf.print(f"Error loading image {path_tensor}: {e}", output_stream=sys.stderr)
        return tf.zeros(INPUT_SHAPE, dtype=tf.float32)

@tf.function
def process_input_row(*args):
    """ Loads the N images corresponding to one sample and the label. """
    
    image_paths = args[:NUM_INPUTS] 
    label = args[NUM_INPUTS]       

    processed_images = [load_and_process_image(path) for path in image_paths]
    return tuple(processed_images), tf.cast(label, tf.float32)

def build_dataset(meta_df, batch_size, shuffle=False):
    """Builds tf.data.Dataset from metadata DataFrame."""
    global INPUT_SHAPE, NUM_INPUTS 

    if meta_df.empty:
        logger.warning("Input DataFrame is empty. Returning empty dataset structure.")
        tensor_specs_inputs = tuple([tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)] * NUM_INPUTS)
        tensor_specs_label = tf.TensorSpec(shape=(), dtype=tf.float32)
        output_signature = (tensor_specs_inputs, tensor_specs_label)
        return tf.data.Dataset.from_generator(lambda: iter([]), output_signature=output_signature).batch(batch_size)

    path_cols = [f'IMG{i}' for i in range(1, NUM_INPUTS + 1)]
    if not all(col in meta_df.columns for col in path_cols + ['Label']):
         raise ValueError(f"DataFrame is missing required columns. Need: {path_cols + ['Label']}. Found: {meta_df.columns}")

    if not pd.api.types.is_numeric_dtype(meta_df['Label']):
        logger.warning("'Label' column is not numeric. Attempting conversion.")
        meta_df['Label'] = pd.to_numeric(meta_df['Label'], errors='coerce')
        meta_df.dropna(subset=['Label'], inplace=True)
        if meta_df.empty:
            logger.warning("DataFrame empty after handling non-numeric labels.")
            return build_dataset(pd.DataFrame(columns=meta_df.columns), batch_size) # Return empty dataset

    path_tensors = [meta_df[col].values for col in path_cols]
    label_tensor = meta_df['Label'].values.astype(np.float32)
    all_tensors_for_slicing = tuple(path_tensors + [label_tensor])
    
    dataset = tf.data.Dataset.from_tensor_slices(all_tensors_for_slicing)
    dataset = dataset.map(process_input_row, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(meta_df), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    logger.info("Dataset build complete.")
    return dataset

def evaluate_models(
    project: str,
    location: str,
    new_model_gcs_uri: str,
    production_model_uri: str,
    evaluation_data_gcs_dir: str,
    batch_size: int,
    metrics_name: str,
    comparison_threshold: float,
    deploy_decision_output_path: str,
    new_model_metrics_output_path: str,
    production_model_metrics_output_path: str,
):
    """Loads models, evaluates, compares, writes outputs."""
    
    logger.info(f"Starting evaluation with effective INPUT_SHAPE: {INPUT_SHAPE}, NUM_INPUTS: {NUM_INPUTS}")
    logger.info(f"New model GCS URI: {new_model_gcs_uri}")
    logger.info(f"Production model URI: {production_model_uri}")
    logger.info(f"Evaluation data GCS dir: {evaluation_data_gcs_dir}")

    Path(os.path.dirname(deploy_decision_output_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(new_model_metrics_output_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(production_model_metrics_output_path)).mkdir(parents=True, exist_ok=True)

    aip.init(project=project, location=location)

    custom_objects_for_loading = {
        'stack_features_lambda_fn': stack_features_lambda_fn,
        'reduce_sum_lambda_fn': reduce_sum_lambda_fn
    }
    
    production_model_uri = production_model_uri + "classification_model.keras"
    
    new_model_loaded = None
    prod_model_loaded = None
    deploy_new_model = False

    try:
        logger.info("Loading new model candidate...")
        if not new_model_gcs_uri or not tf.io.gfile.exists(new_model_gcs_uri):
             raise FileNotFoundError(f"New model artifact not found at GCS URI: {new_model_gcs_uri}")
        new_model_loaded = tf.keras.models.load_model(new_model_gcs_uri, custom_objects=custom_objects_for_loading)
        logger.info("New model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load new model from {new_model_gcs_uri}: {e}", exc_info=True)
        with open(deploy_decision_output_path, 'w') as f: f.write('false')
        raise

    try:
        logger.info("Attempting to load production model...")
        if production_model_uri and tf.io.gfile.exists(production_model_uri):
            logger.info(f"Production model URI found: {production_model_uri}")
            prod_model_loaded = tf.keras.models.load_model(production_model_uri, custom_objects=custom_objects_for_loading)
            logger.info("Production model loaded successfully.")
        elif production_model_uri:
            logger.warning(f"Production model URI '{production_model_uri}' provided but does not exist. Skipping production model load.")
        else:
            logger.info("No production model URI provided. Skipping production model load.")
    except Exception as e:
        logger.warning(f"Could not load production model from '{production_model_uri}': {e}. Comparison will proceed as if no production model exists.", exc_info=True)

    logger.info("Loading evaluation dataset...")
    eval_meta_df = read_image_paths_and_labels_from_gcs(evaluation_data_gcs_dir)
    if eval_meta_df.empty:
         logger.error(f"Evaluation metadata is empty from {evaluation_data_gcs_dir}. Cannot evaluate.")
         with open(deploy_decision_output_path, 'w') as f: f.write('false')
         raise ValueError("Evaluation data could not be loaded.")
    eval_dataset = build_dataset(eval_meta_df, batch_size, shuffle=False)

    logger.info("Evaluating new model...")
    new_eval_results = new_model_loaded.evaluate(eval_dataset, return_dict=True, verbose=0)
    logger.info(f"New model evaluation results: {new_eval_results}")
    new_metrics_dict = {'metrics': [{'name': f"new_model_{k}", 'numberValue': float(v), 'format': "PERCENTAGE" if "accuracy" in k.lower() else "RAW"} for k, v in new_eval_results.items()]}
    with open(new_model_metrics_output_path, 'w') as f: json.dump(new_metrics_dict, f)

    prod_eval_results = {}
    prod_metrics_dict = {'metrics': []}
    if prod_model_loaded:
        logger.info("Evaluating production model...")
        prod_eval_results = prod_model_loaded.evaluate(eval_dataset, return_dict=True, verbose=0)
        logger.info(f"Production model evaluation results: {prod_eval_results}")
        for k, v in prod_eval_results.items():
            prod_metrics_dict['metrics'].append({'name': f"prod_model_{k}", 'numberValue': float(v), 'format': "PERCENTAGE" if "accuracy" in k.lower() else "RAW"})
    else:
        logger.info("Skipping production model evaluation as it was not loaded.")
        prod_metrics_dict['metrics'].append({'name': f"prod_model_loss", 'numberValue': float('inf'), 'format': "RAW"})
        prod_metrics_dict['metrics'].append({'name': f"prod_model_{metrics_name}", 'numberValue': float('-inf') if metrics_name.lower() != 'loss' else float('inf'), 'format': "RAW" if metrics_name.lower() != 'loss' else "PERCENTAGE"}) # Default to worse possible

    with open(production_model_metrics_output_path, 'w') as f: json.dump(prod_metrics_dict, f)

    new_metric_value = new_eval_results.get(metrics_name)
    if prod_model_loaded and metrics_name in prod_eval_results:
        prod_metric_value = prod_eval_results[metrics_name]
    else: 
        prod_metric_value = float('-inf') if metrics_name.lower() != 'loss' else float('inf')

    if new_metric_value is None:
        logger.error(f"Metrics key '{metrics_name}' not found in new model results. Cannot make deploy decision.")
        deploy_new_model = False
    else:
        is_better = False
        comparison_log_message = ""
        if metrics_name.lower() == 'loss':
            is_better = new_metric_value < (prod_metric_value - comparison_threshold)
            comparison_log_message = f"New loss ({new_metric_value:.4f}) vs Prod loss ({prod_metric_value:.4f}) with threshold ({comparison_threshold}). Target: new < (prod - thresh)."
        else: 
            is_better = new_metric_value > (prod_metric_value + comparison_threshold)
            comparison_log_message = f"New {metrics_name} ({new_metric_value:.4f}) vs Prod {metrics_name} ({prod_metric_value:.4f}) with threshold ({comparison_threshold}). Target: new > (prod + thresh)."

        if is_better:
            logger.info(f"New model IS better. {comparison_log_message}")
            deploy_new_model = True
        else:
            logger.info(f"New model is NOT sufficiently better. {comparison_log_message}")
           

    logger.info(f"Final Deploy decision: {deploy_new_model}")
    with open(deploy_decision_output_path, 'w') as f:
        f.write(str(deploy_new_model).lower())

    logger.info("Evaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a new model against a production model.")
    parser.add_argument('--project', type=str, required=True, help="Google Cloud project ID.")
    parser.add_argument('--location', type=str, required=True, help="Google Cloud location for AI Platform.")
    parser.add_argument('--new-model-uri', type=str, required=True, help='GCS URI of the newly trained model directory (e.g., gs://bucket/path/to/model.keras or gs://bucket/path/to/saved_model_dir/).')
    parser.add_argument('--production-model-uri', type=str, required=False, default="", help='GCS URI of the current production model, or empty if none.')
    parser.add_argument('--evaluation-data-gcs-dir', type=str, required=True, help='GCS path to evaluation data directory (containing labels.csv and images/patches).')
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--metrics-name', type=str, default='accuracy', help="Metric name to use for comparison (e.g., 'accuracy', 'loss').")
    parser.add_argument('--comparison-threshold', type=float, default=0.0, help="Threshold for comparison. For accuracy-like metrics (higher is better), new > prod + thresh. For loss (lower is better), new < prod - thresh.")
    parser.add_argument('--deploy-decision-output-path', type=str, required=True, help='Local path to write boolean deploy decision (true/false).')
    parser.add_argument('--new-model-metrics-output-path', type=str, required=True, help='Local path to write new model metrics JSON.')
    parser.add_argument('--production-model-metrics-output-path', type=str, required=True, help='Local path to write production model metrics JSON.')
    
    parser.add_argument('--patch-height', type=int, default=PATCH_HEIGHT, help="Height of the input image patches.")
    parser.add_argument('--patch-width', type=int, default=PATCH_WIDTH, help="Width of the input image patches.")

    args = parser.parse_args()


    PATCH_HEIGHT = args.patch_height
    PATCH_WIDTH = args.patch_width
    INPUT_SHAPE = (PATCH_HEIGHT, PATCH_WIDTH, 1) 

    logger.info(f"--- Evaluation Script Configuration ---")
    logger.info(f"Effective PATCH_HEIGHT: {PATCH_HEIGHT}")
    logger.info(f"Effective PATCH_WIDTH: {PATCH_WIDTH}")
    logger.info(f"Effective NUM_INPUTS: {NUM_INPUTS}")
    logger.info(f"Effective INPUT_SHAPE: {INPUT_SHAPE}")
    logger.info(f"Metrics for comparison: {args.metrics_name}")
    logger.info(f"Comparison threshold: {args.comparison_threshold}")

    evaluate_models(
        project=args.project,
        location=args.location,
        new_model_gcs_uri=args.new_model_uri,
        production_model_uri=args.production_model_uri,
        evaluation_data_gcs_dir=args.evaluation_data_gcs_dir,
        batch_size=args.batch_size,
        metrics_name=args.metrics_name,
        comparison_threshold=args.comparison_threshold,
        deploy_decision_output_path=args.deploy_decision_output_path,
        new_model_metrics_output_path=args.new_model_metrics_output_path,
        production_model_metrics_output_path=args.production_model_metrics_output_path,
    )
