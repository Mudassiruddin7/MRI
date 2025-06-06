import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras_tuner as kt
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PATCH_HEIGHT = None 
PATCH_WIDTH = None  
INPUT_SHAPE = None  
NUM_INPUTS = 5
EXECUTIONS_PER_TRIAL = 2
TUNER_PATIENCE = 10
FINAL_PATIENCE = 10
BATCH_SIZE = 32
FINAL_EPOCHS = 2
TUNER_EPOCHS = 5
MAX_TUNER_TRIALS = 2

def read_image_paths_and_labels_from_gcs(gcs_dir_path):
    """
    Reads labels.csv and lists corresponding jpg files from a GCS directory.
    Expects jpg files directly in gcs_dir_path or in a 'patches/' subdirectory.
    Returns a DataFrame with columns: ID, Label, IMG1, ..., IMG5 (full GCS paths).
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

        # find image files check base path and 'patches/' subdirectory
        image_files = []
        base_jpgs = tf.io.gfile.glob(os.path.join(gcs_dir_path, "*.jpg"))
        patches_jpgs = tf.io.gfile.glob(os.path.join(gcs_dir_path, "patches/*.jpg"))
        if base_jpgs:
            image_files.extend(base_jpgs)
            logger.info(f"Found {len(base_jpgs)} jpg files in base directory.")
        if patches_jpgs:
            image_files.extend(patches_jpgs)
            logger.info(f"Found {len(patches_jpgs)} jpg files in patches/ subdirectory.")

        if not image_files:
            logger.warning(f"No .jpg files found in {gcs_dir_path} or its patches/ subdirectory.")
            return pd.DataFrame()

        # group image paths by their unique id  RequestID_boxN
        id_groups = {}
        for img_path in image_files:
            img_file = os.path.basename(img_path)
            try:
                id_num = '_'.join(img_file.split('_')[:-1]) 
                img_part = img_file.split('_')[-1]
                img_idx_str = img_part.split('.')[0].replace('IMG', '')
                img_idx = int(img_idx_str)
                if not (1 <= img_idx <= 5): continue # ignore if not img1-5

                if id_num not in id_groups: id_groups[id_num] = {}
                id_groups[id_num][f"IMG{img_idx}"] = img_path # store full gcs path

            except (IndexError, ValueError):
                logger.warning(f"Skipping file with unexpected format: {img_file}")
                continue

        data = []
        required_imgs = [f"IMG{i}" for i in range(1, 6)]
        for id_num, images_dict in id_groups.items():
            if all(key in images_dict for key in required_imgs):

                label_key_filename = os.path.basename(images_dict["IMG1"]) 
                label = label_dict.get(label_key_filename)

                if label is not None:
                    sorted_image_paths = [images_dict[f"IMG{i}"] for i in range(1, 6)]
                    row = {'ID': id_num, 'Label': label}
                    row.update({f"IMG{i}": path for i, path in enumerate(sorted_image_paths, 1)})
                    data.append(row)


        df_out = pd.DataFrame(data)
        for col in ['ID', 'Label'] + [f"IMG{i}" for i in range(1, 6)]:
            if col not in df_out.columns: df_out[col] = None 

        # drop rows with missing paths
        path_cols = [f"IMG{i}" for i in range(1, 6)]
        initial_len = len(df_out)
        df_out.dropna(subset=path_cols + ['Label'], inplace=True)
        if len(df_out) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df_out)} rows due to missing paths or labels.")

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
        image = tf.image.decode_jpeg(image, channels=1) # grayscale
        image = tf.image.resize(image, [PATCH_HEIGHT, PATCH_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0 
        image.set_shape(INPUT_SHAPE) 
        return image
    except Exception as e:
        tf.print(f"Error loading image {path_tensor}: {e}", output_stream=sys.stderr)
        return tf.zeros(INPUT_SHAPE, dtype=tf.float32) 

@tf.function
def process_input_row(img1_path, img2_path, img3_path, img4_path, img5_path, label):
    """Loads the 5 images corresponding to one sample."""
    img1 = load_and_process_image(img1_path)
    img2 = load_and_process_image(img2_path)
    img3 = load_and_process_image(img3_path)
    img4 = load_and_process_image(img4_path)
    img5 = load_and_process_image(img5_path)
    return (img1, img2, img3, img4, img5), tf.cast(label, tf.float32)

def build_dataset(meta_df, batch_size, shuffle=False):
    """Builds tf.data.Dataset from metadata DataFrame."""
    global INPUT_SHAPE 
    if meta_df.empty:
        logger.warning("Input DataFrame is empty. Returning empty dataset structure.")
        tensor_specs = (
                           (tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),) * NUM_INPUTS,
                           tf.TensorSpec(shape=(), dtype=tf.float32)
                       )
        return tf.data.Dataset.from_generator(lambda: iter([]), output_signature=tensor_specs).batch(batch_size)

    path_cols = [f'IMG{i}' for i in range(1, 6)]
    if not all(col in meta_df.columns for col in path_cols + ['Label']):
         raise ValueError(f"DataFrame is missing required columns. Need: {path_cols + ['Label']}. Found: {meta_df.columns}")

    if not pd.api.types.is_numeric_dtype(meta_df['Label']):
        logger.warning("'Label' column is not numeric. Attempting conversion.")
        meta_df['Label'] = pd.to_numeric(meta_df['Label'], errors='coerce')
        meta_df.dropna(subset=['Label'], inplace=True)
        if meta_df.empty: 
            logger.warning("DataFrame empty after handling non-numeric labels.")
            return build_dataset(meta_df, batch_size) # return empty dataset

    dataset = tf.data.Dataset.from_tensor_slices((
        meta_df['IMG1'].values, 
        meta_df['IMG2'].values,
        meta_df['IMG3'].values,
        meta_df['IMG4'].values,
        meta_df['IMG5'].values,
        meta_df['Label'].values.astype(np.float32)
    ))


    dataset = dataset.map(process_input_row, num_parallel_calls=tf.data.AUTOTUNE)


    # in-memory cache, use .cache(filename='path') for disk cache test which is more suitable
    logger.info("Adding cache() to dataset pipeline.")
    dataset = dataset.cache()


    # if we decided to add augmentation it would be here before the shuffle
    if shuffle:
        logger.info(f"Adding shuffle(buffer_size={len(meta_df)}) to dataset pipeline.")
        dataset = dataset.shuffle(buffer_size=len(meta_df), reshuffle_each_iteration=True)


    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    logger.info("Dataset build complete with batching and prefetching.")
    return dataset

@tf.keras.utils.register_keras_serializable()
def stack_features_lambda_fn(x_list):
    """
    Stacks a list of tensors along axis 1.
    Args:
        x_list: A list of tensors.
    Returns:
        A single tensor with the input tensors stacked.
    """
    return tf.stack(x_list, axis=1)

@tf.keras.utils.register_keras_serializable()
def reduce_sum_lambda_fn(x):
    """
    Reduces the sum of a tensor along axis 1.
    Args:
        x: Input tensor.
    Returns:
        Tensor with sum reduced along axis 1.
    """
    return tf.reduce_sum(x, axis=1)


def build_model_for_tuner(hp):
    """
    Builds a Keras model with hyperparameter tuning.
    Uses Lambda layers for stacking and sum reduction.
    Relies on global INPUT_SHAPE and NUM_INPUTS.
    """
    global INPUT_SHAPE, NUM_INPUTS 

    if INPUT_SHAPE is None or NUM_INPUTS is None:
        raise ValueError("INPUT_SHAPE or NUM_INPUTS is not set globally before calling build_model_for_tuner.")

    def resnet_block(input_tensor, filters, kernel_size=(3, 3), activation='relu', strides=(1, 1), block_prefix=""):
        """
        A ResNet-style block.
        Args:
            input_tensor: Input tensor to the ResNet block.
            filters: Number of filters in the convolutional layers.
            kernel_size: Kernel size for convolutions.
            activation: Activation function to use.
            strides: Strides for the first convolutional layer.
            block_prefix: A string prefix for layer names to ensure uniqueness.
        Returns:
            Output tensor of the ResNet block.
        """
        x_main = input_tensor
        base_name = f"{block_prefix}_{input_tensor.name.split(':')[0].replace('/', '_')}"

        x_main = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation=None, name=f"{base_name}_main_conv1_{filters}")(x_main)
        x_main = tf.keras.layers.BatchNormalization(name=f"{base_name}_main_bn1_{filters}")(x_main)
        x_main = tf.keras.layers.Activation(activation, name=f"{base_name}_main_act1_{filters}")(x_main)

        x_main = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same', activation=None, name=f"{base_name}_main_conv2_{filters}")(x_main)
        x_main = tf.keras.layers.BatchNormalization(name=f"{base_name}_main_bn2_{filters}")(x_main)

        shortcut = input_tensor
        input_channels = tf.keras.backend.int_shape(input_tensor)[-1]

        is_strides_gt_1 = False
        if isinstance(strides, tuple): is_strides_gt_1 = strides != (1,1)
        elif isinstance(strides, int): is_strides_gt_1 = strides != 1

        if is_strides_gt_1 or input_channels != filters:
            shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', activation=None, name=f"{base_name}_shortcut_conv_{filters}")(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(name=f"{base_name}_shortcut_bn_{filters}")(shortcut)

        output_tensor = tf.keras.layers.Add(name=f"{base_name}_add_{filters}")([shortcut, x_main])
        output_tensor = tf.keras.layers.Activation(activation, name=f"{base_name}_output_act_{filters}")(output_tensor)
        return output_tensor

    def conv_block_with_residuals_hp(input_tensor, hp_filters_1, hp_filters_2, hp_filters_3, hp_spatial_dropout_rate, branch_prefix=""):
        """
        Convolutional block with ResNet-style connections and hyperparameter-tuned filters and dropout.
        Args:
            input_tensor: Input tensor to this convolutional block.
            hp_filters_1, hp_filters_2, hp_filters_3: Hyperparameters for filter counts.
            hp_spatial_dropout_rate: Hyperparameter for spatial dropout.
            branch_prefix: A string prefix for layer names within this branch.
        Returns:
            Output tensor of the block.
        """
        x = resnet_block(input_tensor, filters=hp_filters_1, strides=(1,1), block_prefix=f"{branch_prefix}_res1")
        x = tf.keras.layers.SpatialDropout2D(hp_spatial_dropout_rate, name=f"{branch_prefix}_spatial_dropout1")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"{branch_prefix}_maxpool1")(x)

        x = resnet_block(x, filters=hp_filters_2, strides=(1,1), block_prefix=f"{branch_prefix}_res2")
        x = tf.keras.layers.SpatialDropout2D(hp_spatial_dropout_rate, name=f"{branch_prefix}_spatial_dropout2")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"{branch_prefix}_maxpool2")(x)

        x = resnet_block(x, filters=hp_filters_3, strides=(1,1), block_prefix=f"{branch_prefix}_res3")
        x = tf.keras.layers.SpatialDropout2D(hp_spatial_dropout_rate, name=f"{branch_prefix}_spatial_dropout3")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"{branch_prefix}_maxpool3")(x)

        x = tf.keras.layers.BatchNormalization(name=f"{branch_prefix}_bn_final")(x)
        x = tf.keras.layers.GlobalMaxPooling2D(name=f"{branch_prefix}_globalmaxpool")(x)
        return x

    inputs = [tf.keras.Input(INPUT_SHAPE, name=f'input_{i+1}') for i in range(NUM_INPUTS)]

    hp_resnet_f1 = hp.Int('resnet_filters_1', min_value=16, max_value=32, step=8, default=16)
    hp_resnet_f2 = hp.Int('resnet_filters_2', min_value=32, max_value=64, step=16, default=32)
    hp_resnet_f3 = hp.Int('resnet_filters_3', min_value=64, max_value=128, step=16, default=64)
    hp_spatial_dropout = hp.Float('spatial_dropout_resnet', min_value=0.1, max_value=0.4, step=0.05, default=0.2)

    processed_features = []
    for i, inp in enumerate(inputs):
        feature_branch = conv_block_with_residuals_hp(
            inp,
            hp_filters_1=hp_resnet_f1,
            hp_filters_2=hp_resnet_f2,
            hp_filters_3=hp_resnet_f3,
            hp_spatial_dropout_rate=hp_spatial_dropout,
            branch_prefix=f"input_{i+1}_conv_res_branch" 
        )
        processed_features.append(feature_branch)

    # stack features using lambda layer 
    # only stack if there's more than one feature branch otherwise just take the single feature.
    if len(processed_features) > 1:
        stacked_features = tf.keras.layers.Lambda(
            stack_features_lambda_fn,
            name='stack_features_lambda'
        )(processed_features)
    elif len(processed_features) == 1:
       
        stacked_features = processed_features[0]
        stacked_features = tf.keras.layers.Reshape((1, -1), name='reshape_single_input_for_attention')(stacked_features)
    else:
        raise ValueError("Processed_features list is empty. NUM_INPUTS must be at least 1.")


    # attention
    hp_attention_activation = hp.Choice('attention_activation', values=['tanh', 'relu', 'sigmoid'], default='tanh')
    attention_weights = tf.keras.layers.Dense(1, activation=hp_attention_activation, name='attention_dense')(stacked_features)
    attention_weights = tf.keras.layers.Softmax(axis=1, name='attention_softmax')(attention_weights) # axis=1 is the stream/input axis
    attended_features_mult = tf.keras.layers.Multiply(name='attention_multiply')([stacked_features, attention_weights])

    
    attended_features_sum = tf.keras.layers.Lambda(
        reduce_sum_lambda_fn, 
        name='attention_reduce_sum_lambda'
    )(attended_features_mult)


    x = tf.keras.layers.BatchNormalization(name='post_attention_bn')(attended_features_sum)

    hp_dense1_units = hp.Int('dense1_units', min_value=32, max_value=128, step=32, default=64)
    hp_dense_activation = hp.Choice('dense_activation', values=['relu', 'elu', 'tanh'], default='relu')
    hp_dense_dropout_1 = hp.Float('dense_dropout_1', min_value=0.2, max_value=0.5, step=0.05, default=0.3)
    hp_l2_rate = hp.Float('l2_rate', min_value=1e-5, max_value=1e-3, sampling='log', default=1e-4)

    x = tf.keras.layers.Dense(
        units=hp_dense1_units,
        activation=hp_dense_activation,
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2_rate),
        name='dense_1'
    )(x)
    x = tf.keras.layers.Dropout(hp_dense_dropout_1, name='dropout_1')(x)

    hp_dense2_units = hp.Int('dense2_units', min_value=16, max_value=64, step=16, default=32)
    hp_dense_dropout_2 = hp.Float('dense_dropout_2', min_value=0.2, max_value=0.5, step=0.05, default=0.3)

    x = tf.keras.layers.Dense(
        units=hp_dense2_units,
        activation=hp_dense_activation,
        kernel_regularizer=tf.keras.regularizers.l2(hp_l2_rate),
        name='dense_2'
    )(x)
    x = tf.keras.layers.Dropout(hp_dense_dropout_2, name='dropout_2')(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_classification_model(
    original_train_dir, original_val_dir, original_test_dir,
    corrected_train_dir, 
    output_model_dir, 
    checkpoint_dir, 
    epochs=50,
    tuner_epochs=15,
    max_tuner_trials=15,
    batch_size=32
    ):
    """Loads data, runs tuner, trains best model, saves output."""

    logger.info("--- Loading and Combining Metadata ---")
    orig_train_df = read_image_paths_and_labels_from_gcs(original_train_dir)
    orig_val_df = read_image_paths_and_labels_from_gcs(original_val_dir)
    orig_test_df = read_image_paths_and_labels_from_gcs(original_test_dir)
    corrected_train_dir = corrected_train_dir.replace("gs://", "/gcs/")
    corr_train_df = read_image_paths_and_labels_from_gcs(corrected_train_dir)
    checkpoint_dir = checkpoint_dir.replace("gs://","/gcs/")
    if not corr_train_df.empty:
        logger.info(f"Combining {len(orig_train_df)} original train samples with {len(corr_train_df)} corrected samples.")
        train_meta_df = pd.concat([orig_train_df, corr_train_df], ignore_index=True)
    else:
        logger.info("No corrected training data found or loaded.")
        train_meta_df = orig_train_df

    val_meta_df = orig_val_df
    test_meta_df = orig_test_df

    logger.info(f"Final Train samples: {len(train_meta_df)}")
    logger.info(f"Final Val samples: {len(val_meta_df)}")
    logger.info(f"Final Test samples: {len(test_meta_df)}")

    if train_meta_df.empty or val_meta_df.empty:
        logger.error("Training or validation metadata is empty after loading/combining. Cannot train.")
        return

    logger.info(f"Train Label Distribution:\n{train_meta_df['Label'].value_counts(normalize=True)}") 
    logger.info(f"Validation Label Distribution:\n{val_meta_df['Label'].value_counts(normalize=True)}") 

    logger.info("\n--- Building Datasets ---")
    try:
        train_dataset = build_dataset(train_meta_df, batch_size, shuffle=True)
        val_dataset = build_dataset(val_meta_df, batch_size, shuffle=False)
        test_dataset = build_dataset(test_meta_df, batch_size, shuffle=False)
        logger.info("Datasets built successfully.")
    except Exception as e:
        logger.error(f"Error building datasets: {e}", exc_info=True)
        return

    logger.info("\n--- Setting up Keras Tuner ---")

    tuner_dir = os.path.join(checkpoint_dir, 'keras_tuner')
    tuner_project_name = 'discai_classification_tuning'
    logger.info(f"Keras Tuner directory: {tuner_dir}/{tuner_project_name}")

    tuner = kt.BayesianOptimization(
        hypermodel=build_model_for_tuner,
        objective='val_accuracy',
        max_trials=max_tuner_trials,
        executions_per_trial=EXECUTIONS_PER_TRIAL, 
        directory=tuner_dir,
        project_name=tuner_project_name,
        overwrite=True 
    )
    tuner.search_space_summary()

    search_early_stopping = EarlyStopping(monitor='val_loss', patience=TUNER_PATIENCE, verbose=1)

    logger.info("\n--- Starting Hyperparameter Search ---")
    tuner.search(
        train_dataset,
        epochs=tuner_epochs,
        validation_data=val_dataset,
        callbacks=[search_early_stopping]
    )
    logger.info("\nHyperparameter search finished.")

    try:
        best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
        if not best_hps_list:
             logger.error("Keras Tuner could not find any best hyperparameters. This might be due to all trials failing or other issues.")
             return
        best_hps = best_hps_list[0]

        logger.info("\n--- Best Hyperparameters Found ---")
        for param, value in best_hps.values.items(): logger.info(f"{param}: {value}")

        logger.info("\nBuilding the best model...")
        best_model = tuner.hypermodel.build(best_hps)
        best_model.summary(print_fn=logger.info)

        logger.info("\n--- Training the Best Model ---")
        final_early_stopping = EarlyStopping(monitor='val_loss', patience=FINAL_PATIENCE, verbose=1, restore_best_weights=True)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'final_training_ckpt_{epoch:02d}.weights.h5') # Save weights only is often safer
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

        history = best_model.fit(
            train_dataset,
            epochs=epochs, 
            validation_data=val_dataset,
            callbacks=[final_early_stopping, model_checkpoint_callback]
        )
        logger.info("Best model training finished.")

        logger.info("\n--- Evaluating Best Model ---")
        if test_meta_df.empty:
             logger.warning("Cannot evaluate: Test metadata is empty.")
        elif test_dataset.cardinality().numpy() == 0: 
            logger.warning("Cannot evaluate: Test dataset is empty after processing.")
        else:
            test_loss, test_accuracy = best_model.evaluate(test_dataset)
            logger.info(f'Test Loss: {test_loss}')
            logger.info(f'Test Accuracy: {test_accuracy}')

        logger.info("\n--- Saving Final Model ---")
        
        if not output_model_dir.endswith('/'):
            output_model_dir += '/'
        model_save_path = f"{output_model_dir}classification_model.keras" 
        best_model.save(model_save_path) 
        logger.info(f"Final model saved to: {model_save_path}")


    except (ValueError, IndexError) as e:
        logger.error(f"\nError during post-search tuning/training: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during training/evaluation: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--original-train-dir', type=str, required=True, help='GCS path to original processed training data directory')
    parser.add_argument('--original-val-dir', type=str, required=True, help='GCS path to original processed validation data directory')
    parser.add_argument('--original-test-dir', type=str, required=True, help='GCS path to original processed test data directory')
    parser.add_argument('--corrected-train-dir', type=str, required=True, help='GCS path to corrected processed training data directory (output of preprocess component)')
    # model/checkpoint paths
    parser.add_argument('--model-output-gcs-dir', type=str, default=os.environ.get('AIP_MODEL_DIR'), help='GCS path to save final model')
    parser.add_argument('--checkpoint-gcs-dir', type=str, default=os.environ.get('AIP_CHECKPOINT_DIR'), help='GCS path for checkpoints and tuner results')
    # hyperparameters
    parser.add_argument('--patch-height', type=int, default=30, help='Height of the input image patches.') # New Argument
    parser.add_argument('--patch-width', type=int, default=50, help='Width of the input image patches.')   # New Argument
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=FINAL_EPOCHS, help="Epochs for final model training.")
    parser.add_argument('--tuner-epochs', type=int, default=TUNER_EPOCHS, help="Max epochs per Keras Tuner trial.")
    parser.add_argument('--max-tuner-trials', type=int, default=MAX_TUNER_TRIALS)
    
    parser.add_argument('--project-id', type=str, default=os.environ.get('CLOUD_ML_PROJECT_ID'))
    parser.add_argument('--location', type=str, default=os.environ.get('CLOUD_ML_REGION'))


    args = parser.parse_args()

    PATCH_HEIGHT = args.patch_height
    PATCH_WIDTH = args.patch_width
    INPUT_SHAPE = (PATCH_HEIGHT, PATCH_WIDTH, 1)
    logger.info(f"Using PATCH_HEIGHT: {PATCH_HEIGHT}, PATCH_WIDTH: {PATCH_WIDTH}, INPUT_SHAPE: {INPUT_SHAPE}")


    if not args.model_output_gcs_dir:
        raise ValueError("AIP_MODEL_DIR environment variable not set or is empty. Required for saving model.")
    if not args.checkpoint_gcs_dir:
        
        args.checkpoint_gcs_dir = os.path.join(args.model_output_gcs_dir, 'checkpoints')
        logger.warning(f"AIP_CHECKPOINT_DIR not set or is empty, using default: {args.checkpoint_gcs_dir}")
    
    try:
        if not tf.io.gfile.exists(args.checkpoint_gcs_dir):
            tf.io.gfile.makedirs(args.checkpoint_gcs_dir)
            logger.info(f"Created checkpoint directory: {args.checkpoint_gcs_dir}")
    except Exception as e:
        logger.error(f"Failed to create or access checkpoint directory {args.checkpoint_gcs_dir}: {e}")
        sys.exit(1) 


    

    train_classification_model(
        original_train_dir=args.original_train_dir,
        original_val_dir=args.original_val_dir,
        original_test_dir=args.original_test_dir,
        corrected_train_dir=args.corrected_train_dir,
        output_model_dir=args.model_output_gcs_dir,
        checkpoint_dir=args.checkpoint_gcs_dir,
        epochs=args.epochs,
        tuner_epochs=args.tuner_epochs,
        max_tuner_trials=args.max_tuner_trials,
        batch_size=args.batch_size
    )