import tensorflow as tf
import os
import argparse
import logging
import sys, traceback

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Lambda, Multiply,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from google.cloud.aiplatform.training_utils import cloud_profiler

from preprocessing_utils import load_tfrecord_dataset, split_dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  



def count_samples_in_tfrecord(tfrecord_dir):
    """Counts the total number of samples in the TFRecords."""
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    return sum(1 for _ in tf.data.TFRecordDataset(tfrecord_files))

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

@tf.keras.utils.register_keras_serializable()
def combo_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    return bce + dsc

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    return (2. * intersection + epsilon) / (union + epsilon)

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    return (intersection + epsilon) / (union + epsilon)

@tf.keras.utils.register_keras_serializable() 
def channel_avg_pool(x):
    """Calculates the average across the channel axis."""
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable() 
def channel_max_pool(x):
    """Calculates the maximum across the channel axis."""
    return tf.reduce_max(x, axis=-1, keepdims=True)

def spatial_attention(input_feature, name_prefix): 
    """
    Applies spatial attention mechanism. Uses a name_prefix
    to ensure unique layer names when used multiple times.
    """
    avg_pool = Lambda(channel_avg_pool, name=f'{name_prefix}_channel_avg_pool')(input_feature)
    max_pool = Lambda(channel_max_pool, name=f'{name_prefix}_channel_max_pool')(input_feature)

    concat = Concatenate(axis=-1, name=f'{name_prefix}_concat')([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid',
                       name=f'{name_prefix}_conv')(concat)
    output = Multiply(name=f'{name_prefix}_multiply')([input_feature, attention])
    return output

def build_binary_model(input_shape=(512, 512, 1)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    c1 = Dropout(0.3)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (5, 5), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (5, 5), activation='relu', padding='same')(c2)
    c2 = Dropout(0.3)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (7, 7), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (7, 7), activation='relu', padding='same')(c3)
    c3 = Dropout(0.3)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (9, 9), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (9, 9), activation='relu', padding='same')(c4)
    c4 = Dropout(0.4)(c4) 

    u5 = UpSampling2D((2, 2))(c4)
    c3_att = spatial_attention(c3, name_prefix='sa_c3') 
    u5 = Concatenate()([u5, c3_att])
    c5 = Conv2D(64, (7, 7), activation='relu', padding='same')(u5)
    c5 = Conv2D(64, (7, 7), activation='relu', padding='same')(c5)
    c5 = Dropout(0.3)(c5)

    u6 = UpSampling2D((2, 2))(c5)
    c2_att = spatial_attention(c2, name_prefix='sa_c2') 
    u6 = Concatenate()([u6, c2_att])
    c6 = Conv2D(32, (5, 5), activation='relu', padding='same')(u6)
    c6 = Conv2D(32, (5, 5), activation='relu', padding='same')(c6)
    c6 = Dropout(0.3)(c6)

    u7 = UpSampling2D((2, 2))(c6)
    c1_att = spatial_attention(c1, name_prefix='sa_c1') 
    u7 = Concatenate()([u7, c1_att])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.3)(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=combo_loss, metrics=[dice_coefficient, 'accuracy'])

    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a U-Net model for MRI segmentation.")
    parser.add_argument("--tfrecord_dir", required=True, help="Path to the directory containing TFRecord files.")
    parser.add_argument("--model_output_dir", required=True, help="Path to save the trained model.",default=os.environ['AIP_MODEL_DIR'])
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    args = parser.parse_args()
    try:
        cloud_profiler.init()
    except:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        print("*** Unexpected:", ex_type.__name__, ex_value)
        traceback.print_tb(ex_traceback, limit=10, file=sys.stdout)
   
    batch_size = args.batch_size
    tfrecord_dir = args.tfrecord_dir
    model_output_dir = args.model_output_dir

    
    full_dataset = load_tfrecord_dataset(tfrecord_dir, batch_size=batch_size)
    logger.info("Full dataset loaded.")

    total_samples = count_samples_in_tfrecord(tfrecord_dir)
    if total_samples == 0:
        logger.error(f"No samples found in TFRecords at {tfrecord_dir}. Exiting.")
        sys.exit(1)

    
    total_samples = count_samples_in_tfrecord(tfrecord_dir)
    logger.info(f"dataset count {total_samples}")

    train_samples = int(0.8 * total_samples)  
    test_samples = int(0.1 * total_samples)  
    val_samples = total_samples - (train_samples + test_samples) 

 
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_samples,val_samples, batch_size)
    train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.repeat().prefetch(tf.data.AUTOTUNE)

   
    model = build_binary_model(input_shape=(416, 416, 1))
    model.summary()

    tensorboard_log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    if tensorboard_log_dir:
        logger.info(f"TensorBoard log directory found: {tensorboard_log_dir}. Enabling TensorBoard callback.")
        
    else:
        logger.warning("AIP_TENSORBOARD_LOG_DIR environment variable not set. TensorBoard callback disabled.")
        logger.warning("Ensure the 'tensorboard' parameter is set correctly in the CustomTrainingJobOp definition.")

    print('Setting up the TensorBoard callback ...')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1)
    
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size


    
    local_checkpoint_filepath = "segmentation_model_best.keras"
    checkpoint = ModelCheckpoint(local_checkpoint_filepath, save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=2, restore_best_weights=True, verbose=1)

    logger.info("Starting model training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stopping,tensorboard_callback],
        verbose=1,
    )
    logger.info("Starting model evaluating...")
    results = model.evaluate(test_dataset, verbose=1)

    test_loss = results[0]  
    test_acc = results[1]   
    test_dice_coefficient = results[2]  

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Dice Coefficient: {test_dice_coefficient}")
    model_output_dir = args.model_output_dir 
    
    if not model_output_dir.endswith('/'):
        model_output_dir += '/'
    final_model_gcs_path = f"{model_output_dir}segmentation_model_best.keras"
    

    logger.info(f"Saving final model directly to GCS: {final_model_gcs_path}") 
    try:
        model.save(final_model_gcs_path) 
        logger.info(f"Model successfully saved to {final_model_gcs_path}") 
    except Exception as e:
        logger.error(f"Error saving final model to GCS ({final_model_gcs_path}): {e}", exc_info=True)
        raise 
        

if __name__ == "__main__":
    main()