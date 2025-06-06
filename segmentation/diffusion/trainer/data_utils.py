import tensorflow as tf
from . import config
import os

def parse_tfrecord_raw(example_proto):
    """Parses TFRecord and decodes PNGs without further processing."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image_raw = tf.io.decode_png(parsed_example['image'], channels=config.IMG_CHANNELS_MRI, dtype=tf.uint8)
    mask_raw = tf.io.decode_png(parsed_example['mask'], channels=config.IMG_CHANNELS_MASK, dtype=tf.uint8)
    return image_raw, mask_raw

def parse_tfrecord_for_diffusion(example_proto, img_height, img_width, mask_lower, mask_upper):
    """
    Parses TFRecord, resizes, normalizes image to [-1, 1],
    and converts mask to binary [-1, 1] based on value range.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_png(parsed_example['image'], channels=config.IMG_CHANNELS_MRI, dtype=tf.uint8)
    mask = tf.io.decode_png(parsed_example['mask'], channels=config.IMG_CHANNELS_MASK, dtype=tf.uint8) 

    image = tf.image.resize(image, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = image[96:, 48:-48, :]
    mask = mask[96:, 48:-48, :]

    image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, [config.IMG_HEIGHT, config.IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32)
    image = (image / 255.0) * 2.0 - 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)

    mask = tf.cast(mask, tf.int32) 

    mask_binary = tf.where((mask >= mask_lower) & (mask <= mask_upper), 1.0, -1.0)
    mask_binary = tf.cast(mask_binary, tf.float32)

    image.set_shape([config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MRI])
    mask_binary.set_shape([config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MASK])

    return image, mask_binary

def load_tfrecord_dataset(tfrecord_dir, batch_size, buffer_size, is_train=True, use_preprocessing=True):
    """Loads TFRecord dataset, optionally applying preprocessing."""
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in directory: {tfrecord_dir}")

    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

    if use_preprocessing:
        parser_fn = lambda x: parse_tfrecord_for_diffusion(
            x, img_height=config.IMG_HEIGHT_R, img_width=config.IMG_WIDTH_R,
            mask_lower=config.MASK_TARGET_LOWER, mask_upper=config.MASK_TARGET_UPPER
        )
    else:
        parser_fn = parse_tfrecord_raw

    dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset