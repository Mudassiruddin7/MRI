import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2

from skimage.transform import resize
from skimage import img_as_ubyte

def normalize_image(image_slice):
    """
    Normalize image slice to the range [0, 1].
    
    Args:
        image_slice (numpy.ndarray): The input image slice.
        
    Returns:
        numpy.ndarray: Normalized image slice in the range [0, 1].
    """
    image_slice = image_slice.astype(np.float32)
    min_val, max_val = np.min(image_slice), np.max(image_slice)

    if max_val > min_val: 
        image_slice = (image_slice - min_val) / (max_val - min_val)
    else:
        image_slice = np.zeros_like(image_slice)

    return image_slice

def preprocess_image_and_mask(image_array, mask_array, target_shape):
    """
    Preprocess sagittal slices (z, y, x) -> [z:, :, :].
    
    Args:
        image_array (numpy.ndarray): The 3D image volume (z, y, x).
        mask_array (numpy.ndarray): The 3D mask volume (z, y, x).
        target_shape (tuple): Desired output shape (height, width).

    Returns:
        tuple: Preprocessed 3D image and mask volumes (sagittal slices resized to target shape).
    """
    processed_image_slices = []
    processed_mask_slices = []

    mid_idx = image_array.shape[0] // 2  
    start_idx = max(0, mid_idx - 2)  
    end_idx = min(image_array.shape[0], mid_idx + 3)  

    for i in range(start_idx, end_idx): 
        image_slice = resize(image_array[i, :, :], target_shape, mode="reflect", order=3, preserve_range=True,anti_aliasing=True) # order = bicubic 
        mask_slice = resize(mask_array[i, :, :], target_shape, mode="reflect", order=0, preserve_range=True) # order = nearest-neighbor
        
        
        image_slice = normalize_image(image_slice) # return 0.0-1.0

        image_slice = img_as_ubyte(image_slice)         #0-255 uint8
        # image_slice=cv2.equalizeHist(image_slice) worse results

        mask_slice = mask_slice.astype(np.uint8)

        processed_image_slices.append(image_slice)
        processed_mask_slices.append(mask_slice)

    processed_image = np.stack(processed_image_slices, axis=0)
    processed_mask = np.stack(processed_mask_slices, axis=0)

    return processed_image, processed_mask

def slices_generator(base_dir, target_shape=(512, 512)):
    """
    Loads, preprocesses, and extracts sagittal slices from .mha files.

    Args:
        base_dir (str): Base directory containing 'images/images/' and 'masks/masks/' subdirectories.
        target_shape (tuple): Desired height and width for preprocessing.

    Yields:
        tuple: (original_image, original_mask, preprocessed_image, preprocessed_mask, spacing),
               where:
               - original_image, original_mask: Original 3D arrays (x, y, z) or (z, y, x).
               - preprocessed_image, preprocessed_mask: Preprocessed sagittal slices.
               - spacing: Original voxel spacing.
    """
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".mha")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".mha")])


    for img_file, mask_file in zip(image_files, mask_files):

        if "t2" not in img_file or "SPACE" in img_file:  
            continue  

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        image_array = sitk.GetArrayFromImage(image)

        mask_array = sitk.GetArrayFromImage(mask)  
        spacing = image.GetSpacing()  
        direction = image.GetDirection() 

        third_column = (direction[2], direction[5], direction[8])

        if np.allclose(third_column, (0, 0, 1), atol=1e-3):  
            image_array = np.transpose(image_array, (2, 1, 0))  
            mask_array = np.transpose(mask_array, (2, 1, 0))

            rotated_image = np.array([cv2.rotate(image_array[z], cv2.ROTATE_90_COUNTERCLOCKWISE) for z in range(image_array.shape[0])])
            rotated_mask = np.array([cv2.rotate(mask_array[z], cv2.ROTATE_90_COUNTERCLOCKWISE) for z in range(mask_array.shape[0])])

            image_array = rotated_image
            mask_array = rotated_mask

        preprocessed_image, preprocessed_mask = preprocess_image_and_mask(image_array, mask_array, target_shape)

        yield image_array, mask_array, preprocessed_image, preprocessed_mask, spacing
        

def write_tfrecord(image_slices, mask_slices, output_file):
    """Writes sagittal slices to a TFRecord file."""
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_slice, mask_slice in zip(image_slices, mask_slices):

            image_slice = np.expand_dims(image_slice, axis=-1)  # Add channel dimension
            mask_slice = np.expand_dims(mask_slice, axis=-1)    # Add channel dimension
            print(image_slice.shape,mask_slice.shape)
            
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image_slice).numpy()])),
                'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(mask_slice).numpy()])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def save_preprocessed_data_to_tfrecord(base_dir, target_shape=(512, 512), output_dir='tfrecords'):
    """Preprocess sagittal slices and save to TFRecord files."""
    os.makedirs(output_dir, exist_ok=True)
    generator = slices_generator(base_dir, target_shape)

    for index, (_, _, preprocessed_image, preprocessed_mask, _) in enumerate(generator):
        output_file = os.path.join(output_dir, f"dataset_{index + 1}.tfrecord")
        image_slices = [preprocessed_image[z, :, :] for z in range(preprocessed_image.shape[0])]
        mask_slices = [preprocessed_mask[z, :, :] for z in range(preprocessed_mask.shape[0])]
        
        write_tfrecord(image_slices, mask_slices, output_file)

    print(f"Preprocessed data saved to {output_dir} as TFRecord files.")

def parse_tfrecord(example_proto, lower=201, upper=205):
    """
    Parses a single example from a TFRecord file and converts the mask to binary.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.io.decode_png(parsed_example['image'], channels=1)
    mask = tf.io.decode_png(parsed_example['mask'], channels=1)

    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.cast(mask, tf.uint8)
    mask_binary = tf.where((mask >= lower) & (mask <= upper), 1, 0)

    return image, mask_binary

def crop_image_and_mask(image, mask):
    cropped_image = image[96:, 48:-48, :]
    cropped_mask = mask[96:, 48:-48, :]
    return cropped_image, cropped_mask
    
def load_tfrecord_dataset(tfrecord_dir, batch_size, buffer_size=100):
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(crop_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def split_dataset(dataset, train_samples, val_samples, batch_size):
    """
    Splits a dataset into training, validation, and test sets.
    """
    train_batches = train_samples // batch_size
    val_batches = val_samples // batch_size

    train_dataset = dataset.take(train_batches)
    val_dataset = dataset.skip(train_batches).take(val_batches)
    test_dataset = dataset.skip(train_batches + val_batches)

    return train_dataset, val_dataset, test_dataset