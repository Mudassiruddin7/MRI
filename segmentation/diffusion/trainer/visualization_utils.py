import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def visualize_sample(image, mask, title_prefix="Sample"):
    """Visualizes an image and its corresponding mask side-by-side."""
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()

    image_shape = image.shape
    image_dtype = image.dtype
    mask_shape = mask.shape
    mask_dtype = mask.dtype

    if image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    img_cmap = 'gray'
    img_vmin, img_vmax = None, None
    if image_dtype == np.float32 or image_dtype == np.float64:
        img_vmin, img_vmax = -1.0, 1.0 

    im = plt.imshow(image, cmap=img_cmap, vmin=img_vmin, vmax=img_vmax)
    plt.title(f"{title_prefix} - Image\nShape: {image_shape}\nDtype: {image_dtype}")
    plt.colorbar(im, label='Pixel Intensity')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    mask_cmap = 'viridis' 
    mask_vmin, mask_vmax = None, None
    if mask_dtype == np.float32 or mask_dtype == np.float64:
        mask_cmap = 'gray'
        mask_vmin, mask_vmax = -1.0, 1.0
    elif mask_dtype == np.uint8 or mask_dtype == np.uint16:
        unique_vals = np.unique(mask)
        if len(unique_vals) <= 2:
             mask_cmap = 'gray'

    im_mask = plt.imshow(mask, cmap=mask_cmap, vmin=mask_vmin, vmax=mask_vmax)
    plt.title(f"{title_prefix} - Mask\nShape: {mask_shape}\nDtype: {mask_dtype}")
    plt.colorbar(im_mask, label='Mask Value')
    plt.axis('off')

    plt.suptitle(f"{title_prefix} Data Visualization")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()