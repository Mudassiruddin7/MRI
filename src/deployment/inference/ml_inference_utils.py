import logging
import numpy as np
import cv2 
import tensorflow as tf 

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def generate_mask(input_tensor, segmentation_model, threshold=0.5):
    """
    Generates a segmentation mask using the provided loaded model.

    Args:
        input_tensor (np.ndarray or tf.Tensor): Preprocessed input image tensor,
                                                expected shape (1, H, W, 1) float32 [0,1].
                                                H, W should match the model's input layer (e.g., 416x416).
        segmentation_model (tf.keras.Model): Loaded Keras segmentation model.
        threshold (float): Threshold to apply to probability map for binary mask.

    Returns:
        np.ndarray: Binary segmentation mask (H, W) as uint8 {0, 1}, or None if error.
    """
    if segmentation_model is None:
        logger.error("generate_mask: Segmentation model is not loaded.")
        return None
    if input_tensor is None:
        logger.error("generate_mask: Input tensor is None.")
        return None

    logger.info(f"Running segmentation prediction on input shape: {input_tensor.shape}")
    try:
        if input_tensor.ndim == 3: 
            input_tensor_batch = np.expand_dims(input_tensor, axis=0)
        elif input_tensor.ndim == 4 and input_tensor.shape[0] == 1: 
             input_tensor_batch = input_tensor
        else:
             raise ValueError(f"Invalid input dimensions for prediction: {input_tensor.shape}. Expected (1, H, W, C) or (H, W, C).")

        
        model_input_shape = segmentation_model.input_shape[1:] 
        if input_tensor_batch.shape[1:] != model_input_shape:
             logger.warning(f"Input shape {input_tensor_batch.shape[1:]} differs from model expected {model_input_shape}. Prediction might fail.")


        prediction = segmentation_model.predict(input_tensor_batch, verbose=0)
        logger.info(f"Segmentation model raw output shape: {prediction.shape}") 

        
        if prediction.ndim == 4 and prediction.shape[0] == 1 and prediction.shape[-1] == 1:
             mask = (prediction[0, ..., 0] > threshold).astype(np.uint8)
             logger.info(f"Generated binary mask with shape: {mask.shape}, Unique values: {np.unique(mask)}")
             return mask
        else:
             logger.error(f"Unexpected prediction output shape: {prediction.shape}. Cannot extract mask.")
             return None

    except Exception as e:
        logger.error(f"Error during segmentation prediction: {e}", exc_info=True)
        return None

def find_bounding_boxes_contours(mask, min_area_threshold=100, padding=10):
    """
    Finds bounding boxes using contours from a binary mask, filters by area,
    and applies safe padding.

    Args:
        mask (np.ndarray): Input binary mask (H, W), uint8 {0, 1}.
        min_area_threshold (int): Minimum contour area to consider.
        padding (int): Pixels to add around the calculated bounding box.

    Returns:
        list: List of tuples, where each tuple is (x, y, w, h) for a bounding box.
    """
    if mask is None or mask.size == 0:
        logger.warning("find_bounding_boxes_contours: Input mask is empty.")
        return []
    if mask.dtype != np.uint8:
        logger.warning(f"find_bounding_boxes_contours: Mask dtype is {mask.dtype}, converting to uint8.")
        mask = mask.astype(np.uint8)

    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask_2d = mask.squeeze(axis=-1)
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        logger.error(f"find_bounding_boxes_contours: Invalid mask dimensions: {mask.shape}")
        return []

    contours, hierarchy = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_h, img_w = mask_2d.shape[:2]
    logger.info(f"find_bounding_boxes_contours: Found {len(contours)} raw external contours.")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)

            # apply safe padding
            pad_x = padding // 2
            pad_y = padding // 2
            x1p = max(0, x - pad_x)
            y1p = max(0, y - pad_y)
            x2_padded = min(img_w, (x + w) + (padding - pad_x))
            y2_padded = min(img_h, (y + h) + (padding - pad_y))
            final_w = x2_padded - x1p
            final_h = y2_padded - y1p

            if final_w > 0 and final_h > 0:
                logger.debug(f"  -> Box {i}: Area={area:.0f} -> (x={x1p}, y={y1p}, w={final_w}, h={final_h})")
                boxes.append((x1p, y1p, final_w, final_h))

    # sort boxes top-to-bottom left-to-right for consistent order
    boxes.sort(key=lambda b: (b[1], b[0]))
    logger.info(f"Found {len(boxes)} boxes after filtering and padding.")
    return boxes

def classification_preprocessing_5slices(slice_stack_numpy, boxes, target_class_shape=(40, 80)):
    """
    Extracts patches from a 5-slice stack based on bounding boxes,
    resizes, and normalizes them individually for classification.

    Args:
        slice_stack_numpy (np.ndarray): Preprocessed 5-slice stack
                                        (5, H, W, 1) float32 [0,1]. H,W should be cropped size (e.g. 416x416).
        boxes (list): List of tuples [(x, y, w, h)] from find_bounding_boxes.
        target_class_shape (tuple): Target (H, W) for the classification patches.

    Returns:
        list: List of patch sets. Each set corresponds to a box and contains
              5 normalized patches [patch_sl0, patch_sl1, ... patch_sl4],
              where each patch is a (target_H, target_W) float32 [0,1] numpy array.
              Returns empty list if no boxes or error occurs.
    """
    if slice_stack_numpy is None or slice_stack_numpy.ndim != 4 or slice_stack_numpy.shape[0] != 5 or slice_stack_numpy.shape[-1] != 1:
        logger.error(f"classification_preprocessing_5slices: Invalid slice stack shape: {slice_stack_numpy.shape if slice_stack_numpy is not None else 'None'}. Expected (5, H, W, 1).")
        return []
    if not boxes:
        logger.info("classification_preprocessing_5slices: No bounding boxes provided.")
        return []

    logger.info(f"Extracting classification patches for {len(boxes)} boxes. Target patch shape: {target_class_shape}")
    list_of_5_patch_sets = []
    stack_h, stack_w = slice_stack_numpy.shape[1:3] 

    for i, box in enumerate(boxes):
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > stack_w or y + h > stack_h:
             logger.warning(f"Box {i} coordinates {box} are outside stack bounds ({stack_w}x{stack_h}). Skipping.")
             continue

        current_disc_patches = []
        try:
            for slice_idx in range(5):
                
                patch_from_slice = slice_stack_numpy[slice_idx, y:y+h, x:x+w, 0]

                if patch_from_slice.size == 0:
                    logger.warning(f"Box {i}: Empty patch extracted for box {box} from slice {slice_idx}. Skipping this disc.")
                    current_disc_patches = [] 
                    break 

                
                resized_patch = cv2.resize(
                    patch_from_slice, 
                    (target_class_shape[1], target_class_shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )

                min_p, max_p = np.min(resized_patch), np.max(resized_patch)
                if max_p > min_p:
                    normalized_patch = (resized_patch - min_p) / (max_p - min_p)
                else:
                    normalized_patch = np.zeros_like(resized_patch)

                current_disc_patches.append(normalized_patch.astype(np.float32))

            if len(current_disc_patches) == 5:
                list_of_5_patch_sets.append(current_disc_patches)
                logger.debug(f"  -> Box {i}: Successfully extracted and processed 5 patches.")

        except Exception as patch_err:
             logger.error(f"Box {i}: Error processing patches for box {box}: {patch_err}", exc_info=True)

    logger.info(f"Successfully processed {len(list_of_5_patch_sets)} sets of patches.")
    return list_of_5_patch_sets

def predict_label_5slices(list_of_5_patch_sets, classification_model):
    """
    Predicts labels using a list of 5 patches for each item (disc).

    Args:
        list_of_5_patch_sets (list): Output from classification_preprocessing_5slices.
                                     List[List[np.ndarray(H_patch, W_patch)]]
        classification_model (tf.keras.Model): Loaded Keras classification model.
                                                **ASSUMES MODEL EXPECTS A LIST OF 5 INPUTS.**
                                                Each input tensor should be (1, H_patch, W_patch, 1).

    Returns:
        list: List of raw prediction outputs (floats) for each patch set.
              Returns empty list if model not loaded or input is empty.
              Contains NaN for sets where prediction failed.
    """
    if classification_model is None:
        logger.error("predict_label_5slices: Classification model is not loaded.")
        return []
    if not list_of_5_patch_sets:
        logger.info("predict_label_5slices: No patch sets to predict.")
        return []


    try:
        if isinstance(classification_model.input, list) and len(classification_model.input) == 5:
             logger.info("Classification model expects a list of 5 inputs.")
             input_is_list = True
        else:
             logger.warning("Classification model does NOT seem to expect a list of 5 inputs. Assuming single stacked input. Input prep might be incorrect.")
             
             input_is_list = False 
    except Exception as model_check_err:
         logger.warning(f"Could not reliably determine classification model input structure: {model_check_err}. Assuming list input.")
         input_is_list = True 

    logger.info(f"Running classification prediction for {len(list_of_5_patch_sets)} patch sets.")
    predictions_list = []

    for i, five_patches in enumerate(list_of_5_patch_sets):
        if len(five_patches) != 5:
             logger.warning(f"Skipping prediction for set {i}, expected 5 patches, got {len(five_patches)}.")
             predictions_list.append(float('nan'))
             continue

        input_data_for_model = []
        try:
            if input_is_list:
                for patch in five_patches:
                    patch_tensor = tf.cast(patch, dtype=tf.float32)
                    patch_tensor = tf.expand_dims(patch_tensor, axis=0) # add batch dim -> (1, h, w)
                    patch_tensor = tf.expand_dims(patch_tensor, axis=-1) # add channel dim -> (1, h, w, 1)
                    input_data_for_model.append(patch_tensor)
            else:
                 logger.error("Model expects single input, stacking logic not implemented yet.")
                 raise NotImplementedError("Classification model expecting single stacked input is not yet handled.")


            predictions = classification_model.predict(input_data_for_model, verbose=0)

           
            if isinstance(predictions, list): 
                 raw_prediction = predictions[0][0][0] 
            else: 
                 raw_prediction = predictions[0][0]

            predictions_list.append(float(raw_prediction)) 
            logger.debug(f"  -> Set {i}: Prediction = {raw_prediction:.4f}")

        except Exception as e:
            logger.error(f"Error predicting patch set {i}: {e}", exc_info=True)
            predictions_list.append(float('nan')) 

    logger.info(f"Finished classification prediction. Results count: {len(predictions_list)}")
    return predictions_list

def create_overlay_image(base_image_slice, boxes, color=(255, 0, 0), thickness=1, font_scale=0.5, text_color=(255, 255, 255)):
    """
    Draws bounding boxes with labels ('Disc 1', 'Disc 2', ...) on a grayscale image slice.

    Args:
        base_image_slice (np.ndarray): The original (or preprocessed) grayscale
                                       image slice (H, W), expected float32 [0,1].
        boxes (list): List of tuples [(x, y, w, h)] from find_bounding_boxes.
                      Boxes are assumed to be sorted top-to-bottom, left-to-right.
        color (tuple): BGR color for the bounding boxes (default green).
        thickness (int): Thickness of the bounding box lines.
        font_scale (float): Font scale for the labels.
        text_color (tuple): BGR color for the labels (default white).

    Returns:
        np.ndarray: Image slice (H, W, 3) as uint8 [0,255] with boxes and labels drawn.
                    Returns None if error occurs.
    """
    if base_image_slice is None: logger.error("create_overlay_image: base_image_slice is None."); return None
    if base_image_slice.ndim != 2: logger.error(f"create_overlay_image: Expected 2D base image, got shape {base_image_slice.shape}"); return None

    try:
        if np.max(base_image_slice) <= 1.0 and np.min(base_image_slice) >= 0.0:
             img_uint8 = (base_image_slice * 255).astype(np.uint8)
        else:
             logger.warning("create_overlay_image: Input slice not in expected [0,1] float range. Attempting direct conversion.")
             img_uint8 = base_image_slice.astype(np.uint8)

        overlay_img = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if boxes:
            logger.info(f"Drawing {len(boxes)} boxes and labels on overlay image.")
            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color, thickness)

                label = f"Disc {i+1}"
                text_x = x + 5
                text_y = y - 5
                if text_y < 10: text_y = y + 15 # move below if too close to top
                if text_x + (len(label) * int(10 * font_scale)) > overlay_img.shape[1]: # estimated width check
                    text_x = x - (len(label) * int(10 * font_scale)) 

                cv2.putText(overlay_img, label, (text_x, text_y), font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)
        else:
            logger.info("No boxes provided to draw on overlay.")

        return overlay_img 

    except Exception as e:
        logger.error(f"Error creating overlay image: {e}", exc_info=True)
        return None