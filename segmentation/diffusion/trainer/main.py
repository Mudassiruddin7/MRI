# main.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import logging 
import time
import traceback

from . import config
from . import data_utils
from . import model_utils
from . import diffusion_utils
from . import visualization_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Conditional Diffusion Model for MRI Segmentation")
    parser.add_argument(
        "--mode",
        choices=["train", "sample", "inspect_data"],
        help="Operation mode: train, sample, or inspect_data"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to pre-trained weights to load for training or sampling")

    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")

    parser.add_argument("--base_channels", type=int, default=config.MODEL_BASE_CHANNELS, help="Base channels for U-Net")
    parser.add_argument("--num_down_blocks", type=int, default=config.MODEL_NUM_DOWN_BLOCKS, help="Number of U-Net down/up blocks")
    parser.add_argument("--time_emb_dim", type=int, default=config.MODEL_TIME_EMB_DIM, help="Dimension for time embedding")
    
    parser.add_argument("--tfrecord_dir", type=str, required=True, help="GCS path to TFRecord directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="GCS path for saving checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="GCS path to save generated samples")

    args = parser.parse_args()
    return args

def run_training(args):
    logging.info("--- Starting Training ---")
    logging.info(f"Epochs: {args.epochs}, Learning Rate: {args.lr}, Batch Size: {args.batch_size}")
    logging.info(f"Model Params: base_channels={args.base_channels}, num_down_blocks={args.num_down_blocks}, time_emb_dim={args.time_emb_dim}")

    logging.info("Loading training dataset...")
    train_dataset = data_utils.load_tfrecord_dataset(
        tfrecord_dir=args.tfrecord_dir,
        batch_size=args.batch_size,
        buffer_size=config.SHUFFLE_BUFFER_SIZE,
        is_train=True,
        use_preprocessing=True
    )
    logging.info("Dataset loaded.")

    logging.info("Building model...")
    model = model_utils.build_conditional_unet(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MASK),
        condition_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MRI),
        output_channels=config.IMG_CHANNELS_MASK,
        time_embedding_dim=args.time_emb_dim,
        base_channels=args.base_channels,
        num_down_blocks=args.num_down_blocks
    )
    model.summary(print_fn=logging.info) 

    if args.load_weights:
        logging.info(f"Loading weights from: {args.load_weights}")
        try:
            model.load_weights(args.load_weights)
        except Exception as e:
            logging.error(f"Could not load weights: {e}")
            return 

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    mse_loss = tf.keras.losses.MeanSquaredError()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    @tf.function
    def train_step(batch_mri, batch_clean_mask, model):
        batch_size = tf.shape(batch_clean_mask)[0]
        t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=config.TIMESTEPS, dtype=tf.int64)
        noise = tf.random.normal(shape=tf.shape(batch_clean_mask), dtype=tf.float32)
        batch_noisy_mask = diffusion_utils.q_sample(batch_clean_mask, t, noise) 

        with tf.GradientTape() as tape:
            predicted_noise = model([batch_noisy_mask, t, batch_mri], training=True)
            loss = mse_loss(noise, predicted_noise)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    epoch_losses = []
    logging.info("Starting training loop...")
    for epoch in range(args.epochs):
        start_time = time.time()
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        total_epoch_loss = 0.0
        num_batches = 0
        for step, (batch_mri, batch_clean_mask) in enumerate(train_dataset):
            batch_loss = train_step(batch_mri, batch_clean_mask, model)
            total_epoch_loss += batch_loss
            num_batches += 1
            if step % 50 == 0:
                logging.info(f"  Step {step}: Batch Loss = {batch_loss.numpy():.4f}")

        avg_epoch_loss = total_epoch_loss / max(1, num_batches) 
        epoch_losses.append(avg_epoch_loss.numpy())
        epoch_duration = time.time() - start_time
        logging.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")
        logging.info(f"  Average Epoch Loss: {avg_epoch_loss.numpy():.4f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.checkpoint_dir, f"unet_epoch_{epoch+1}.weights.h5")
            try:
                model.save_weights(save_path)
                logging.info(f"  Checkpoint saved to {save_path}")
            except Exception as e:
                logging.error(f"  Error saving checkpoint: {e}")

    logging.info("--- Training Finished ---")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.grid(True)
    loss_plot_path = os.path.join(args.checkpoint_dir, "training_loss.png")
    plt.savefig(loss_plot_path)
    logging.info(f"Training loss plot saved to {loss_plot_path}")

def run_sampling(args):
    logging.info("--- Running Sampling/Inference ---")
    logging.info(f"Loading weights from: {args.load_weights}")
    logging.info(f"Saving results to: {args.output_dir}")
    logging.info(f"Model Params: base_channels={args.base_channels}, num_down_blocks={args.num_down_blocks}, time_emb_dim={args.time_emb_dim}")


    if not args.load_weights or not os.path.exists(args.load_weights):
        logging.error("Model weights file not specified or not found. Use --load_weights.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Building model structure...")
    model = model_utils.build_conditional_unet(
         input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MASK),
        condition_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MRI),
        output_channels=config.IMG_CHANNELS_MASK,
        time_embedding_dim=args.time_emb_dim,
        base_channels=args.base_channels,
        num_down_blocks=args.num_down_blocks
    )

    try:
        model.load_weights(args.load_weights)
        logging.info("Model weights loaded successfully.")
    except Exception as e:
        logging.error(f"Could not load weights: {e}")
        return

    logging.info("Loading test data sample(s)...")
    try:
        test_dataset = data_utils.load_tfrecord_dataset(
            args.tfrecord_dir, batch_size=args.num_samples, buffer_size=1, is_train=False, use_preprocessing=True
        )
        test_batch = next(iter(test_dataset))
        test_mri = test_batch[0] 
        true_mask = test_batch[1] 
        logging.info(f"Loaded {args.num_samples} samples for inference.")
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        return

    generated_mask = diffusion_utils.generate_segmentation(model, test_mri, num_images=args.num_samples)

    logging.info("Processing and saving results...")
    true_mask_vis = (true_mask + 1.0) / 2.0 # normalize true mask [-1,1] -> [0,1]
    mri_display = (test_mri + 1.0) / 2.0 # normalize mri [-1,1] -> [0,1]

    for i in range(args.num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.squeeze(mri_display[i].numpy()), cmap='gray', vmin=0.0, vmax=1.0)
        axes[0].set_title("Input MRI")
        axes[0].axis('off')

        axes[1].imshow(np.squeeze(generated_mask[i].numpy()), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Generated Segmentation")
        axes[1].axis('off')

        axes[2].imshow(np.squeeze(true_mask_vis[i].numpy()), cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

        fig.suptitle(f"Inference Result - Sample {i+1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(args.output_dir, f"result_sample_{i+1}.png")
        plt.savefig(save_path)
        logging.info(f"Saved result image to {save_path}")
        plt.close(fig) 

def run_inspect_data(args):
    logging.info("--- Inspecting Data ---")
    try:
        logging.info("Loading RAW data sample...")
        raw_dataset = data_utils.load_tfrecord_dataset(
            args.tfrecord_dir, batch_size=1, buffer_size=1, is_train=False, use_preprocessing=False
        )
        raw_example = next(iter(raw_dataset))
        raw_image, raw_mask = raw_example[0][0], raw_example[1][0]
        logging.info(f"Raw Image Shape: {raw_image.shape}, dtype: {raw_image.dtype}, Range: [{tf.reduce_min(raw_image)}, {tf.reduce_max(raw_image)}]")
        logging.info(f"Raw Mask Shape: {raw_mask.shape}, dtype: {raw_mask.dtype}, Range: [{tf.reduce_min(raw_mask)}, {tf.reduce_max(raw_mask)}]")
        visualization_utils.visualize_sample(raw_image, raw_mask, title_prefix="Raw Inspect")

        logging.info("Loading PROCESSED data sample...")
        proc_dataset = data_utils.load_tfrecord_dataset(
            args.tfrecord_dir, batch_size=1, buffer_size=1, is_train=False, use_preprocessing=True
        )
        proc_example = next(iter(proc_dataset))
        proc_image, proc_mask = proc_example[0][0], proc_example[1][0]
        logging.info(f"Processed Image Shape: {proc_image.shape}, dtype: {proc_image.dtype}, Range: [{tf.reduce_min(proc_image):.2f}, {tf.reduce_max(proc_image):.2f}]")
        logging.info(f"Processed Mask Shape: {proc_mask.shape}, dtype: {proc_mask.dtype}, Range: [{tf.reduce_min(proc_mask):.2f}, {tf.reduce_max(proc_mask):.2f}]")
        visualization_utils.visualize_sample(proc_image, proc_mask, title_prefix="Processed Inspect")

        
    except Exception as e:
        logging.error(f"Error during data inspection: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Running in mode: {args.mode}")

    if args.mode == "train":
        logging.info("--- Starting Training Stage ---")
        run_training(args) 
        logging.info("--- Training Stage Finished ---")

        logging.info("--- Starting Post-Training Sampling Stage ---")

        final_weights_filename = f"unet_epoch_{args.epochs}.weights.h5"
        final_weights_path = os.path.join(args.checkpoint_dir, final_weights_filename)
        logging.info(f"Looking for final weights at: {final_weights_path}")

        if os.path.exists(final_weights_path):
            logging.info("Final weights found. Preparing arguments for sampling...")

            args.load_weights = final_weights_path
            logging.info(f"Sampling Args: num_samples={args.num_samples}, output_dir={args.output_dir}")
            logging.info(f"Other args (tfrecord_dir, model params, etc.) reused from training args.")

            try:
                run_sampling(args)
                logging.info("--- Post-Training Sampling Stage Finished ---")
            except Exception as e:
                logging.error(f"Error during post-training sampling: {e}", exc_info=True)
        else:
            logging.error(f"Final weights file not found at {final_weights_path}. Cannot run post-training sampling.")

    elif args.mode == "sample":

        if not args.load_weights or not args.output_dir or not args.tfrecord_dir:
             logging.error("Missing required arguments (--load_weights, --output_dir, --tfrecord_dir) for sample mode.")
        else:
            run_sampling(args)

    elif args.mode == "inspect_data":
        run_inspect_data(args) 

    else:
        logging.error(f"Unknown mode: {args.mode}")

    logging.info("Script finished.")