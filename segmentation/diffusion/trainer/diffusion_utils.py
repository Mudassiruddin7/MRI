import tensorflow as tf
import numpy as np
import time
from . import config 

print("Calculating diffusion schedule constants...")

betas = tf.linspace(config.BETA_START, config.BETA_END, config.TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = tf.math.cumprod(alphas, axis=0)
alphas_cumprod_prev = tf.pad(alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)

sqrt_alphas_cumprod = tf.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

posterior_log_variance_clipped = tf.math.log(tf.maximum(posterior_variance, 1e-20))

sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / alphas_cumprod - 1.0) 
posterior_mean_coef1 = tf.sqrt(alphas_cumprod_prev) * betas / (1. - alphas_cumprod)
posterior_mean_coef2 = tf.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

print("Diffusion constants calculation finished.")
print(f"  betas shape: {betas.shape}")
print(f"  alphas_cumprod shape: {alphas_cumprod.shape}")
print(f"  posterior_variance shape: {posterior_variance.shape}")

def extract(a, t, x_shape):
    """Extracts values from 'a' at indices 't' and reshapes."""
    batch_size = tf.shape(t)[0]
    out = tf.gather(a, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])

@tf.function
def q_sample(x_start, t, noise=None):
    """Adds noise to data x_start at timestep t."""
    if noise is None:
        noise = tf.random.normal(shape=tf.shape(x_start), dtype=tf.float32)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

@tf.function
def p_sample(model, y_t, t_tensor, i, condition_mri):
    """Samples y_{t-1} from y_t using the DDPM sampling formula."""
    pred_noise = model([y_t, t_tensor, condition_mri], training=False)

    pred_x0 = extract(sqrt_recip_alphas_cumprod, t_tensor, y_t.shape) * y_t - \
              extract(sqrt_recipm1_alphas_cumprod, t_tensor, y_t.shape) * pred_noise
    pred_x0 = tf.clip_by_value(pred_x0, -1., 1.)

    model_mean = extract(posterior_mean_coef1, t_tensor, y_t.shape) * pred_x0 + \
                 extract(posterior_mean_coef2, t_tensor, y_t.shape) * y_t

    model_log_variance = extract(posterior_log_variance_clipped, t_tensor, y_t.shape)
    model_stddev = tf.exp(0.5 * model_log_variance)

    z = tf.random.normal(shape=tf.shape(y_t), dtype=tf.float32)
    z = tf.where(i == 0, tf.zeros_like(z), z) 

    pred_img_prev = model_mean + model_stddev * z
    return pred_img_prev


def generate_segmentation(model, input_mri, num_images=1, verbose=True):
    """Generates segmentation masks using the reverse diffusion process."""
    if verbose: print(f"Starting segmentation generation for {num_images} image(s)...")
    start_gen_time = time.time()

    if len(tf.shape(input_mri)) == 3:
        input_mri = tf.expand_dims(input_mri, 0)
    if num_images > 1 and tf.shape(input_mri)[0] == 1:
       input_mri = tf.repeat(input_mri, repeats=num_images, axis=0)
    elif num_images != tf.shape(input_mri)[0] and tf.shape(input_mri)[0] != 1 :
       raise ValueError("num_images must match the batch size of input_mri if batch size > 1")
    batch_size = tf.shape(input_mri)[0]

    target_shape = (batch_size, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS_MASK)

    current_mask = tf.random.normal(shape=target_shape, dtype=tf.float32)

    for i in tf.range(config.TIMESTEPS - 1, -1, -1):
        if verbose and (i.numpy() % 100 == 0 or i.numpy() == config.TIMESTEPS - 1):
            print(f"  Sampling timestep {i.numpy()}...")
        t_tensor = tf.fill((batch_size,), i)
        current_mask = p_sample(model, current_mask, t_tensor, i, input_mri)

    generated_mask = current_mask
    end_gen_time = time.time()
    if verbose: print(f"Generation finished in {end_gen_time - start_gen_time:.2f} seconds.")

    generated_mask = (generated_mask + 1.0) / 2.0
    generated_mask = tf.clip_by_value(generated_mask, 0.0, 1.0)

    return generated_mask