import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def BasicDownBlock(filters, kernel_size=3, activation='relu'):
    """A simpler downsampling block for the basic U-Net."""
    def apply(x):
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
        skip = x
        x = layers.MaxPooling2D(pool_size=2)(x)
        return x, skip
    return apply

def BasicUpBlock(filters, kernel_size=3, activation='relu'):
    """A simpler upsampling block for the basic U-Net."""
    def apply(x, skip_conn):
        x = layers.UpSampling2D(size=2, interpolation='bilinear')(x) 
        x = layers.Concatenate()([x, skip_conn])
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
        return x
    return apply

class SinusoidalEmbeddingLayer(layers.Layer):
    """Custom Keras layer for calculating sinusoidal timestep embeddings."""
    def __init__(self, embedding_dim=32, embedding_min_frequency=1.0, embedding_max_frequency=1000.0, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency

    def build(self, input_shape):
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(self.embedding_min_frequency),
                tf.math.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
            )
        )
        self.angular_speeds = tf.cast(2.0 * np.pi * frequencies, dtype=self.compute_dtype)
        super().build(input_shape)
        
    def call(self, x):

        x = tf.cast(x, dtype=self.compute_dtype)
        x_expanded = tf.expand_dims(x, -1)
        sin_embeddings = tf.sin(self.angular_speeds * x_expanded)
        cos_embeddings = tf.cos(self.angular_speeds * x_expanded)
        embeddings = tf.concat([sin_embeddings, cos_embeddings], axis=-1)
        return embeddings 

    def get_config(self):
        # for serialization
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "embedding_min_frequency": self.embedding_min_frequency,
            "embedding_max_frequency": self.embedding_max_frequency,
        })
        return config

def TimeConditioningBlock(target_channels):
    """
    Adds projected time embedding to the feature map.
    Assumes t_proj is the output of an MLP applied to the raw time embedding.
    """
    def apply(features, t_proj):
        time_bias = layers.Dense(target_channels, activation=None)(t_proj)
        time_bias = layers.Reshape((1, 1, target_channels))(time_bias)
        return features + time_bias
    return apply

def build_conditional_unet(input_shape, condition_shape, output_channels, time_embedding_dim, base_channels=32, num_down_blocks=2):
    """
    Builds the full Conditional U-Net accepting noisy mask, timestep,
    and the condition (MRI slice). Uses Input Concatenation for conditioning.
    """
    noisy_input = layers.Input(shape=input_shape, name="noisy_mask_input") # y_t
    time_input = layers.Input(shape=(), dtype=tf.int64, name="timestep_input") # t
    condition_input = layers.Input(shape=condition_shape, name="condition_input") # x (MRI)

    t_emb = SinusoidalEmbeddingLayer(embedding_dim=time_embedding_dim)(time_input)
    time_emb_projected_dim = time_embedding_dim * 4
    t_proj = layers.Dense(time_emb_projected_dim, activation='swish')(t_emb)
    t_proj = layers.Dense(time_emb_projected_dim, activation='swish')(t_proj)

    
    concatenated_input = layers.Concatenate(axis=-1)([noisy_input, condition_input])

    current_channels = base_channels 
    x = layers.Conv2D(current_channels, 3, padding='same', activation='relu')(concatenated_input)

    x = TimeConditioningBlock(current_channels)(x, t_proj) 

    skips = []
    for i in range(num_down_blocks):
        block_input_channels = current_channels
        x, skip = BasicDownBlock(block_input_channels)(x)
        skips.append(skip)
        current_channels *= 2

    bottleneck_input_channels = current_channels // 2
    bottleneck_conv_channels = current_channels

    x = TimeConditioningBlock(bottleneck_input_channels)(x, t_proj)
    x = layers.Conv2D(bottleneck_conv_channels, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(bottleneck_conv_channels, 3, padding='same', activation='relu')(x)
    x = TimeConditioningBlock(bottleneck_conv_channels)(x, t_proj)

    for i in reversed(range(num_down_blocks)):
        current_channels //= 2
        x = BasicUpBlock(current_channels)(x, skips[i])
        x = TimeConditioningBlock(current_channels)(x, t_proj)

    output_tensor = layers.Conv2D(output_channels, 1, padding='same', activation=None, name='unet_output')(x)

    model = Model(inputs=[noisy_input, time_input, condition_input], outputs=output_tensor, name="Conditional_UNet")
    return model