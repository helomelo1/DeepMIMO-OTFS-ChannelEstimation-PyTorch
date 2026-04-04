"""
PositionNet — TensorFlow / Keras implementation.

Paper: "Deep Learning-based Channel Estimation for
        Massive MIMO-OTFS Communication Systems"
        (Payami & Blostein, WTS 2024)

Architecture (Fig. 2):
    Conv3D -> Conv3D -> LayerNorm -> BatchNorm -> Rescaling
    -> Softmax (spatial) -> Dense (relu) -> Dense (sigmoid)

Total params: ~3800
Loss: Cosine Similarity
Optimizer: AdamW (lr=1e-3)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Tuned to give ~3800 total params (paper states "only 3800 parameters").
# 64 feature maps as stated in the paper text would yield >4500 params,
# so we use 51 to match the reported parameter count.
NUM_FILTERS = 51
DENSE_UNITS = 14

# Spatial dimensions (from project's data generation config)
M_TAU = 150
N_NU = 10
NT = 16


class SpatialSoftmax(layers.Layer):
    """Softmax over all spatial dimensions, applied per-channel.

    Analogous to argmax in OMP but differentiable — highlights bins
    most likely to contain non-zero channel elements.
    """

    def call(self, inputs):
        shape = tf.shape(inputs)
        channels = inputs.shape[-1]
        flat = tf.reshape(inputs, [shape[0], -1, channels])
        flat = tf.nn.softmax(flat, axis=1)
        return tf.reshape(flat, shape)


def cosine_similarity_loss(y_true, y_pred):
    """1 - cosine_similarity, averaged over the batch."""
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    cos = tf.reduce_sum(y_true * y_pred, axis=1) / (
        tf.norm(y_true, axis=1) * tf.norm(y_pred, axis=1) + 1e-8
    )
    return tf.reduce_mean(1.0 - cos)


def build_position_net(
    Mt=M_TAU, Nv=N_NU, Nt=NT,
    num_filters=NUM_FILTERS, dense_units=DENSE_UNITS,
):
    """Build PositionNet.

    Input shape:  (batch, Mt, Nv, Nt, 1)   — |Phi^H @ y_DD| reshaped
    Output shape: (batch, Mt, Nv, Nt, 1)   — per-bin probability (sigmoid)
    """
    inputs = keras.Input(shape=(Mt, Nv, Nt, 1), name="input")

    # Two Conv3D layers (paper: "two concatenated Convolutional layers")
    x = layers.Conv3D(num_filters, kernel_size=1, activation="relu", name="conv1")(inputs)
    x = layers.Conv3D(num_filters, kernel_size=1, activation="relu", name="conv2")(x)

    # Normalization
    x = layers.LayerNormalization(name="layer_norm")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)

    # Fixed rescaling before softmax (Keras Rescaling has 0 trainable params)
    x = layers.Rescaling(scale=1.0, name="rescaling")(x)

    # Spatial softmax — the key layer (replaces argmax from OMP)
    x = SpatialSoftmax(name="spatial_softmax")(x)

    # Two Dense layers for nonlinear mapping
    x = layers.Dense(dense_units, activation="relu", name="fc1")(x)
    x = layers.Dense(1, activation="sigmoid", name="fc2")(x)

    return keras.Model(inputs, x, name="PositionNet")


if __name__ == "__main__":
    model = build_position_net()
    model.summary()

    print(f"\nTotal params:     {model.count_params()}")
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"Trainable:        {trainable}")
    print(f"Non-trainable:    {non_trainable}")
