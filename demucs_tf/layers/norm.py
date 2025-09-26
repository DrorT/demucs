"""Normalization layers used by the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


class GroupNorm(tf.keras.layers.Layer):
    """Group Normalization operating on tensors with shape ``(B, C, T)``.

    This mirrors the PyTorch ``nn.GroupNorm`` default behaviour and avoids relying on
    TensorFlow Addons so the port can run in minimal environments.
    """

    def __init__(self, groups: int, epsilon: float = 1e-5, name: Optional[str] = None):
        super().__init__(name=name)
        if groups <= 0:
            raise ValueError("`groups` must be a positive integer.")
        self.groups = int(groups)
        self.epsilon = float(epsilon)
        self.gamma: Optional[tf.Variable] = None
        self.beta: Optional[tf.Variable] = None

    def build(self, input_shape):  # type: ignore[override]
        channels = int(input_shape[1])
        if channels % self.groups != 0:
            raise ValueError(
                f"Number of channels ({channels}) must be divisible by groups ({self.groups})."
            )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.gamma is None or self.beta is None:
            raise RuntimeError("Layer has not been built yet.")
        x = inputs
        batch = tf.shape(x)[0]
        channels = tf.shape(x)[1]
        length = tf.shape(x)[2]
        group_channels = channels // self.groups
        x = tf.reshape(x, [batch, self.groups, group_channels, length])
        mean = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[2, 3], keepdims=True)
        x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
        x = tf.reshape(x, [batch, channels, length])
        x = x * self.gamma[:, None] + self.beta[:, None]
        return x


class LayerScale(tf.keras.layers.Layer):
    """LayerScale as described in https://arxiv.org/abs/2103.17239."""

    def __init__(self, channels: int, init: float = 1e-4, name: Optional[str] = None):
        super().__init__(name=name)
        self.channels = int(channels)
        self.init = float(init)
        self.scale: Optional[tf.Variable] = None

    def build(self, input_shape):  # type: ignore[override]
        self.scale = self.add_weight(
            name="scale",
            shape=(self.channels,),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.scale is None:
            raise RuntimeError("Layer has not been built yet.")
        return inputs * self.scale[:, None]


__all__ = ["GroupNorm", "LayerScale"]
