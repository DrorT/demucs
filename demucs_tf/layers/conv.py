"""Convolutional layers for the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

Number = Union[int, tf.Tensor]


def _ensure_tuple(value: Union[int, Sequence[int]], length: int = 2) -> Tuple[int, ...]:
    if isinstance(value, Sequence):
        values = tuple(int(v) for v in value)
    else:
        values = (int(value),)
    if len(values) == 1:
        values = values * length
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {values}")
    return values


class Conv1DWithPadding(tf.keras.layers.Layer):
    """1D convolution operating on channel-first tensors (B, C, T).

    This is a thin wrapper around :class:`tf.keras.layers.Conv1D` that mirrors the
    PyTorch ``nn.Conv1d`` behaviour, including explicit control on padding and dilation.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = _ensure_tuple(padding, 2)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.conv: Optional[tf.keras.layers.Conv1D] = None

    def build(self, input_shape):  # type: ignore[override]
        channels = int(input_shape[1])  # channels-first
        if channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({channels}) must be divisible by groups ({self.groups})."
            )
        self.conv = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            dilation_rate=self.dilation,
            groups=self.groups,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.conv is None:
            raise RuntimeError("Layer has not been built yet.")
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (B, T, C)
        if any(self.padding):
            pad_left, pad_right = self.padding
            x = tf.pad(x, [[0, 0], [pad_left, pad_right], [0, 0]])
        x = self.conv(x)
        x = tf.transpose(x, perm=[0, 2, 1])  # back to (B, C, T)
        return x


class ConvTranspose1D(tf.keras.layers.Layer):
    """1D transposed convolution for channel-first inputs.

    TensorFlow provides ``Conv1DTranspose`` starting from TF 2.12, but the behaviour
    differs slightly from PyTorch. This implementation mirrors the PyTorch output size
    computation and accepts ``padding``/``output_padding`` parameters.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: int = 0,
        dilation: int = 1,
        use_bias: bool = True,
        kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = _ensure_tuple(padding, 2)
        self.output_padding = int(output_padding)
        self.dilation = int(dilation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel: Optional[tf.Variable] = None
        self.bias: Optional[tf.Variable] = None

    def build(self, input_shape):  # type: ignore[override]
        in_channels = int(input_shape[1])
        kernel_shape = (self.kernel_size, self.filters, in_channels)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def _full_output_length(self, input_length: Number) -> Number:
        kernel = self.dilation * (self.kernel_size - 1) + 1
        return (
            (input_length - 1) * self.stride
            + kernel
            + self.output_padding
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.kernel is None:
            raise RuntimeError("Layer has not been built yet.")
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (B, T, C_in)
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        full_length = self._full_output_length(length)
        output_shape = tf.stack([batch_size, full_length, self.filters])
        strides = self.stride
        output = tf.nn.conv1d_transpose(
            x,
            filters=self.kernel,
            output_shape=output_shape,
            strides=strides,
            padding="VALID",
            dilations=self.dilation,
        )
        if self.use_bias and self.bias is not None:
            output += self.bias
        pad_left, pad_right = self.padding
        if pad_left or pad_right:
            start = pad_left
            end = tf.shape(output)[1] - pad_right
            output = output[:, start:end, :]
        output = tf.transpose(output, perm=[0, 2, 1])  # (B, C_out, T)
        return output


__all__ = ["Conv1DWithPadding", "ConvTranspose1D"]
