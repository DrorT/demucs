"""TensorFlow implementation of the Demucs separator."""

from __future__ import annotations

import os
from typing import List, Sequence, TYPE_CHECKING

import tensorflow as tf

from demucs_tf.blocks import DConv
from demucs_tf.layers import BLSTM, Conv1DWithPadding, ConvTranspose1D, GroupNorm
from demucs_tf.utils import center_trim, resample_frac

if TYPE_CHECKING:  # pragma: no cover
    from demucs_tf.checkpoints.loader import AssignmentReport


class GELU(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:  # type: ignore[override]
        return tf.nn.gelu(inputs)


class ReLU(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:  # type: ignore[override]
        return tf.nn.relu(inputs)


class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit applied along the channel axis."""

    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:  # type: ignore[override]
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return a * tf.nn.sigmoid(b)


def _make_activation(use_gelu: bool) -> tf.keras.layers.Layer:
    return GELU() if use_gelu else ReLU()


def _make_rewrite_activation(use_glu: bool) -> tf.keras.layers.Layer:
    return GLU(axis=1) if use_glu else ReLU()


class DemucsTF(tf.keras.Model):
    """TensorFlow/Keras port of :class:`demucs.demucs.Demucs`."""

    def __init__(
        self,
        sources: Sequence[str],
        audio_channels: int = 2,
        channels: int = 64,
        growth: float = 2.0,
        depth: int = 6,
        rewrite: bool = True,
        lstm_layers: int = 0,
        kernel_size: int = 8,
        stride: int = 4,
        context: int = 1,
        gelu: bool = True,
        glu: bool = True,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_mode: int = 1,
        dconv_depth: int = 2,
        dconv_comp: float = 4.0,
        dconv_attn: int = 4,
        dconv_lstm: int = 4,
        dconv_init: float = 1e-4,
        normalize: bool = True,
        resample: bool = True,
        samplerate: int = 44100,
        segment: float = 4 * 10,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sources = list(sources)
        self.audio_channels = int(audio_channels)
        self.kernel_size = int(kernel_size)
        self.context = int(context)
        self.stride = int(stride)
        self.depth = int(depth)
        self.growth = float(growth)
        self.rewrite = bool(rewrite)
        self.lstm_layers = int(lstm_layers)
        self.resample = bool(resample)
        self.channels = int(channels)
        self.normalize = bool(normalize)
        self.samplerate = int(samplerate)
        self.segment = float(segment)
        self.gelu = bool(gelu)
        self.glu = bool(glu)
        self.norm_starts = int(norm_starts)
        self.norm_groups = int(norm_groups)
        self.dconv_depth = int(dconv_depth)
        self.dconv_comp = float(dconv_comp)
        self.dconv_attn = int(dconv_attn)
        self.dconv_lstm = int(dconv_lstm)
        self.dconv_init = float(dconv_init)
        self.dconv_mode = int(dconv_mode)
        self.num_sources = len(self.sources)

        ch_scale = 2 if glu else 1

        self.encoder_layers: List[tf.keras.layers.Layer] = []
        self.decoder_layers: List[tf.keras.layers.Layer] = []

        in_channels = self.audio_channels
        current_channels = self.channels

        for index in range(self.depth):
            use_group_norm = index >= norm_starts

            encode_layers: List[tf.keras.layers.Layer] = []
            encode_layers.append(
                Conv1DWithPadding(
                    filters=current_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                )
            )
            if use_group_norm:
                encode_layers.append(GroupNorm(norm_groups))
            encode_layers.append(_make_activation(gelu))

            attn = index >= dconv_attn
            lstm = index >= dconv_lstm
            if self.dconv_mode & 1:
                encode_layers.append(
                    DConv(
                        channels=current_channels,
                        depth=dconv_depth,
                        compress=dconv_comp,
                        init=dconv_init,
                        attn=attn,
                        lstm=lstm,
                    )
                )
            if self.rewrite:
                encode_layers.append(
                    Conv1DWithPadding(
                        filters=ch_scale * current_channels,
                        kernel_size=1,
                    )
                )
                if use_group_norm:
                    encode_layers.append(GroupNorm(norm_groups))
                encode_layers.append(_make_rewrite_activation(glu))

            encoder = tf.keras.Sequential(encode_layers, name=f"encoder_{index}")
            self.encoder_layers.append(encoder)
            self._track_trackable(encoder, name=f"encoder_{index}")

            decode_layers: List[tf.keras.layers.Layer] = []
            out_channels = in_channels if index > 0 else self.num_sources * self.audio_channels

            if self.rewrite:
                decode_layers.append(
                    Conv1DWithPadding(
                        filters=ch_scale * current_channels,
                        kernel_size=2 * self.context + 1,
                        padding=(self.context, self.context),
                    )
                )
                if use_group_norm:
                    decode_layers.append(GroupNorm(norm_groups))
                decode_layers.append(_make_rewrite_activation(glu))

            if self.dconv_mode & 2:
                decode_layers.append(
                    DConv(
                        channels=current_channels,
                        depth=dconv_depth,
                        compress=dconv_comp,
                        init=dconv_init,
                        attn=attn,
                        lstm=lstm,
                    )
                )

            decode_layers.append(
                ConvTranspose1D(
                    filters=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                )
            )

            if index > 0:
                if use_group_norm:
                    decode_layers.append(GroupNorm(norm_groups))
                decode_layers.append(_make_activation(gelu))

            decoder = tf.keras.Sequential(decode_layers, name=f"decoder_{index}")
            self.decoder_layers.insert(0, decoder)
            self._track_trackable(decoder, name=f"decoder_{self.depth - 1 - index}")

            in_channels = current_channels
            current_channels = int(growth * current_channels)

        self.hidden_channels = in_channels
        self.lstm = (
            BLSTM(self.hidden_channels, layers=lstm_layers)
            if lstm_layers > 0
            else None
        )

    def load_pytorch_checkpoint(self, checkpoint: str | os.PathLike[str]) -> "AssignmentReport":
        from demucs_tf.checkpoints.loader import load_demucs_tf_weights

        return load_demucs_tf_weights(self, checkpoint)

    def valid_length(self, length: tf.Tensor) -> tf.Tensor:
        length = tf.cast(length, tf.int32)
        if self.resample:
            length = length * 2
        for _ in range(self.depth):
            length = tf.cast(length, tf.float32)
            length = tf.math.ceil((length - self.kernel_size) / self.stride) + 1.0
            length = tf.maximum(length, 1.0)
            length = tf.cast(length, tf.int32)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        if self.resample:
            length = tf.cast(tf.math.ceil(tf.cast(length, tf.float32) / 2.0), tf.int32)
        return length

    def call(self, mix: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = mix
        batch = tf.shape(x)[0]
        length = tf.shape(x)[-1]
        dtype = x.dtype

        if self.normalize:
            mono = tf.reduce_mean(mix, axis=1, keepdims=True)
            mean = tf.reduce_mean(mono, axis=-1, keepdims=True)
            std = tf.math.reduce_std(mono, axis=-1, keepdims=True)
            std = tf.maximum(std, tf.constant(1e-5, dtype=dtype))
            x = (x - mean) / std
        else:
            mean = tf.zeros([batch, 1, 1], dtype=dtype)
            std = tf.ones([batch, 1, 1], dtype=dtype)

        target_length = self.valid_length(length)
        delta = target_length - length
        pad_left = delta // 2
        pad_right = delta - pad_left
        x = tf.pad(x, [[0, 0], [0, 0], [pad_left, pad_right]])

        if self.resample:
            x = resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder_layers:
            x = encode(x, training=training)
            saved.append(x)

        if self.lstm is not None:
            x = self.lstm(x, training=training)

        for decode in self.decoder_layers:
            skip = saved.pop()
            skip = center_trim(skip, x)
            x = decode(x + skip, training=training)

        if self.resample:
            x = resample_frac(x, 2, 1)
        x = x * std + mean
        x = center_trim(x, mix)

        t = tf.shape(x)[-1]
        x = tf.reshape(x, [batch, self.num_sources, self.audio_channels, t])
        return x


__all__ = ["DemucsTF"]
