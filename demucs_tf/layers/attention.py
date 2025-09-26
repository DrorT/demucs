"""Attention layers used in the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Optional

import math

import tensorflow as tf

from .conv import Conv1DWithPadding


class LocalState(tf.keras.layers.Layer):
    """Local state attention block replicating the PyTorch implementation."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        nfreqs: int = 0,
        ndecay: int = 4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if channels % heads != 0:
            raise ValueError("`channels` must be divisible by `heads`.")
        self.channels = int(channels)
        self.heads = int(heads)
        self.nfreqs = int(nfreqs)
        self.ndecay = int(ndecay)
        self.content = Conv1DWithPadding(channels, kernel_size=1)
        self.query = Conv1DWithPadding(channels, kernel_size=1)
        self.key = Conv1DWithPadding(channels, kernel_size=1)
        self.query_freqs = None
        if self.nfreqs:
            self.query_freqs = Conv1DWithPadding(self.heads * self.nfreqs, kernel_size=1)
        self.query_decay = None
        if self.ndecay:
            self.query_decay = Conv1DWithPadding(self.heads * self.ndecay, kernel_size=1)
        self.proj = Conv1DWithPadding(self.channels, kernel_size=1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        x = inputs
        batch = tf.shape(x)[0]
        channels = tf.shape(x)[1]
        length = tf.shape(x)[2]
        head_dim = self.channels // self.heads
        time_indices = tf.cast(tf.range(length), x.dtype)
        delta = time_indices[None, :] - time_indices[:, None]

        queries = self.query(x)
        queries = tf.reshape(queries, [batch, self.heads, head_dim, length])
        keys = self.key(x)
        keys = tf.reshape(keys, [batch, self.heads, head_dim, length])
        dots = tf.einsum("bhct,bhcs->bhts", keys, queries)
        dots = dots / tf.sqrt(tf.cast(head_dim, x.dtype))

        freq_kernel = None
        if self.nfreqs and self.query_freqs is not None:
            periods = tf.cast(tf.range(1, self.nfreqs + 1), x.dtype)
            freq_kernel = tf.math.cos(
                tf.constant(2.0 * math.pi, dtype=x.dtype) * delta[None, :, :] / periods[:, None, None]
            )
            freq_q = self.query_freqs(x)  # (B, heads * nfreqs, T)
            freq_q = tf.reshape(freq_q, [batch, self.heads, self.nfreqs, length])
            freq_q = freq_q / tf.sqrt(tf.cast(self.nfreqs, x.dtype))
            dots += tf.einsum("kts,bhks->bhts", freq_kernel, freq_q)

        if self.ndecay and self.query_decay is not None:
            decays = tf.cast(tf.range(1, self.ndecay + 1), x.dtype)
            decay_kernel = -decays[:, None, None] * tf.abs(delta)[None, :, :] / tf.sqrt(
                tf.cast(self.ndecay, x.dtype)
            )
            decay_q = self.query_decay(x)
            decay_q = tf.reshape(decay_q, [batch, self.heads, self.ndecay, length])
            decay_q = tf.sigmoid(decay_q) / 2.0
            dots += tf.einsum("kts,bhks->bhts", decay_kernel, decay_q)

        mask = tf.eye(length, dtype=tf.bool)
        dots = tf.where(mask[None, None, :, :], tf.fill(tf.shape(dots), tf.cast(-100.0, x.dtype)), dots)
        weights = tf.nn.softmax(dots, axis=2)

        content = self.content(x)
        content = tf.reshape(content, [batch, self.heads, head_dim, length])
        result = tf.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs and freq_kernel is not None:
            time_sig = tf.einsum("bhts,kts->bhks", weights, freq_kernel)
            result = tf.concat([result, tf.reshape(time_sig, [batch, self.heads, self.nfreqs, length])], axis=2)
        result = tf.reshape(result, [batch, -1, length])
        result = self.proj(result)
        return inputs + result


__all__ = ["LocalState"]
