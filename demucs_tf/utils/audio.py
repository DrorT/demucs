"""Audio-related helpers for the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def center_trim(tensor: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    """Center trim ``tensor`` to match the temporal length of ``reference``.

    Both tensors are expected to follow the ``(B, C, T)`` layout.
    """

    target_length = tf.shape(reference)[-1]
    length = tf.shape(tensor)[-1]
    delta = length - target_length
    def trim():
        crop = delta // 2
        return tensor[..., crop : crop + target_length]

    def identity():
        return tensor

    return tf.cond(delta > 0, trim, identity)


def unfold(x: tf.Tensor, kernel_size: int, stride: int) -> tf.Tensor:
    """Replicates PyTorch ``unfold`` for 1D sequences on channel-first tensors."""

    if kernel_size <= 0 or stride <= 0:
        raise ValueError("`kernel_size` and `stride` must be positive integers.")
    x_t = tf.transpose(x, perm=[0, 2, 1])  # (B, T, C)
    frames = tf.signal.frame(x_t, frame_length=kernel_size, frame_step=stride, axis=1)
    frames = tf.transpose(frames, perm=[0, 3, 1, 2])  # (B, C, frames, kernel)
    return frames


__all__ = ["center_trim", "unfold"]
