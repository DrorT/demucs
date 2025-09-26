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


def resample_frac(x: tf.Tensor, old: int, new: int) -> tf.Tensor:
    """Resample the waveform by a fractional ratio ``new / old``.

    This mirrors :func:`julius.resample_frac` semantics used in the PyTorch codebase,
    where ``old`` and ``new`` correspond to the original and target sample rates.
    """

    if old <= 0 or new <= 0:
        raise ValueError("`old` and `new` must be positive integers.")
    x_t = tf.transpose(x, perm=[0, 2, 1])  # (B, T, C)
    resample_poly = getattr(tf.signal, "resample_poly", None)
    if resample_poly is not None:
        y = resample_poly(x_t, up=new, down=old, axis=1)
    else:
        length = tf.shape(x_t)[1]
        target = tf.cast(
            tf.math.round(tf.cast(length, tf.float32) * (new / old)),
            tf.int32,
        )
        x_img = tf.expand_dims(x_t, axis=2)  # (B, T, 1, C)
        y_img = tf.image.resize(
            x_img,
            size=[target, 1],
            method="bilinear",
            antialias=True,
        )
        y = tf.squeeze(y_img, axis=2)
    return tf.transpose(y, perm=[0, 2, 1])


__all__ = ["center_trim", "unfold", "resample_frac"]
