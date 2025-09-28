"""Audio-related helpers for the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Sequence, Tuple

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


def pad1d(
    x: tf.Tensor,
    paddings: Tuple[int, int],
    mode: str = "constant",
    constant_values: float = 0.0,
) -> tf.Tensor:
    """1D padding helper mirroring :func:`torch.nn.functional.pad` semantics.

    Args:
        x: Tensor with shape ``(..., T)``.
        paddings: Tuple of ``(left, right)`` padding sizes.
        mode: Either ``"constant"`` or ``"reflect"``.
        constant_values: Fill value used when ``mode == "constant"``.
    """

    left, right = paddings
    if left < 0 or right < 0:
        raise ValueError("padding values must be non-negative")
    if mode not in {"constant", "reflect"}:
        raise ValueError("pad1d currently supports 'constant' and 'reflect' modes only")

    if mode == "reflect":
        length = tf.shape(x)[-1]
        max_pad = tf.maximum(left, right)
        def reflect_padding() -> tf.Tensor:
            extra = max_pad - length + 1
            extra_right = tf.minimum(right, extra)
            extra_left = extra - extra_right
            new_left = left - extra_left
            new_right = right - extra_right
            padded = tf.pad(
                x,
                [[0, 0]] * (len(x.shape) - 1) + [[extra_left, extra_right]],
                mode="CONSTANT",
                constant_values=constant_values,
            )
            return tf.pad(
                padded,
                [[0, 0]] * (len(x.shape) - 1) + [[new_left, new_right]],
                mode="REFLECT",
            )

        return tf.cond(length <= max_pad, reflect_padding, lambda: tf.pad(
            x,
            [[0, 0]] * (len(x.shape) - 1) + [[left, right]],
            mode="REFLECT",
        ))

    return tf.pad(
        x,
        [[0, 0]] * (len(x.shape) - 1) + [[left, right]],
        mode="CONSTANT",
        constant_values=constant_values,
    )


def stft(
    x: tf.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    pad: int = 0,
    window: tf.Tensor | None = None,
    center: bool = True,
) -> tf.Tensor:
    """Compute a complex STFT matching Demucs settings.

    Args:
        x: Tensor shaped ``(B, C, T)``.
        n_fft: FFT size.
        hop_length: Hop size (defaults to ``n_fft // 4``).
        pad: Additional zero-padding factor (currently only ``0`` supported).
        window: Optional analysis window; defaults to Hann.
    """

    if pad != 0:
        raise NotImplementedError("pad > 0 is not supported in TF STFT helper yet")
    if hop_length is None:
        hop_length = n_fft // 4
    batch = tf.shape(x)[0]
    channels = tf.shape(x)[1]
    if center:
        pad_amount = n_fft // 2
        x = tf.pad(
            x,
            [[0, 0], [0, 0], [pad_amount, pad_amount]],
            mode="REFLECT",
        )
    length = tf.shape(x)[2]
    x_flat = tf.reshape(x, [-1, length])
    window = window or tf.signal.hann_window(n_fft, periodic=True)
    z = tf.signal.stft(
        x_flat,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=lambda frame_length, dtype: tf.cast(window, dtype),
        pad_end=False,
    )
    z = z / tf.sqrt(tf.cast(n_fft, z.dtype))
    frames = tf.shape(z)[-2]
    freqs = tf.shape(z)[-1]
    z = tf.reshape(z, [batch, channels, frames, freqs])
    return tf.transpose(z, perm=[0, 1, 3, 2])


def istft(
    z: tf.Tensor,
    hop_length: int | None = None,
    length: int | None = None,
    window: tf.Tensor | None = None,
) -> tf.Tensor:
    """Inverse-STFT counterpart of :func:`stft`."""

    batch = tf.shape(z)[0]
    channels = tf.shape(z)[1]
    freqs = tf.shape(z)[2]
    frames = tf.shape(z)[3]
    n_fft = 2 * (freqs - 1)
    if hop_length is None:
        hop_length = n_fft // 4
    window = window or tf.signal.hann_window(n_fft, periodic=True)
    z_flat = tf.reshape(z, [-1, freqs, frames])
    x = tf.signal.inverse_stft(
        z_flat,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=lambda frame_length, dtype: tf.cast(window, dtype),
    )
    if length is not None:
        if isinstance(length, tf.Tensor):
            length = tf.cast(length, tf.int32)
            indices = tf.range(length)
            x = tf.gather(x, indices, axis=-1)
        else:
            x = x[..., :length]
    x = x * tf.sqrt(tf.cast(n_fft, x.dtype))
    return tf.reshape(x, [batch, channels, -1])


__all__ = ["center_trim", "unfold", "resample_frac", "pad1d", "stft", "istft"]
