"""Helpers to transfer Demucs weights from PyTorch to TensorFlow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import torch

from demucs_tf.layers.conv import Conv1DWithPadding, ConvTranspose1D
from demucs_tf.layers.norm import GroupNorm, LayerScale
from demucs_tf.layers.recurrent import BLSTM

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Loading & bookkeeping
# -----------------------------------------------------------------------------


def load_pytorch_state_dict(checkpoint: Path | str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and return its ``state_dict`` on CPU.

    The helper is resilient to checkpoints that wrap the weights in a top-level
    dictionary (e.g. ``{"state_dict": ...}``). All tensors are detached, moved to
    CPU, and left in ``torch.Tensor`` form so the caller can choose when to
    convert to NumPy.
    """

    checkpoint = Path(checkpoint)
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, Mapping) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    result: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        else:
            LOGGER.debug("Skipping non-tensor entry %s of type %s", key, type(value))
    return result


@dataclass
class AssignmentReport:
    """Tracks which PyTorch weights were consumed when porting to TF."""

    assigned: Dict[str, str]
    missing: Sequence[str]
    unused: Sequence[str]


# -----------------------------------------------------------------------------
# Tensor conversions
# -----------------------------------------------------------------------------


def pytorch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a PyTorch tensor and return a NumPy array."""

    return tensor.detach().cpu().numpy()


def pytorch_to_tf_conv1d(weight: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch ``nn.Conv1d`` kernel to TensorFlow ``Conv1D`` format."""

    # PyTorch: (out_channels, in_channels, kernel)
    # TensorFlow (channels-last): (kernel, in_channels, out_channels)
    return pytorch_to_numpy(weight).transpose(2, 1, 0)


def pytorch_to_tf_conv_transpose1d(weight: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch ``nn.ConvTranspose1d`` kernel to TF format."""

    # PyTorch: (in_channels, out_channels, kernel)
    # TensorFlow: (kernel, out_channels, in_channels)
    return pytorch_to_numpy(weight).transpose(2, 1, 0)


def pytorch_to_tf_dense(weight: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch ``nn.Linear`` kernel to Keras dense format."""

    # PyTorch: (out_features, in_features)
    # Keras: (in_features, out_features)
    return pytorch_to_numpy(weight).transpose(1, 0)


def combine_pytorch_biases(bias_ih: torch.Tensor, bias_hh: torch.Tensor) -> np.ndarray:
    """Sum the input/hidden LSTM biases and return a NumPy array."""

    return pytorch_to_numpy(bias_ih + bias_hh)


# -----------------------------------------------------------------------------
# Assignment helpers for concrete layer types
# -----------------------------------------------------------------------------


def ensure_layer_built(layer: tf.keras.layers.Layer, example_shape: Sequence[int]) -> None:
    """Ensure a layer is built, calling it with zeros if necessary."""

    if not layer.built:
        dummy = tf.zeros(example_shape, dtype=tf.float32)
        layer(dummy)


def assign_conv1d_weights(
    layer: Conv1DWithPadding,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch Conv1d weights to a :class:`Conv1DWithPadding` layer."""

    if example_shape is None:
        example_shape = [1, weight.shape[1], 4]
    ensure_layer_built(layer, example_shape)
    kernel = pytorch_to_tf_conv1d(weight)
    layer.conv.kernel.assign(kernel)
    if layer.use_bias and bias is not None and layer.conv.bias is not None:
        layer.conv.bias.assign(pytorch_to_numpy(bias))


def assign_conv_transpose1d_weights(
    layer: ConvTranspose1D,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch ConvTranspose1d weights to a :class:`ConvTranspose1D`."""

    if example_shape is None:
        example_shape = [1, weight.shape[0], 4]
    ensure_layer_built(layer, example_shape)
    kernel = pytorch_to_tf_conv_transpose1d(weight)
    layer.kernel.assign(kernel)
    if layer.use_bias and bias is not None and layer.bias is not None:
        layer.bias.assign(pytorch_to_numpy(bias))


def assign_dense_weights(
    layer: tf.keras.layers.Dense,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch linear weights to a :class:`tf.keras.layers.Dense`."""

    if example_shape is None:
        example_shape = [1, weight.shape[1]]
    ensure_layer_built(layer, example_shape)
    kernel = pytorch_to_tf_dense(weight)
    layer.kernel.assign(kernel)
    if layer.use_bias and bias is not None and layer.bias is not None:
        layer.bias.assign(pytorch_to_numpy(bias))


def assign_group_norm(
    layer: GroupNorm,
    weight: torch.Tensor,
    bias: torch.Tensor,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch GroupNorm parameters."""

    if example_shape is None:
        example_shape = [1, weight.shape[0], 4]
    ensure_layer_built(layer, example_shape)
    layer.gamma.assign(pytorch_to_numpy(weight))
    layer.beta.assign(pytorch_to_numpy(bias))


def assign_layer_scale(layer: LayerScale, scale: torch.Tensor) -> None:
    """Assign LayerScale parameters."""

    if not layer.built:
        # Build with dummy input of shape (1, channels, 1)
        ensure_layer_built(layer, [1, scale.shape[0], 1])
    layer.scale.assign(pytorch_to_numpy(scale))


# -----------------------------------------------------------------------------
# LSTM assignment (BLSTM)
# -----------------------------------------------------------------------------


def split_lstm_gates(kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the concatenated gates of an LSTM kernel."""

    unit_count = kernel.shape[-1] // 4
    i = kernel[:, :unit_count]
    f = kernel[:, unit_count : 2 * unit_count]
    g = kernel[:, 2 * unit_count : 3 * unit_count]
    o = kernel[:, 3 * unit_count :]
    return i, f, g, o


def reorder_tf_lstm_kernel(kernel: np.ndarray) -> np.ndarray:
    """Ensure the gate order is (i, f, c, o) which matches Keras expectations."""

    i, f, g, o = split_lstm_gates(kernel)
    return np.concatenate([i, f, g, o], axis=-1)


def assign_lstm_cell(
    cell: tf.keras.layers.LSTMCell,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    input_dim: int,
) -> None:
    """Assign weights to a single ``tf.keras.layers.LSTMCell``."""

    kernel = pytorch_to_numpy(weight_ih).transpose(1, 0)
    recurrent_kernel = pytorch_to_numpy(weight_hh).transpose(1, 0)
    kernel = reorder_tf_lstm_kernel(kernel)
    recurrent_kernel = reorder_tf_lstm_kernel(recurrent_kernel)
    bias = combine_pytorch_biases(bias_ih, bias_hh)
    if cell.kernel.shape != kernel.shape:
        raise ValueError(
            f"Kernel shape mismatch. Expected {cell.kernel.shape}, got {kernel.shape}"
        )
    if cell.recurrent_kernel.shape != recurrent_kernel.shape:
        raise ValueError(
            "Recurrent kernel shape mismatch. Expected "
            f"{cell.recurrent_kernel.shape}, got {recurrent_kernel.shape}"
        )
    cell.kernel.assign(kernel)
    cell.recurrent_kernel.assign(recurrent_kernel)
    cell.bias.assign(bias)


def assign_blstm(
    layer: BLSTM,
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
    example_shape: Optional[Sequence[int]] = None,
) -> AssignmentReport:
    """Assign weights from a PyTorch BLSTM module to the TF ``BLSTM`` layer.

    Parameters
    ----------
    layer:
        The TensorFlow BLSTM layer that should receive the converted weights.
    state_dict:
        Mapping containing PyTorch weights. Only the relevant entries with the
        provided ``prefix`` will be consumed.
    prefix:
        Name prefix used in the PyTorch module (e.g. ``"lstm"``).
    example_shape:
        Optional input shape (B, C, T) used to build the layer if it has not
        been built yet.
    """

    if example_shape is None:
        example_shape = [1, layer.units, max(layer.max_steps or 2, 2)]
    ensure_layer_built(layer, example_shape)

    assigned: Dict[str, str] = {}
    unused = []
    missing = []

    forward_cells = layer.forward_cell.cells if layer.forward_cell else []
    backward_cells = layer.backward_cell.cells if layer.backward_cell else []

    consumed: set[str] = set()

    for idx in range(layer.layers):
        for direction, cells in [("", forward_cells), ("_reverse", backward_cells)]:
            key_base = f"{prefix}.weight_ih_l{idx}{direction}"
            if key_base not in state_dict:
                missing.append(key_base)
                continue
            weight_ih = state_dict[key_base]
            weight_hh_key = f"{prefix}.weight_hh_l{idx}{direction}"
            bias_ih_key = f"{prefix}.bias_ih_l{idx}{direction}"
            bias_hh_key = f"{prefix}.bias_hh_l{idx}{direction}"
            missing_keys = [
                key
                for key in [weight_hh_key, bias_ih_key, bias_hh_key]
                if key not in state_dict
            ]
            if missing_keys:
                missing.extend(missing_keys)
                continue
            weight_hh = state_dict[weight_hh_key]
            bias_ih = state_dict[bias_ih_key]
            bias_hh = state_dict[bias_hh_key]
            cell = cells[idx]
            assign_lstm_cell(
                cell,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                input_dim=weight_ih.shape[1],
            )
            assigned[key_base] = cell.name
            consumed.update({key_base, weight_hh_key, bias_ih_key, bias_hh_key})

    # Projection layer (PyTorch ``linear`` -> TF Dense)
    linear_weight_key = f"{prefix.replace('lstm', 'linear')}.weight"
    linear_bias_key = f"{prefix.replace('lstm', 'linear')}.bias"
    if linear_weight_key in state_dict and layer.proj is not None:
        assign_dense_weights(layer.proj, state_dict[linear_weight_key], state_dict[linear_bias_key])
        assigned[linear_weight_key] = layer.proj.name
        consumed.update({linear_weight_key, linear_bias_key})
    else:
        missing.extend([linear_weight_key, linear_bias_key])

    for key in state_dict.keys():
        if key.startswith(prefix) or key.startswith(prefix.replace("lstm", "linear")):
            if key not in consumed:
                unused.append(key)

    return AssignmentReport(assigned=assigned, missing=missing, unused=unused)


__all__ = [
    "load_pytorch_state_dict",
    "pytorch_to_tf_conv1d",
    "pytorch_to_tf_conv_transpose1d",
    "pytorch_to_tf_dense",
    "assign_conv1d_weights",
    "assign_conv_transpose1d_weights",
    "assign_dense_weights",
    "assign_group_norm",
    "assign_layer_scale",
    "assign_blstm",
    "AssignmentReport",
]
