"""Helpers to transfer Demucs weights from PyTorch to TensorFlow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf
import torch

from demucs_tf.blocks import DConv
from demucs_tf.layers.attention import LocalState
from demucs_tf.layers.conv import Conv1DWithPadding, ConvTranspose1D
from demucs_tf.layers.norm import GroupNorm, LayerScale
from demucs_tf.layers.recurrent import BLSTM
from demucs_tf.models.transformer import ChannelLastGroupNorm, LayerScale1D, MultiHeadAttention

if TYPE_CHECKING:  # pragma: no cover
    from demucs_tf.models.demucs import DemucsTF

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
    load_kwargs = {"map_location": "cpu"}
    try:
        obj = torch.load(checkpoint, weights_only=False, **load_kwargs)
    except TypeError:
        obj = torch.load(checkpoint, **load_kwargs)
    if isinstance(obj, Mapping):
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "state" in obj:
            state = obj["state"]
        else:
            state = obj
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


def pytorch_to_tf_conv2d(weight: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch ``nn.Conv2d`` kernel to TensorFlow format."""

    # PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
    # TensorFlow: (kernel_h, kernel_w, in_channels, out_channels)
    return pytorch_to_numpy(weight).transpose(2, 3, 1, 0)


def pytorch_to_tf_conv_transpose2d(weight: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch ``nn.ConvTranspose2d`` kernel to TensorFlow format."""

    # PyTorch: (in_channels, out_channels, kernel_h, kernel_w)
    # TensorFlow: (kernel_h, kernel_w, out_channels, in_channels)
    return pytorch_to_numpy(weight).transpose(2, 3, 1, 0)


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


def assign_conv2d_weights(
    layer: tf.keras.layers.Conv2D,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch Conv2d weights to a :class:`tf.keras.layers.Conv2D`."""

    if example_shape is None:
        example_shape = [None, weight.shape[2], weight.shape[3], weight.shape[1]]
    if not layer.built:
        layer.build(example_shape)
    kernel = pytorch_to_tf_conv2d(weight)
    layer.kernel.assign(kernel)
    if layer.use_bias and bias is not None and layer.bias is not None:
        layer.bias.assign(pytorch_to_numpy(bias))


def assign_conv_transpose2d_weights(
    layer: tf.keras.layers.Conv2DTranspose,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    example_shape: Optional[Sequence[int]] = None,
) -> None:
    """Assign PyTorch ConvTranspose2d weights to :class:`tf.keras.layers.Conv2DTranspose`."""

    if example_shape is None:
        height = max(int(weight.shape[2]), 1)
        width = max(int(weight.shape[3]), 1)
        example_shape = [None, height, width, weight.shape[0]]
    if not layer.built:
        layer.build(example_shape)
    kernel = pytorch_to_tf_conv_transpose2d(weight)
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


def assign_local_state(
    layer: LocalState,
    consume,
    record,
    prefix: str,
) -> None:
    """Assign weights to a :class:`LocalState` attention block."""

    components = [
        ("content", layer.content),
        ("query", layer.query),
        ("key", layer.key),
    ]
    for name, sublayer in components:
        weight_key = f"{prefix}.{name}.weight"
        bias_key = f"{prefix}.{name}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None or bias is None:
            continue
        assign_conv1d_weights(sublayer, weight, bias)
        record(weight_key, sublayer.name or f"{layer.name}_{name}")
        record(bias_key, sublayer.name or f"{layer.name}_{name}")

    if layer.query_freqs is not None:
        weight_key = f"{prefix}.query_freqs.weight"
        bias_key = f"{prefix}.query_freqs.bias"
        weight = consume(weight_key, required=False)
        bias = consume(bias_key, required=False)
        if weight is not None and bias is not None:
            assign_conv1d_weights(layer.query_freqs, weight, bias)
            record(weight_key, layer.query_freqs.name or f"{layer.name}_query_freqs")
            record(bias_key, layer.query_freqs.name or f"{layer.name}_query_freqs")

    if layer.query_decay is not None:
        weight_key = f"{prefix}.query_decay.weight"
        bias_key = f"{prefix}.query_decay.bias"
        weight = consume(weight_key, required=False)
        bias = consume(bias_key, required=False)
        if weight is not None and bias is not None:
            assign_conv1d_weights(layer.query_decay, weight, bias)
            record(weight_key, layer.query_decay.name or f"{layer.name}_query_decay")
            record(bias_key, layer.query_decay.name or f"{layer.name}_query_decay")

    weight_key = f"{prefix}.proj.weight"
    bias_key = f"{prefix}.proj.bias"
    weight = consume(weight_key)
    bias = consume(bias_key)
    if weight is not None and bias is not None:
        assign_conv1d_weights(layer.proj, weight, bias)
        record(weight_key, layer.proj.name or f"{layer.name}_proj")
        record(bias_key, layer.proj.name or f"{layer.name}_proj")


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

    consumed: set[str] = set()

    bidir_layers = getattr(layer, "rnn_layers", [])
    for idx, bidir in enumerate(bidir_layers):
        forward_lstm = getattr(bidir, "forward_layer", None)
        backward_lstm = getattr(bidir, "backward_layer", None)
        if forward_lstm is None or backward_lstm is None:
            missing.append(f"{prefix}.weight_ih_l{idx}")
            continue

        # Forward direction
        keys = {
            "weight_ih": f"{prefix}.weight_ih_l{idx}",
            "weight_hh": f"{prefix}.weight_hh_l{idx}",
            "bias_ih": f"{prefix}.bias_ih_l{idx}",
            "bias_hh": f"{prefix}.bias_hh_l{idx}",
        }
        if all(k in state_dict for k in keys.values()):
            assign_lstm_cell(
                forward_lstm.cell,
                state_dict[keys["weight_ih"]],
                state_dict[keys["weight_hh"]],
                state_dict[keys["bias_ih"]],
                state_dict[keys["bias_hh"]],
                input_dim=state_dict[keys["weight_ih"]].shape[1],
            )
            for label, key in keys.items():
                assigned[key] = forward_lstm.name
                consumed.add(key)
        else:
            for key in keys.values():
                if key not in state_dict:
                    missing.append(key)

        # Backward direction
        keys_rev = {
            "weight_ih": f"{prefix}.weight_ih_l{idx}_reverse",
            "weight_hh": f"{prefix}.weight_hh_l{idx}_reverse",
            "bias_ih": f"{prefix}.bias_ih_l{idx}_reverse",
            "bias_hh": f"{prefix}.bias_hh_l{idx}_reverse",
        }
        if all(k in state_dict for k in keys_rev.values()):
            assign_lstm_cell(
                backward_lstm.cell,
                state_dict[keys_rev["weight_ih"]],
                state_dict[keys_rev["weight_hh"]],
                state_dict[keys_rev["bias_ih"]],
                state_dict[keys_rev["bias_hh"]],
                input_dim=state_dict[keys_rev["weight_ih"]].shape[1],
            )
            for key in keys_rev.values():
                assigned[key] = backward_lstm.name
                consumed.add(key)
        else:
            for key in keys_rev.values():
                if key not in state_dict:
                    missing.append(key)

    # Projection layer (PyTorch ``linear`` -> TF Dense)
    linear_weight_key = f"{prefix.replace('lstm', 'linear')}.weight"
    linear_bias_key = f"{prefix.replace('lstm', 'linear')}.bias"
    if linear_weight_key in state_dict and layer.proj is not None:
        assign_dense_weights(layer.proj, state_dict[linear_weight_key], state_dict[linear_bias_key])
        assigned[linear_weight_key] = layer.proj.name
        assigned[linear_bias_key] = layer.proj.name
        consumed.update({linear_weight_key, linear_bias_key})
    else:
        missing.extend([linear_weight_key, linear_bias_key])

    for key in state_dict.keys():
        if key.startswith(prefix) or key.startswith(prefix.replace("lstm", "linear")):
            if key not in consumed:
                unused.append(key)

    return AssignmentReport(assigned=assigned, missing=missing, unused=unused)


def load_demucs_tf_weights(model: "DemucsTF", checkpoint: Path | str) -> AssignmentReport:
    """Load a PyTorch Demucs checkpoint into the TensorFlow model."""

    state = load_pytorch_state_dict(checkpoint)

    # Backward compatibility with early Demucs checkpoints.
    for idx in range(model.depth):
        for branch in ("encoder", "decoder"):
            for suffix in ("weight", "bias"):
                old_key = f"{branch}.{idx}.2.{suffix}"
                new_key = f"{branch}.{idx}.3.{suffix}"
                if old_key in state and new_key not in state:
                    state[new_key] = state.pop(old_key)

    assigned: Dict[str, str] = {}
    missing: List[str] = []
    unused = set(state.keys())

    def consume(key: str, required: bool = True) -> Optional[torch.Tensor]:
        value = state.get(key)
        if value is None:
            if required:
                missing.append(key)
            return None
        unused.discard(key)
        return value

    def record(key: str, target: str) -> None:
        assigned[key] = target

    def assign_conv(layer: Conv1DWithPadding, prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None or bias is None:
            return
        assign_conv1d_weights(layer, weight, bias)
        record(weight_key, layer.name or layer.__class__.__name__)
        record(bias_key, layer.name or layer.__class__.__name__)

    def assign_conv_transpose(layer: ConvTranspose1D, prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None or bias is None:
            return
        assign_conv_transpose1d_weights(layer, weight, bias)
        record(weight_key, layer.name or layer.__class__.__name__)
        record(bias_key, layer.name or layer.__class__.__name__)

    def assign_conv2d_layer(layer: tf.keras.layers.Conv2D, prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None:
            return
        assign_conv2d_weights(layer, weight, bias)
        record(weight_key, layer.name or layer.__class__.__name__)
        if bias is not None:
            record(bias_key, layer.name or layer.__class__.__name__)

    def assign_conv_transpose2d_layer(layer: tf.keras.layers.Conv2DTranspose, prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None:
            return
        assign_conv_transpose2d_weights(layer, weight, bias)
        record(weight_key, layer.name or layer.__class__.__name__)
        if bias is not None:
            record(bias_key, layer.name or layer.__class__.__name__)

    def assign_group_norm_layer(layer: GroupNorm, prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None or bias is None:
            return
        assign_group_norm(layer, weight, bias)
        record(weight_key, layer.name or layer.__class__.__name__)
        record(bias_key, layer.name or layer.__class__.__name__)

    def assign_layerscale(layer: LayerScale, key: str) -> None:
        scale = consume(key)
        if scale is None:
            return
        assign_layer_scale(layer, scale)
        record(key, layer.name or layer.__class__.__name__)

    def assign_dconv(layer: DConv, prefix: str) -> None:
        for block_index, block in enumerate(layer.blocks):
            block_prefix = f"{prefix}.layers.{block_index}"

            conv1_prefix = f"{block_prefix}.0"
            assign_conv(block.conv1, conv1_prefix)

            if getattr(block, "norm1", None) is not None:
                assign_group_norm_layer(block.norm1, f"{block_prefix}.1")

            # Determine positions of optional modules and downstream layers.
            has_attn = getattr(block, "attn", None) is not None
            has_lstm = getattr(block, "lstm", None) is not None
            shift = int(has_attn) + int(has_lstm)

            if has_lstm:
                lstm_prefix = f"{block_prefix}.3.lstm"
                lstm_layer = block.lstm
                if lstm_layer is not None:
                    report = assign_blstm(
                        lstm_layer,
                        state,
                        lstm_prefix,
                        example_shape=[1, lstm_layer.units, max(lstm_layer.max_steps or 8, 8)],
                    )
                    missing.extend(report.missing)
                    for key, target in report.assigned.items():
                        unused.discard(key)
                        record(key, target)

            if has_attn:
                attn_index = 4 if has_lstm else 3
                attn_prefix = f"{block_prefix}.{attn_index}"
                attn_layer = block.attn
                if attn_layer is not None:
                    assign_local_state(attn_layer, consume, record, attn_prefix)

            conv2_index = 3 + shift
            norm2_index = 4 + shift
            scale_index = 6 + shift

            assign_conv(block.conv2, f"{block_prefix}.{conv2_index}")

            if getattr(block, "norm2", None) is not None:
                assign_group_norm_layer(block.norm2, f"{block_prefix}.{norm2_index}")

            assign_layerscale(block.scale, f"{block_prefix}.{scale_index}.scale")

    def assign_layer_norm_layer(
        layer: tf.keras.layers.Layer,
        prefix: str,
        example_length: int = 4,
    ) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = consume(weight_key)
        bias = consume(bias_key)
        if weight is None or bias is None:
            return
        channels = weight.shape[0]
        example_shape = [1, example_length, channels]
        if isinstance(layer, ChannelLastGroupNorm):
            ensure_layer_built(layer, example_shape)
            layer.gamma.assign(pytorch_to_numpy(weight))
            layer.beta.assign(pytorch_to_numpy(bias))
        elif isinstance(layer, GroupNorm):
            assign_group_norm(layer, weight, bias, example_shape)
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            ensure_layer_built(layer, example_shape)
            layer.gamma.assign(pytorch_to_numpy(weight))
            layer.beta.assign(pytorch_to_numpy(bias))
        else:
            return
        record(weight_key, layer.name or layer.__class__.__name__)
        record(bias_key, layer.name or layer.__class__.__name__)

    def assign_layerscale1d(layer: LayerScale1D, key: str) -> None:
        scale = consume(key)
        if scale is None:
            return
        channels = scale.shape[0]
        example_shape = [1, 4, channels]
        ensure_layer_built(layer, example_shape)
        layer.scale.assign(pytorch_to_numpy(scale))
        record(key, layer.name or layer.__class__.__name__)

    def assign_multi_head_attention_layer(
        layer: MultiHeadAttention,
        prefix: str,
        example_length: int = 4,
    ) -> None:
        weight_key = f"{prefix}.in_proj_weight"
        bias_key = f"{prefix}.in_proj_bias"
        out_weight_key = f"{prefix}.out_proj.weight"
        out_bias_key = f"{prefix}.out_proj.bias"
        in_proj_weight = consume(weight_key)
        in_proj_bias = consume(bias_key)
        out_weight = consume(out_weight_key)
        out_bias = consume(out_bias_key)
        if in_proj_weight is None or in_proj_bias is None or out_weight is None or out_bias is None:
            return
        embed_dim = in_proj_weight.shape[1]
        example_shape = [1, example_length, embed_dim]
        if not layer.built:
            dummy = tf.zeros(example_shape, dtype=tf.float32)
            layer(dummy, dummy, dummy)
        w_q, w_k, w_v = torch.chunk(in_proj_weight, 3, dim=0)
        b_q, b_k, b_v = torch.chunk(in_proj_bias, 3, dim=0)
        assign_dense_weights(layer.q_proj, w_q, b_q, example_shape=[1, embed_dim])
        assign_dense_weights(layer.k_proj, w_k, b_k, example_shape=[1, embed_dim])
        assign_dense_weights(layer.v_proj, w_v, b_v, example_shape=[1, embed_dim])
        assign_dense_weights(layer.out_proj, out_weight, out_bias, example_shape=[1, embed_dim])
        record(weight_key, layer.name or f"{layer.__class__.__name__}_qkv")
        record(bias_key, layer.name or f"{layer.__class__.__name__}_qkv")
        record(out_weight_key, layer.out_proj.name or f"{layer.__class__.__name__}_out")
        record(out_bias_key, layer.out_proj.name or f"{layer.__class__.__name__}_out")

    def assign_transformer_block(layer: tf.keras.layers.Layer, prefix: str) -> None:
        if hasattr(layer, "attn"):
            assign_multi_head_attention_layer(layer.attn, f"{prefix}.self_attn")
        if hasattr(layer, "cross_attn"):
            assign_multi_head_attention_layer(layer.cross_attn, f"{prefix}.cross_attn")

        linear1_key = f"{prefix}.linear1.weight"
        linear1_bias_key = f"{prefix}.linear1.bias"
        linear2_key = f"{prefix}.linear2.weight"
        linear2_bias_key = f"{prefix}.linear2.bias"
        weight1 = consume(linear1_key)
        bias1 = consume(linear1_bias_key)
        weight2 = consume(linear2_key)
        bias2 = consume(linear2_bias_key)
        if weight1 is not None and bias1 is not None:
            assign_dense_weights(layer.linear1, weight1, bias1)
            record(linear1_key, layer.linear1.name or f"{layer.__class__.__name__}_ff1")
            record(linear1_bias_key, layer.linear1.name or f"{layer.__class__.__name__}_ff1")
        if weight2 is not None and bias2 is not None:
            assign_dense_weights(layer.linear2, weight2, bias2)
            record(linear2_key, layer.linear2.name or f"{layer.__class__.__name__}_ff2")
            record(linear2_bias_key, layer.linear2.name or f"{layer.__class__.__name__}_ff2")

        for norm_name in ("norm1", "norm2", "norm3", "norm_out"):
            norm_layer = getattr(layer, norm_name, None)
            if norm_layer is not None:
                assign_layer_norm_layer(norm_layer, f"{prefix}.{norm_name}")

        for gamma_name in ("gamma_1", "gamma_2"):
            gamma_layer = getattr(layer, gamma_name, None)
            if isinstance(gamma_layer, LayerScale1D):
                assign_layerscale1d(gamma_layer, f"{prefix}.{gamma_name}.scale")

    def assign_cross_transformer(transformer, prefix: str = "crosstransformer") -> None:
        if transformer is None:
            return

        if getattr(transformer, "position_embeddings", None) is not None:
            weight_key = f"{prefix}.position_embeddings.embedding.weight"
            weight = consume(weight_key, required=False)
            if weight is not None:
                layer = transformer.position_embeddings
                if not layer.built:
                    layer(tf.zeros([1, 1], dtype=tf.int32))
                layer.embedding.embeddings.assign(pytorch_to_numpy(weight))
                record(weight_key, layer.name or f"{prefix}_position_embeddings")

        norm_pairs = [
            (transformer.norm_in, f"{prefix}.norm_in"),
            (transformer.norm_in_t, f"{prefix}.norm_in_t"),
        ]
        for norm_layer, norm_prefix in norm_pairs:
            if norm_layer is not None:
                assign_layer_norm_layer(norm_layer, norm_prefix)

        for idx, layer in enumerate(getattr(transformer, "layers_spec", [])):
            assign_transformer_block(layer, f"{prefix}.layers.{idx}")

        for idx, layer in enumerate(getattr(transformer, "layers_time", [])):
            assign_transformer_block(layer, f"{prefix}.layers_t.{idx}")

    def assign_encoder_branch(layers: Sequence[tf.keras.layers.Layer], prefix_root: str) -> None:
        for idx, layer in enumerate(layers):
            prefix = f"{prefix_root}.{idx}"
            conv_layer = getattr(layer, "conv", None)
            if isinstance(conv_layer, Conv1DWithPadding):
                assign_conv(conv_layer, f"{prefix}.conv")
            elif isinstance(conv_layer, tf.keras.layers.Conv2D):
                assign_conv2d_layer(conv_layer, f"{prefix}.conv")

            rewrite_layer = getattr(layer, "rewrite", None)
            if isinstance(rewrite_layer, Conv1DWithPadding):
                assign_conv(rewrite_layer, f"{prefix}.rewrite")
            elif isinstance(rewrite_layer, tf.keras.layers.Conv2D):
                assign_conv2d_layer(rewrite_layer, f"{prefix}.rewrite")

            norm1 = getattr(layer, "norm1", None)
            norm1_prefix = f"{prefix}.norm1"
            if isinstance(norm1, GroupNorm) and norm1_prefix in state:
                assign_group_norm_layer(norm1, norm1_prefix)

            norm2 = getattr(layer, "norm2", None)
            norm2_prefix = f"{prefix}.norm2"
            if isinstance(norm2, GroupNorm) and norm2_prefix in state:
                assign_group_norm_layer(norm2, norm2_prefix)

            dconv_layer = getattr(layer, "dconv", None)
            if isinstance(dconv_layer, DConv):
                assign_dconv(dconv_layer, f"{prefix}.dconv")

    def assign_decoder_branch(layers: Sequence[tf.keras.layers.Layer], prefix_root: str) -> None:
        for idx, layer in enumerate(layers):
            prefix = f"{prefix_root}.{idx}"
            rewrite_layer = getattr(layer, "rewrite", None)
            if isinstance(rewrite_layer, Conv1DWithPadding):
                assign_conv(rewrite_layer, f"{prefix}.rewrite")
            elif isinstance(rewrite_layer, tf.keras.layers.Conv2D):
                assign_conv2d_layer(rewrite_layer, f"{prefix}.rewrite")

            norm1 = getattr(layer, "norm1", None)
            norm1_prefix = f"{prefix}.norm1"
            if isinstance(norm1, GroupNorm) and norm1_prefix in state:
                assign_group_norm_layer(norm1, norm1_prefix)

            dconv_layer = getattr(layer, "dconv", None)
            if isinstance(dconv_layer, DConv):
                assign_dconv(dconv_layer, f"{prefix}.dconv")

            conv_t_layer = getattr(layer, "conv_transpose", None)
            if isinstance(conv_t_layer, ConvTranspose1D):
                assign_conv_transpose(conv_t_layer, f"{prefix}.conv_tr")
            elif isinstance(conv_t_layer, tf.keras.layers.Conv2DTranspose):
                assign_conv_transpose2d_layer(conv_t_layer, f"{prefix}.conv_tr")

            norm2 = getattr(layer, "norm2", None)
            norm2_prefix = f"{prefix}.norm2"
            if isinstance(norm2, GroupNorm) and norm2_prefix in state:
                assign_group_norm_layer(norm2, norm2_prefix)

    # Hybrid encoder/decoder branches ----------------------------------------------
    assign_encoder_branch(model.encoder_layers, "encoder")
    assign_encoder_branch(model.tencoder_layers, "tencoder")
    assign_decoder_branch(model.decoder_layers, "decoder")
    assign_decoder_branch(model.tdecoder_layers, "tdecoder")

    # Bottleneck BLSTM ---------------------------------------------------------------
    if model.lstm is not None:
        report = assign_blstm(model.lstm, state, "lstm.lstm")
        missing.extend(report.missing)
        for key, target in report.assigned.items():
            unused.discard(key)
            record(key, target)

    if getattr(model, "channel_upsampler", None) is not None:
        assign_conv(model.channel_upsampler, "channel_upsampler")
    if getattr(model, "channel_downsampler", None) is not None:
        assign_conv(model.channel_downsampler, "channel_downsampler")
    if getattr(model, "channel_upsampler_t", None) is not None:
        assign_conv(model.channel_upsampler_t, "channel_upsampler_t")
    if getattr(model, "channel_downsampler_t", None) is not None:
        assign_conv(model.channel_downsampler_t, "channel_downsampler_t")

    if getattr(model, "freq_emb", None) is not None:
        weight_key = "freq_emb.embedding.weight"
        weight = consume(weight_key, required=False)
        if weight is not None:
            freq_emb_layer = model.freq_emb
            if not freq_emb_layer.built:
                freq_emb_layer(tf.zeros([1, 1], dtype=tf.int32))
            freq_emb_layer.embedding.embeddings.assign(pytorch_to_numpy(weight))
            record(weight_key, freq_emb_layer.name or "freq_emb")

    if getattr(model, "crosstransformer", None) is not None:
        assign_cross_transformer(model.crosstransformer)

    return AssignmentReport(
        assigned=assigned,
        missing=missing,
        unused=sorted(unused),
    )


__all__ = [
    "load_pytorch_state_dict",
    "pytorch_to_tf_conv1d",
    "pytorch_to_tf_conv_transpose1d",
    "pytorch_to_tf_conv2d",
    "pytorch_to_tf_conv_transpose2d",
    "pytorch_to_tf_dense",
    "assign_conv1d_weights",
    "assign_conv_transpose1d_weights",
    "assign_conv2d_weights",
    "assign_conv_transpose2d_weights",
    "assign_dense_weights",
    "assign_group_norm",
    "assign_layer_scale",
    "assign_blstm",
    "assign_local_state",
    "load_demucs_tf_weights",
    "AssignmentReport",
]
