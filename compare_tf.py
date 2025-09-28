#!/usr/bin/env python3
"""Layer-by-layer comparison between PyTorch HTDemucs and TensorFlow port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import tensorflow as tf

from demucs.htdemucs import HTDemucs
from demucs_tf.models import HTDemucsTF


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = dict(config)
    config.pop("model", None)
    return config


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.unsqueeze(0)


def _align_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    if len(a.shape) != len(b.shape):
        raise ValueError(f"Cannot align tensors with different ranks: {a.shape} vs {b.shape}")
    slices_a = []
    slices_b = []
    for dim_a, dim_b in zip(a.shape, b.shape):
        target = min(dim_a, dim_b)
        start_a = max((dim_a - target) // 2, 0)
        start_b = max((dim_b - target) // 2, 0)
        slices_a.append(slice(start_a, start_a + target))
        slices_b.append(slice(start_b, start_b + target))
    return a[tuple(slices_a)], b[tuple(slices_b)]


def numpy_mae(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a_aligned, b_aligned = _align_arrays(a, b)
    diff = np.abs(a_aligned - b_aligned)
    return float(diff.mean()), float(diff.max())


def inspect_torch_encoder_layer(
    layer: torch.nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    inject: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        weight_dtype = layer.conv.weight.dtype if hasattr(layer, "conv") else torch.float32
        x = torch.from_numpy(inputs).to(device=device, dtype=weight_dtype)
        inj_tensor: Optional[torch.Tensor]
        if inject is not None:
            inj_tensor = torch.from_numpy(inject).to(device=device, dtype=weight_dtype)
        else:
            inj_tensor = None

        outputs: Dict[str, np.ndarray] = {}
        y = x
        if not getattr(layer, "freq", False) and y.dim() == 4:
            b, c, fr, t = y.shape
            y = y.view(b, -1, t)
        if not getattr(layer, "freq", False):
            length = y.shape[-1]
            remainder = length % layer.stride
            if remainder != 0:
                pad_right = layer.stride - remainder
                y = F.pad(y, (0, pad_right))
            prepared = y
        else:
            prepared = y
            pad_amount = getattr(layer, "pad", 0)
            if pad_amount:
                prepared = F.pad(prepared, (0, 0, pad_amount, pad_amount))
        outputs["prepared"] = prepared.detach().cpu().numpy()

        conv_out = layer.conv(y)
        outputs["conv"] = conv_out.detach().cpu().numpy()
        if getattr(layer, "empty", False):
            outputs["final"] = outputs["conv"]
            return outputs

        current = conv_out
        if inj_tensor is not None:
            addend = inj_tensor
            if addend.dim() == 3 and current.dim() == 4:
                addend = addend.unsqueeze(2)
            current = current + addend
        outputs["post_inject"] = current.detach().cpu().numpy()

        norm1 = getattr(layer, "norm1", None)
        if norm1 is not None:
            current = norm1(current)
        outputs["norm1"] = current.detach().cpu().numpy()

        current = F.gelu(current)
        outputs["gelu"] = current.detach().cpu().numpy()

        dconv = getattr(layer, "dconv", None)
        if dconv is not None:
            if getattr(layer, "freq", False):
                b, c, fr, t = current.shape
                flat = current.permute(0, 2, 1, 3).reshape(-1, c, t)
                flat = dconv(flat)
                current = flat.view(b, fr, c, t).permute(0, 2, 1, 3)
            else:
                current = dconv(current)
        outputs["dconv"] = current.detach().cpu().numpy()

        rewrite = getattr(layer, "rewrite", None)
        if rewrite is None:
            outputs["final"] = outputs["dconv"]
            return outputs

        rewritten = rewrite(current)
        outputs["rewrite"] = rewritten.detach().cpu().numpy()

        norm2 = getattr(layer, "norm2", None)
        if norm2 is not None:
            rewritten = norm2(rewritten)
        outputs["norm2"] = rewritten.detach().cpu().numpy()

        glu_out = F.glu(rewritten, dim=1)
        outputs["glu"] = glu_out.detach().cpu().numpy()
        outputs["final"] = outputs["glu"]
        return outputs


class TorchCapture:
    """Registers forward hooks on HTDemucs modules to capture inputs and outputs."""

    def __init__(self, model: HTDemucs):
        self.model = model
        self.storage: Dict[str, List[np.ndarray]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def _register_module_list(self, modules: Iterable[torch.nn.Module], prefix: str) -> None:
        for idx, module in enumerate(modules):
            name = f"{prefix}.{idx}"
            pre_handle = module.register_forward_pre_hook(self._make_pre_hook(name))
            self.handles.append(pre_handle)
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_pre_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs):  # noqa: ANN001
            if not inputs:
                return
            primary = inputs[0]
            if torch.is_tensor(primary):
                self.storage.setdefault(f"{name}.input", []).append(primary.detach().cpu().numpy())
            if len(inputs) > 1:
                secondary = inputs[1]
                if torch.is_tensor(secondary):
                    self.storage.setdefault(f"{name}.inject", []).append(secondary.detach().cpu().numpy())
                else:
                    self.storage.setdefault(f"{name}.inject", []).append(None)
        return hook

    def _make_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs, output):  # noqa: ANN001
            if isinstance(output, tuple):
                primary = output[0]
                secondary = output[1]
            else:
                primary = output
                secondary = None
            self.storage.setdefault(name, []).append(primary.detach().cpu().numpy())
            if torch.is_tensor(secondary):
                self.storage.setdefault(f"{name}.pre", []).append(secondary.detach().cpu().numpy())
            if name == "crosstransformer":
                captured_inputs = []
                for item in inputs:
                    if torch.is_tensor(item):
                        captured_inputs.append(item.detach().cpu().numpy())
                    else:
                        captured_inputs.append(item)
                self.storage.setdefault(f"{name}.input", []).append(tuple(captured_inputs))
        return hook

    def _register(self) -> None:
        self._register_module_list(self.model.encoder, "encoder")
        if hasattr(self.model, "tencoder"):
            self._register_module_list(self.model.tencoder, "tencoder")
        self._register_module_list(self.model.decoder, "decoder")
        if hasattr(self.model, "tdecoder"):
            self._register_module_list(self.model.tdecoder, "tdecoder")
        if getattr(self.model, "crosstransformer", None) is not None:
            handle = self.model.crosstransformer.register_forward_hook(self._make_hook("crosstransformer"))
            self.handles.append(handle)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepare_models(config: Dict, checkpoint_path: Path) -> Tuple[HTDemucs, HTDemucsTF]:
    torch_model = HTDemucs(**config)
    load_kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        state = torch.load(checkpoint_path, **load_kwargs)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "state" in state and isinstance(state["state"], dict):
        state = state["state"]
    torch_model.load_state_dict(state)
    torch_model.use_train_segment = False
    torch_model.eval()

    tf_model = HTDemucsTF(**config)
    tf_model.use_train_segment = False
    tf_model.enable_debug()
    report = tf_model.load_pytorch_checkpoint(checkpoint_path)
    missing = getattr(report, "missing", [])
    unused = getattr(report, "unused", [])
    print(f"[loader] missing TF assignments: {len(missing)}")
    print(f"[loader] unused PyTorch weights: {len(unused)}")

    return torch_model, tf_model


def compare_layers(
    name: str,
    torch_values: List[Optional[np.ndarray]],
    tf_values: List[Optional[np.ndarray]],
) -> None:
    count = min(len(torch_values), len(tf_values))
    print(f"\n{name} layers: {len(torch_values)} torch / {len(tf_values)} tf")
    for idx in range(count):
        torch_arr = torch_values[idx]
        tf_arr = tf_values[idx]
        if torch_arr is None or tf_arr is None:
            print(f"  {name}[{idx:02d}] missing data (torch={torch_arr is not None}, tf={tf_arr is not None})")
            continue
        aligned_torch, aligned_tf = _align_arrays(torch_arr, tf_arr)
        mae, max_abs = numpy_mae(aligned_torch, aligned_tf)
        print(
            f"  {name}[{idx:02d}] torch={torch_arr.shape} tf={tf_arr.shape} aligned={aligned_torch.shape} mae={mae:.6e} max={max_abs:.6e}"
        )
    if len(torch_values) != len(tf_values):
        print(f"  ⚠ length mismatch: torch={len(torch_values)} tf={len(tf_values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("remote/955717e8-8726e21a.th"))
    parser.add_argument("--config", type=Path, default=Path("remote/955717e8-8726e21a.json"))
    parser.add_argument("--audio", type=Path, default=Path("so_short.mp3"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")

    config = load_config(args.config)
    torch.manual_seed(0)
    mix = load_audio(args.audio, config["samplerate"])
    mix = mix.to(args.device)

    torch_model, tf_model = prepare_models(config, args.checkpoint)
    torch_model.to(args.device)
    stage_indices = [0, 1]
    tf_model._debug_encoder_stage_targets = set(stage_indices)

    with torch.no_grad():
        capture = TorchCapture(torch_model)
        spec = torch_model._spec(mix)  # type: ignore[attr-defined]
        mag = torch_model._magnitude(spec)  # type: ignore[attr-defined]
        torch_spec_out, torch_time_out = torch_model.forward_core(mag, mix)
        capture.close()

    storage = capture.storage

    tf_mix = tf.convert_to_tensor(mix.cpu().numpy())
    tf_out = tf_model(tf_mix, training=False)

    with torch.no_grad():
        torch_mag = mag.detach().cpu()
        mag_mean = torch_mag.mean(dim=(1, 2, 3), keepdim=True)
        mag_std = torch_mag.std(dim=(1, 2, 3), keepdim=True)
        torch_mag_norm = (torch_mag - mag_mean) / (1e-5 + mag_std)
    torch_mag_norm_np = torch_mag_norm.numpy()

    def first_entry(key: str):  # noqa: ANN001
        values = storage.get(key)
        if values:
            return values[0]
        return None

    encoder_torch = [first_entry(f"encoder.{i}") for i in range(len(torch_model.encoder))]
    decoder_torch = [first_entry(f"decoder.{i}") for i in range(len(torch_model.decoder))]
    decoder_pre_torch = [first_entry(f"decoder.{i}.pre") for i in range(len(torch_model.decoder))]
    tencoder_torch = []
    if hasattr(torch_model, "tencoder"):
        tencoder_torch = [first_entry(f"tencoder.{i}") for i in range(len(torch_model.tencoder))]
    tdecoder_torch = []
    if hasattr(torch_model, "tdecoder"):
        tdecoder_torch = [first_entry(f"tdecoder.{i}") for i in range(len(torch_model.tdecoder))]
    transformer_x = first_entry("crosstransformer")
    transformer_xt = first_entry("crosstransformer.pre")

    encoder_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_encoder", [])]
    encoder_inputs_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_encoder_inputs", [])]
    decoder_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_decoder", [])]
    decoder_pre_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_decoder_pre", [])]
    tencoder_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_tencoder", [])]
    tdecoder_tf = [tensor.numpy() for tensor in getattr(tf_model, "_debug_tdecoder", [])]
    transformer_tf_in = getattr(tf_model, "_debug_crosstransformer_in", None)
    transformer_tf_out = getattr(tf_model, "_debug_crosstransformer_out", None)

    compare_layers("encoder", encoder_torch, encoder_tf)
    compare_layers("tencoder", tencoder_torch, tencoder_tf)
    compare_layers("decoder", decoder_torch, decoder_tf)
    compare_layers("decoder_pre", decoder_pre_torch, decoder_pre_tf)
    compare_layers("tdecoder", tdecoder_torch, tdecoder_tf)

    transformer_input = storage.get("crosstransformer.input")
    if transformer_input and transformer_tf_in is not None:
        torch_in_spec, torch_in_time = transformer_input[0]
        tf_in_spec, tf_in_time = transformer_tf_in
        print(
            f"\ntransformer in shapes: torch_spec={torch_in_spec.shape} "
            f"tf_spec={tf_in_spec.shape} torch_time={torch_in_time.shape} tf_time={tf_in_time.shape}"
        )
        mae_spec_in, max_spec_in = numpy_mae(torch_in_spec, tf_in_spec.numpy())
        mae_time_in, max_time_in = numpy_mae(torch_in_time, tf_in_time.numpy())
        print("\ntransformer in:")
        print(f"  spec mae={mae_spec_in:.6e} max={max_spec_in:.6e}")
        print(f"  time mae={mae_time_in:.6e} max={max_time_in:.6e}")

    if hasattr(torch_model, "channel_upsampler") and hasattr(tf_model, "channel_upsampler"):
        torch_kernel = torch_model.channel_upsampler.weight.detach().cpu().numpy()
        tf_kernel = tf_model.channel_upsampler.conv.kernel.numpy()
        torch_bias = torch_model.channel_upsampler.bias.detach().cpu().numpy()
        tf_bias = tf_model.channel_upsampler.conv.bias.numpy() if tf_model.channel_upsampler.conv.bias is not None else None
        kernel_diff = np.abs(np.transpose(torch_kernel, (2, 1, 0)) - tf_kernel)
        print(
            "\nchannel upsampler weights:",
            f"kernel_mae={kernel_diff.mean():.6e}",
            f"kernel_max={kernel_diff.max():.6e}",
        )
        if tf_bias is not None:
            bias_diff = np.abs(torch_bias - tf_bias)
            print(f"  bias_mae={bias_diff.mean():.6e} bias_max={bias_diff.max():.6e}")

    # Inspect intermediate outputs for selected encoder layers.
    for layer_index in stage_indices:
        if layer_index >= len(torch_model.encoder):
            continue

        torch_input_prev_np = first_entry(f"encoder.{layer_index}.input")
        if torch_input_prev_np is None:
            if layer_index == 0:
                torch_input_prev_np = torch_mag_norm_np
            elif layer_index - 1 < len(encoder_torch):
                torch_input_prev_np = encoder_torch[layer_index - 1]

        if torch_input_prev_np is None:
            print(f"\nencoder[{layer_index}] stage breakdown: missing torch input capture")
            continue

        torch_layer = torch_model.encoder[layer_index]
        torch_inject_raw = first_entry(f"encoder.{layer_index}.inject")
        torch_inject_np: Optional[np.ndarray]
        if isinstance(torch_inject_raw, np.ndarray):
            torch_inject_np = torch_inject_raw
        else:
            torch_inject_np = None
        torch_stage = inspect_torch_encoder_layer(
            torch_layer,
            torch_input_prev_np,
            torch.device(args.device),
            inject=torch_inject_np,
        )

        tf_stage_map_shared: Dict[str, np.ndarray] = {}
        tf_stage_map_actual: Dict[str, np.ndarray] = {}
        if layer_index < len(tf_model.encoder_layers):
            tf_layer = tf_model.encoder_layers[layer_index]

            shared_input = tf.convert_to_tensor(torch_input_prev_np)
            shared_inject = tf.convert_to_tensor(torch_inject_np) if torch_inject_np is not None else None
            stage_debug_tf_shared: Dict[str, tf.Tensor] = {}
            _ = tf_layer.call(shared_input, inject=shared_inject, training=False, debug=stage_debug_tf_shared)
            tf_stage_map_shared = {key: value.numpy() for key, value in stage_debug_tf_shared.items()}

            if layer_index < len(encoder_inputs_tf):
                tf_input_prev_np = encoder_inputs_tf[layer_index]
                tf_input_prev = tf.convert_to_tensor(tf_input_prev_np)
                tf_inject_actual = None
                if hasattr(tf_model, "tencoder_layers") and layer_index < len(tf_model.tencoder_layers):
                    tf_tenc_layer = tf_model.tencoder_layers[layer_index]
                    if getattr(tf_tenc_layer, "empty", False) and layer_index < len(tencoder_tf):
                        tf_inject_actual = tf.convert_to_tensor(tencoder_tf[layer_index])
                stage_debug_tf_actual: Dict[str, tf.Tensor] = {}
                _ = tf_layer.call(tf_input_prev, inject=tf_inject_actual, training=False, debug=stage_debug_tf_actual)
                tf_stage_map_actual = {key: value.numpy() for key, value in stage_debug_tf_actual.items()}

        if tf_stage_map_shared:
            print(f"\nencoder[{layer_index}] stage breakdown:")
            shared_keys = sorted(set(torch_stage.keys()) & set(tf_stage_map_shared.keys()))
            for key in shared_keys:
                mae, max_abs = numpy_mae(torch_stage[key], tf_stage_map_shared[key])
                print(f"  {key}: mae={mae:.6e} max={max_abs:.6e}")
            missing_tf = set(torch_stage.keys()) - set(tf_stage_map_shared.keys())
            missing_torch = set(tf_stage_map_shared.keys()) - set(torch_stage.keys())
            if missing_tf:
                print(f"  ⚠ missing TF stages: {sorted(missing_tf)}")
            if missing_torch:
                print(f"  ⚠ missing torch stages: {sorted(missing_torch)}")

            if tf_stage_map_actual:
                actual_keys = sorted(set(torch_stage.keys()) & set(tf_stage_map_actual.keys()))
                if actual_keys:
                    print("  torch vs tf actual:")
                    for key in actual_keys:
                        mae, max_abs = numpy_mae(torch_stage[key], tf_stage_map_actual[key])
                        print(f"    {key}: mae={mae:.6e} max={max_abs:.6e}")
                shared_actual_keys = sorted(set(tf_stage_map_shared.keys()) & set(tf_stage_map_actual.keys()))
                if shared_actual_keys:
                    print("  tf actual vs shared:")
                    for key in shared_actual_keys:
                        mae, max_abs = numpy_mae(tf_stage_map_shared[key], tf_stage_map_actual[key])
                        print(f"    {key}: mae={mae:.6e} max={max_abs:.6e}")
                input_ref = torch_input_prev_np
                input_tf = encoder_inputs_tf[layer_index]
                input_mae, input_max = numpy_mae(input_ref, input_tf)
                print(f"  input mae={input_mae:.6e} max={input_max:.6e}")
        else:
            print(f"\nencoder[{layer_index}] stage breakdown: TF stage data unavailable")

    if transformer_x is not None and transformer_xt is not None and transformer_tf_out is not None:
        tf_x, tf_xt = transformer_tf_out
        mae_x, max_x = numpy_mae(transformer_x, tf_x.numpy())
        mae_xt, max_xt = numpy_mae(transformer_xt, tf_xt.numpy())
        print("\ntransformer out:")
        print(f"  spec mae={mae_x:.6e} max={max_x:.6e}")
        print(f"  time mae={mae_xt:.6e} max={max_xt:.6e}")

        if hasattr(torch_model, "_spec"):
            spec = torch_model._spec(mix)
            print(f"\nPyTorch spec shape: {tuple(spec.shape)}")
            del spec

        freq_lengths_tf = [int(length.numpy()) for length in getattr(tf_model, "_debug_freq_lengths", [])]
        time_lengths_tf = [int(length.numpy()) for length in getattr(tf_model, "_debug_time_lengths", [])]
        spec_meta = getattr(tf_model, "_debug_spec_meta", None)
        if freq_lengths_tf or time_lengths_tf:
            print("tf freq lengths:", freq_lengths_tf)
            print("tf time lengths:", time_lengths_tf)
        if spec_meta is not None:
            length_tf, frames_tf, total_tf = (int(value.numpy()) for value in spec_meta)
            print(f"tf spec meta: length={length_tf} frames={frames_tf} total_frames={total_tf}")

    torch_spec = torch_spec_out.detach().cpu().numpy()
    tf_spec = tf_model._debug_mask.numpy() if isinstance(tf_model._debug_mask, tf.Tensor) else tf_model._debug_mask
    mae_spec, max_spec = numpy_mae(torch_spec, tf_spec)

    torch_time = torch_time_out.detach().cpu().numpy()
    tf_time = tf_model._debug_time.numpy() if isinstance(tf_model._debug_time, tf.Tensor) else tf_model._debug_time
    mae_time, max_time = numpy_mae(torch_time, tf_time)

    print("\nfinal tensors:")
    print(f"  mask mae={mae_spec:.6e} max={max_spec:.6e}")
    print(f"  time mae={mae_time:.6e} max={max_time:.6e}")

    with torch.no_grad():
        torch_wave = torch_model(mix).detach().cpu().numpy()
    tf_wave = tf_out.numpy()
    mae_wave, max_wave = numpy_mae(torch_wave, tf_wave)
    print("\nwaveform parity:")
    print(f"  waveform mae={mae_wave:.6e} max={max_wave:.6e}")


if __name__ == "__main__":
    main()
