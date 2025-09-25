#!/usr/bin/env python3
"""
Inspect a web debug pack produced by the browser demo and compare against local ONNX.

Usage:
  python -m demucs.tools.inspect_debug_pack --pack /path/to/demucs_debug_pack.json --onnx /path/to/htdemucs_core.onnx

This will:
  - Parse the debug JSON (base64 float32 tensors)
  - Run the local ONNX model with the captured inputs (mag, mix)
  - Compare the browser's spec_out/time_out to local outputs and report statistics

Optional flags:
  --rt wasm|cpu (defaults to cpu); selects an EP for onnxruntime if available
  --rt-threads N (for CPU EP threading, if supported)
  --tol 1e-5 (absolute tolerance for quick close check)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    print("ERROR: onnxruntime is required. pip install onnxruntime", file=sys.stderr)
    raise

from demucs.spec import ispectro


def b64_to_f32(data_b64: str) -> np.ndarray:
    raw = base64.b64decode(data_b64)
    return np.frombuffer(raw, dtype=np.float32)


@dataclass
class Pack:
    meta: Dict[str, Any]
    shapes: Dict[str, Any]
    mag: np.ndarray
    mix: np.ndarray
    spec_out: np.ndarray
    time_out: np.ndarray


def load_pack(path: str) -> Pack:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    meta = obj.get("meta", {})
    shapes = obj.get("shapes", {})
    t = obj.get("tensors", {})

    mag = b64_to_f32(t["mag"]["data"]).reshape(shapes["mag"]).copy()
    mix = b64_to_f32(t["mix"]["data"]).reshape(shapes["mix"]).copy()
    spec_out = (
        b64_to_f32(t["spec_out"]["data"]).reshape(shapes["spec_out"]).copy()
    )
    time_out = (
        b64_to_f32(t["time_out"]["data"]).reshape(shapes["time_out"]).copy()
    )
    return Pack(meta=meta, shapes=shapes, mag=mag, mix=mix, spec_out=spec_out, time_out=time_out)


def run_local_onnx(onnx_path: str, mag: np.ndarray, mix: np.ndarray, rt: str = "cpu", rt_threads: int | None = None) -> Tuple[np.ndarray, np.ndarray, str]:
    providers = []
    if rt == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = None  # default

    so = ort.SessionOptions()
    if rt_threads:
        try:
            so.intra_op_num_threads = int(rt_threads)
        except Exception:
            pass
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    feeds = {}
    # Input names are stable in our export: 'mag' and 'mix'
    feeds["mag"] = mag.astype(np.float32)
    feeds["mix"] = mix.astype(np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, feeds)
    # Detect which is spec vs time by rank
    a, b = res
    if a.ndim == 5 and b.ndim == 4:
        spec, tim = a, b
    elif a.ndim == 4 and b.ndim == 5:
        spec, tim = b, a
    else:
        # fallback by matching C*2 dim
        if a.shape[2] == mag.shape[1]:
            spec, tim = a, b
        else:
            spec, tim = b, a
    ep = sess.get_providers()[0] if sess.get_providers() else "unknown"
    return spec, tim, ep


def _mask_cac_tensor(spec_out_f: torch.Tensor) -> torch.Tensor:
        """Convert float spec_out [B,S,C*2,F,T] into complex [B,S,C,F,T]."""
        B, S, C2, F, T = spec_out_f.shape
        assert C2 % 2 == 0, (spec_out_f.shape,)
        C = C2 // 2
        # [B,S,C*2,F,T] -> [B,S,C,2,F,T] -> [B,S,C,F,T,2]
        m = spec_out_f.view(B, S, C, 2, F, T).permute(0, 1, 2, 4, 5, 3).contiguous()
        return torch.view_as_complex(m)


def _ispec_like(z: torch.Tensor, length: int, nfft: int) -> torch.Tensor:
        """Mirror of the JS iSpecLike: pad (time frames 2,2), add Nyquist, iSTFT with nfft=2*F, crop center padding.

        Args:
            z: complex tensor [B,S,C,F,T]
            length: target output length (L)
            nfft: analysis nfft used to produce F (F = nfft//2)
        Returns: [B,S,C,L]
        """
        hl = (nfft // 4)
        B, S, C, F, T = z.shape
        z3 = z.view(B * S * C, F, T)
        # Add Nyquist bin back: input has F = nfft//2; synth needs F+1 for real iSTFT
        z3 = torch.nn.functional.pad(z3, (0, 0, 0, 1))  # pad freq dim: add 1 bin at end
        # Pad in time frames (center=True => 2 frames pad on both sides)
        z3 = torch.nn.functional.pad(z3, (2, 2))
        pad = int(hl * 3 // 2)
        le = int(hl * np.ceil(length / hl) + 2 * pad)
        x3 = ispectro(z3, int(hl), length=le)
        x3 = x3[..., pad : pad + length]
        return x3.view(B, S, C, length)


def stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    d = x.astype(np.float64) - y.astype(np.float64)
    mad = float(np.mean(np.abs(d)))
    rms = float(np.sqrt(np.mean(d * d)))
    mx = float(np.max(np.abs(d)))
    # relative to magnitude of reference
    denom = np.maximum(1e-12, np.mean(np.abs(y)))
    rel = float(mad / denom)
    return {"mean_abs": mad, "rms": rms, "max_abs": mx, "mean_abs_rel": rel}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, help="Path to demucs_debug_pack.json")
    p.add_argument("--onnx", required=True, help="Path to htdemucs_core.onnx")
    p.add_argument("--rt", default="cpu", choices=["cpu"], help="Execution provider preset")
    p.add_argument("--rt-threads", type=int, default=None, help="Intra-op threads for CPU EP")
    p.add_argument("--tol", type=float, default=1e-5, help="Absolute tolerance for quick close check")
    args = p.parse_args(argv)

    pack = load_pack(args.pack)
    spec_l, time_l, ep = run_local_onnx(args.onnx, pack.mag, pack.mix, args.rt, args.rt_threads)
    print(f"Local ONNX EP={ep}")

    # Shape checks
    print("Shapes:", {
        "mag": tuple(pack.mag.shape),
        "mix": tuple(pack.mix.shape),
        "spec_web": tuple(pack.spec_out.shape),
        "time_web": tuple(pack.time_out.shape),
        "spec_local": tuple(spec_l.shape),
        "time_local": tuple(time_l.shape),
    })

    s_spec = stats(spec_l, pack.spec_out)
    s_time = stats(time_l, pack.time_out)
    print("Diff spec_out (local - web):", s_spec)
    print("Diff time_out (local - web):", s_time)

    # Reconstruct waveforms from both web & local core outputs to see if mismatch is significant downstream.
    try:
        nfft = int(pack.meta.get("nfft", 4096))
        length = int(pack.mix.shape[-1])
        # Web tensors -> torch
        spec_web_t = torch.from_numpy(pack.spec_out)
        time_web_t = torch.from_numpy(pack.time_out)
        zout_web = _mask_cac_tensor(spec_web_t)  # [B,S,C,F,T] complex
        x_web = _ispec_like(zout_web, length, nfft)  # [B,S,C,L]
        y_web = (x_web + time_web_t).squeeze(0)  # [S,C,L]

        # Local tensors -> torch
        spec_loc_t = torch.from_numpy(spec_l)
        time_loc_t = torch.from_numpy(time_l)
        zout_loc = _mask_cac_tensor(spec_loc_t)
        x_loc = _ispec_like(zout_loc, length, nfft)
        y_loc = (x_loc + time_loc_t).squeeze(0)  # [S,C,L]

        # Waveform diff stats (per combined output)
        wy = y_loc.detach().cpu().numpy() - y_web.detach().cpu().numpy()
        wf_stats = {
            "mean_abs": float(np.mean(np.abs(wy))),
            "rms": float(np.sqrt(np.mean(wy * wy))),
            "max_abs": float(np.max(np.abs(wy))),
        }
        print("Waveform diff (local - web) after iSTFT+combine:", wf_stats)

        # Mixture consistency: sum of stems vs original mix (web and local)
        mix = torch.from_numpy(pack.mix).squeeze(0)  # [C,L]
        rec_web = y_web.sum(dim=0)  # [C,L]
        rec_loc = y_loc.sum(dim=0)
        err_web = (rec_web - mix).detach().cpu().numpy()
        err_loc = (rec_loc - mix).detach().cpu().numpy()
        mix_np = mix.detach().cpu().numpy()
        def rel_rms(e, ref):
            return float(np.sqrt(np.mean(e * e)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))
        print("Mixture consistency (web): rel_RMS(err vs mix) =", rel_rms(err_web, mix_np))
        print("Mixture consistency (local): rel_RMS(err vs mix) =", rel_rms(err_loc, mix_np))
    except Exception as e:  # pragma: no cover
        print("Note: waveform reconstruction diagnostics skipped due to error:", e)

    ok_spec = s_spec["max_abs"] <= args.tol
    ok_time = s_time["max_abs"] <= args.tol
    if ok_spec and ok_time:
        print("PASS: Local ONNX matches web outputs within tolerance")
        return 0
    else:
        print("WARN: Mismatch exceeds tolerance. Provider differences (WASM vs CPU) can cause small numeric drifts; use waveform diff and mixture-consistency metrics above to judge significance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
