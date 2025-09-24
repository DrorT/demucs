#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from typing import Optional

from demucs.audio import AudioFile, save_audio, convert_audio_channels
from demucs.spec import spectro, ispectro
from demucs.hdemucs import pad1d
from demucs.pretrained import SOURCES, get_model


def _spec_like(x: torch.Tensor, nfft: int) -> torch.Tensor:
    hl = nfft // 4
    length = x.shape[-1]
    # Pad reflect to match Demucs convention
    le = int(np.ceil(length / hl))
    pad = hl // 2 * 3
    x = pad1d(x, (pad, pad + le * hl - length), mode="reflect")
    z = spectro(x, nfft, hl)[..., :-1, :]
    assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
    z = z[..., 2 : 2 + le]
    return z


def _ispec_like(z: torch.Tensor, length: int, nfft: int) -> torch.Tensor:
    hl = (nfft // 4) // (4 ** 0)
    z = torch.nn.functional.pad(z, (0, 0, 0, 1))
    z = torch.nn.functional.pad(z, (2, 2))
    pad = int(hl // 2 * 3)
    le = int(hl * np.ceil(length / hl) + 2 * pad)
    x = ispectro(z, int(hl), length=le)
    x = x[..., pad : pad + length]
    return x


def _magnitude_cac(z: torch.Tensor, cac: bool) -> torch.Tensor:
    if cac:
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
    else:
        m = z.abs()
    return m


def _mask_cac(spec_out: np.ndarray, cac: bool) -> torch.Tensor:
    # Convert ONNX spec_out to complex spectrogram per Demucs convention (CAC only)
    m = torch.from_numpy(spec_out)
    if cac:
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out
    else:
        # If not CAC, this would be a magnitude mask; Skipping Wiener here, just ratio-mask.
        raise NotImplementedError("Non-CAC masking is not implemented in this demo")


def separate_with_onnx(
    input_path: Path,
    onnx_path: Path,
    output_dir: Path,
    samplerate: int,
    channels: int,
    nfft: int,
    cac: bool,
    segment: float,
    overlap: float = 0.25,
    stem_ext: str = "wav",
    device: str = "auto",
    profile: bool = False,
    json_metrics: Optional[Path] = None,
    ort_num_threads: int | None = None,
):
    t_start = time.perf_counter()
    # Prepare ONNX session with provider/device selection
    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider not available. Install onnxruntime-gpu or choose --device cpu/auto.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:  # auto
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    if ort_num_threads is not None and ort_num_threads > 0:
        # Set both intra/inter; users can tune further if desired
        so.intra_op_num_threads = ort_num_threads
        so.inter_op_num_threads = max(1, ort_num_threads // 2)

    t_sess0 = time.perf_counter()
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    t_sess1 = time.perf_counter()
    used_provider = sess.get_providers()[0] if hasattr(sess, "get_providers") else providers[0]

    # Read audio
    t_io0 = time.perf_counter()
    af = AudioFile(input_path)
    # Read single audio stream, convert channels at load (no resample in AudioFile)
    wav = af.read(streams=0, samplerate=None, channels=None)
    wav = convert_audio_channels(wav, channels)
    if af.samplerate() != samplerate:
        # Resample via torch (julius) using the convert_audio helper pattern
        import julius

        wav = julius.resample_frac(wav, af.samplerate(), samplerate)
    t_io1 = time.perf_counter()

    # [C,T] -> [1,C,T]
    mix_full = wav.unsqueeze(0)
    B, C, L = mix_full.shape

    # Chunking
    seg_len = int(segment * samplerate)
    stride = int((1 - overlap) * seg_len)
    if stride <= 0:
        stride = seg_len
    offsets = list(range(0, L, stride))
    # triangular weights
    weight = torch.cat(
        [torch.arange(1, seg_len // 2 + 1), torch.arange(seg_len - seg_len // 2, 0, -1)]
    ).float()
    weight = weight / weight.max()

    S = len(SOURCES)
    out_total = torch.zeros(B, S, C, L)
    sum_weight = torch.zeros(L)

    # Timers
    t_spec_total = 0.0
    t_infer_total = 0.0
    t_post_total = 0.0
    chunks = 0

    for off in offsets:
        chunk = mix_full[..., off : off + seg_len]
        # Pad to seg_len
        if chunk.shape[-1] < seg_len:
            chunk = torch.nn.functional.pad(chunk, (0, seg_len - chunk.shape[-1]))

        # Pre: STFT and magnitude
        t0 = time.perf_counter()
        z = _spec_like(chunk, nfft)
        mag = _magnitude_cac(z, cac)
        t1 = time.perf_counter()

        # Run ONNX core
        t2 = time.perf_counter()
        inputs = {
            "mag": mag.cpu().numpy(),
            "mix": chunk.cpu().numpy(),
        }
        spec_out, time_out = sess.run(None, inputs)
        t3 = time.perf_counter()

        # Post: mask/iSTFT + sum with time branch
        t4 = time.perf_counter()
        zout = _mask_cac(spec_out, cac)
        x = _ispec_like(zout, chunk.shape[-1], nfft)
        xt = torch.from_numpy(time_out)
        y = (x + xt).squeeze(0)  # [S,C,T]

        out_len = min(seg_len, L - off)
        out_total[..., off : off + out_len] += (weight[:out_len] * y[..., :out_len]).unsqueeze(0)
        sum_weight[off : off + out_len] += weight[:out_len]

        t5 = time.perf_counter()
        t_spec_total += (t1 - t0)
        t_infer_total += (t3 - t2)
        t_post_total += (t5 - t4)
        chunks += 1

    out_final = out_total / sum_weight.clamp_min(1e-6)
    out_final = out_final.squeeze(0)  # [S,C,T]

    # Save stems
    t_save0 = time.perf_counter()
    track_name = input_path.stem
    base_dir = output_dir / track_name
    base_dir.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(SOURCES):
        save_audio(out_final[k], base_dir / f"{name}.{stem_ext}", samplerate)
    t_save1 = time.perf_counter()

    t_end = time.perf_counter()

    # Metrics
    audio_seconds = L / float(samplerate)
    total_time = t_end - t_start
    processing_time = (t_sess1 - t_sess0) + (t_io1 - t_io0) + t_spec_total + t_infer_total + t_post_total + (t_save1 - t_save0)
    rtf = total_time / max(1e-9, audio_seconds)
    metrics = {
        "file": str(input_path),
        "samplerate": samplerate,
        "channels": channels,
        "nfft": nfft,
        "cac": bool(cac),
        "segment_seconds": float(segment),
        "overlap": float(overlap),
        "audio_seconds": audio_seconds,
        "chunks": chunks,
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "provider": used_provider,
        "providers_requested": providers,
        "ort_num_threads": ort_num_threads,
        "time_total_sec": total_time,
        "time_session_create_sec": (t_sess1 - t_sess0),
        "time_io_sec": (t_io1 - t_io0),
        "time_spec_total_sec": t_spec_total,
        "time_infer_total_sec": t_infer_total,
        "time_post_total_sec": t_post_total,
        "time_save_sec": (t_save1 - t_save0),
        "rtf": rtf,
        "avg_infer_ms_per_chunk": (t_infer_total / max(1, chunks)) * 1000.0,
    }

    # Always print concise summary
    print(
        f"Provider: {used_provider} | audio: {audio_seconds:.2f}s | total: {total_time:.3f}s | RTF: {rtf:.3f} | "
        f"chunks: {chunks} | infer_total: {t_infer_total:.3f}s (~{metrics['avg_infer_ms_per_chunk']:.1f} ms/chunk)"
    )

    if profile:
        print(
            "Profile: session={:.3f}s, io={:.3f}s, spec={:.3f}s, infer={:.3f}s, post={:.3f}s, save={:.3f}s".format(
                metrics["time_session_create_sec"],
                metrics["time_io_sec"],
                metrics["time_spec_total_sec"],
                metrics["time_infer_total_sec"],
                metrics["time_post_total_sec"],
                metrics["time_save_sec"],
            )
        )

    if json_metrics:
        try:
            Path(json_metrics).parent.mkdir(parents=True, exist_ok=True)
            with open(json_metrics, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write metrics JSON to {json_metrics}: {e}")
    return base_dir


def main():
    p = argparse.ArgumentParser(description="Separate using ONNX core model")
    p.add_argument("input", type=Path, help="Input audio file (e.g., MP3)")
    p.add_argument("--onnx", type=Path, default=Path("htdemucs_core.onnx"))
    p.add_argument("--out", type=Path, default=Path("separated_onnx"))
    p.add_argument("-n", "--name", default="htdemucs", help="Torch model name to read meta (sr, seg, nfft, cac)")
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--nfft", type=int, default=None)
    p.add_argument("--segment", type=float, default=None)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--ext", default="wav", choices=["wav", "mp3", "flac"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Select ORT device/provider")
    p.add_argument("--ort-num-threads", type=int, default=None, help="Limit ONNX Runtime CPU threads")
    p.add_argument("--profile", action="store_true", help="Print detailed timing breakdown")
    p.add_argument("--json-metrics", type=Path, default=None, help="Write metrics JSON to this path")
    args = p.parse_args()

    # Load torch model only to read metadata (segment, sr, channels, cac, nfft)
    torch_model = get_model(args.name)
    # Unwrap bags for metadata
    try:
        from demucs.apply import BagOfModels
        if isinstance(torch_model, BagOfModels):
            torch_model = torch_model.models[0]
    except Exception:
        pass
    samplerate = args.sr or torch_model.samplerate
    channels = args.channels or torch_model.audio_channels
    nfft = args.nfft or getattr(torch_model, 'nfft', 4096)
    cac = getattr(torch_model, 'cac', True)
    segment = args.segment if args.segment is not None else float(torch_model.segment)

    out_dir = separate_with_onnx(
        input_path=args.input,
        onnx_path=args.onnx,
        output_dir=args.out,
        samplerate=samplerate,
        channels=channels,
        nfft=nfft,
        cac=cac,
        segment=segment,
        overlap=args.overlap,
        stem_ext=args.ext,
        device=args.device,
        profile=args.profile,
        json_metrics=args.json_metrics,
        ort_num_threads=args.ort_num_threads,
    )
    print(f"Saved stems to: {out_dir}")


if __name__ == "__main__":
    main()
