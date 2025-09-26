"""Command line interface for running DemucsTF separation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch

from demucs_tf.models import DemucsTF

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------


def _maybe_resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError(
            "Sample rate mismatch. Install 'librosa' to enable resampling, "
            f"or provide audio at {target_sr} Hz. (current: {sr} Hz)"
        ) from exc
    audio_resampled = []
    for channel in audio:
        audio_resampled.append(
            librosa.resample(  # type: ignore[attr-defined]
                channel,
                orig_sr=sr,
                target_sr=target_sr,
            )
        )
    return np.stack(audio_resampled).astype(np.float32)


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.T  # (channels, time)
    audio = _maybe_resample(audio, sr, target_sr)
    return audio.astype(np.float32)


# -----------------------------------------------------------------------------
# Checkpoint/config helpers
# -----------------------------------------------------------------------------


_EXPECTED_KEYS: Sequence[str] = (
    "audio_channels",
    "channels",
    "growth",
    "depth",
    "rewrite",
    "lstm_layers",
    "kernel_size",
    "stride",
    "context",
    "gelu",
    "glu",
    "norm_starts",
    "norm_groups",
    "dconv_mode",
    "dconv_depth",
    "dconv_comp",
    "dconv_attn",
    "dconv_lstm",
    "dconv_init",
    "normalize",
    "resample",
    "samplerate",
    "segment",
)


def _load_json(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_checkpoint_config(state: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(state.get("config"), dict):
        candidates.append(state["config"])
    metadata = state.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("config"), dict):
        candidates.append(metadata["config"])
    args = state.get("args")
    if isinstance(args, dict):
        candidates.append(args)
    elif hasattr(args, "__dict__"):
        candidates.append(vars(args))

    merged: Dict[str, Any] = {}
    for candidate in candidates:
        merged.update(candidate)
    return merged


def build_model_from_checkpoint(
    checkpoint_path: Path,
    config_override: Path | None,
) -> DemucsTF:
    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    config = _extract_checkpoint_config(checkpoint)
    overrides = _load_json(config_override)
    config.update(overrides)

    if "sources" not in config:
        LOGGER.warning(
            "Checkpoint lacks `sources`; falling back to default Demucs stems"
        )
        config["sources"] = ["drums", "bass", "other", "vocals"]

    kwargs = {key: config[key] for key in _EXPECTED_KEYS if key in config}
    sources = config["sources"]

    model = DemucsTF(sources=sources, **kwargs)
    report = model.load_pytorch_checkpoint(checkpoint_path)
    if report.missing:
        LOGGER.warning("Missing weights: \n%s", "\n".join(sorted(report.missing)))
    if report.unused:
        LOGGER.warning("Unused weights: \n%s", "\n".join(sorted(report.unused)))
    return model


# -----------------------------------------------------------------------------
# Separation
# -----------------------------------------------------------------------------


def separate(model: DemucsTF, mix: np.ndarray) -> np.ndarray:
    mix_tensor = tf.convert_to_tensor(mix[None, ...])
    with tf.device("/CPU:0"):
        estimates = model(mix_tensor, training=False)
    return estimates.numpy()[0]


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DemucsTF separation")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the PyTorch Demucs checkpoint (.th)",
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more audio files to separate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with keys to override the checkpoint config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("separated_tf"),
        help="Directory where separated stems will be written",
    )
    parser.add_argument(
        "--stem-format",
        type=str,
        default="{stem}",
        help="Filename format for stems (default: {stem})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stem files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model = build_model_from_checkpoint(args.checkpoint, args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    for audio_path in args.inputs:
        if not audio_path.exists():
            LOGGER.error("Input audio %s does not exist", audio_path)
            continue
        LOGGER.info("Separating %s", audio_path)
        mix = load_audio(audio_path, model.samplerate)
        stems = separate(model, mix)
        for source, audio in zip(model.sources, stems):
            stem_name = args.stem_format.format(stem=source, mix=audio_path.stem)
            out_path = args.output / f"{audio_path.stem}_{stem_name}.wav"
            if out_path.exists() and not args.overwrite:
                LOGGER.warning("Skipping existing file %s (use --overwrite)", out_path)
                continue
            sf.write(out_path, audio.T, model.samplerate)
            LOGGER.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
