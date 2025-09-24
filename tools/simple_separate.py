#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model, BagOfModels
from demucs.audio import AudioFile, convert_audio, save_audio


def separate_file(
    input_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    device: str = None,
    shifts: int = 0,
    split: bool = True,
    overlap: float = 0.25,
    segment: float | None = None,
    stem_ext: str = "wav",
):
    model = get_model(model_name)

    # BagOfModels is supported directly by apply_model
    sources = model.sources if not isinstance(model, BagOfModels) else model.sources
    samplerate = model.samplerate
    channels = model.audio_channels

    af = AudioFile(input_path)
    wav = af.read(streams=0, samplerate=samplerate, channels=channels)
    # shape [C, T] -> [1, C, T]
    mix = wav.unsqueeze(0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        out = apply_model(
            model,
            mix,
            shifts=shifts,
            split=split,
            overlap=overlap,
            progress=True,
            device=device,
            segment=segment,
        )

    # out: [1, S, C, T]
    out = out.squeeze(0).cpu()

    track_name = input_path.stem
    base_dir = output_dir / model_name / track_name
    base_dir.mkdir(parents=True, exist_ok=True)

    for k, name in enumerate(sources):
        stem_path = base_dir / f"{name}.{stem_ext}"
        save_audio(out[k], stem_path, samplerate)

    return base_dir


def main():
    parser = argparse.ArgumentParser(description="Simple Demucs-style separation script")
    parser.add_argument("input", type=Path, help="Input audio file (e.g., MP3)")
    parser.add_argument("--out", type=Path, default=Path("separated_simple"), help="Output directory root")
    parser.add_argument("-n", "--name", default="htdemucs", help="Model name/signature")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--shifts", type=int, default=0, help="Time shift trick iterations (0 disables)")
    parser.add_argument("--no-split", action="store_true", help="Disable chunked processing")
    parser.add_argument("--overlap", type=float, default=0.25, help="Chunk overlap ratio")
    parser.add_argument("--segment", type=float, default=None, help="Override segment seconds")
    parser.add_argument("--ext", default="wav", choices=["wav", "mp3", "flac"], help="Stem output format")

    args = parser.parse_args()
    split = not args.no_split

    out_dir = separate_file(
        input_path=args.input,
        output_dir=args.out,
        model_name=args.name,
        device=args.device,
        shifts=args.shifts,
        split=split,
        overlap=args.overlap,
        segment=args.segment,
        stem_ext=args.ext,
    )

    print(f"Saved stems to: {out_dir}")


if __name__ == "__main__":
    main()
