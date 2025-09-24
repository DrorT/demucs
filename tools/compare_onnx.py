#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import torchaudio as ta

from demucs.pretrained import get_model
from demucs.apply import BagOfModels


def run_pytorch_full(model, wav):
    with torch.no_grad():
        out = model(wav)
    return out.cpu().numpy()


def run_core_pre_post(model, wav, spec_out, time_out, orig_length=None):
    with torch.no_grad():
        # Recreate STFT of the (possibly padded) input
        z = model._spec(wav)
        # Apply mask using spec_out and z
        zout = model._mask(z, torch.from_numpy(spec_out).to(z.device))
        # iSTFT to waveform (use full padded length)
        x = model._ispec(zout, wav.shape[-1])
        # Combine with time branch
        xt = torch.from_numpy(time_out).to(x.device)
        y = (xt + x)
        if orig_length is not None:
            y = y[..., :orig_length]
        return y.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch HTDemucs vs ONNX core parity')
    parser.add_argument('-n', '--name', default='htdemucs')
    parser.add_argument('-m', '--model', type=Path, default=Path('htdemucs_core.onnx'))
    parser.add_argument('-i', '--input', type=Path, default=Path('test.mp3'))
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--segment', type=float, default=None)
    parser.add_argument('--providers', nargs='*', default=None, help='ONNX providers list')
    args = parser.parse_args()

    model = get_model(args.name)
    if isinstance(model, BagOfModels):
        model = model.models[0]
    model = model.eval()

    wav, sr = ta.load(str(args.input))
    if sr != args.sr:
        wav = ta.functional.resample(wav, sr, args.sr)
    wav = wav[: model.audio_channels]
    wav = wav.unsqueeze(0)

    # Determine inference length
    orig_length = wav.shape[-1]
    if args.segment is not None:
        infer_length = int(args.segment * args.sr)
    elif model.use_train_segment and not model.training:
        infer_length = int(model.segment * model.samplerate)
    else:
        infer_length = orig_length

    if wav.shape[-1] < infer_length:
        wav_proc = torch.nn.functional.pad(wav, (0, infer_length - wav.shape[-1]))
    else:
        wav_proc = wav[..., :infer_length]

    # Prepare ONNX session
    sess_options = ort.SessionOptions()
    providers = args.providers or ort.get_available_providers()
    print('Using ONNX providers:', providers)
    sess = ort.InferenceSession(str(args.model), sess_options, providers=providers)

    # Build core inputs
    with torch.no_grad():
        z = model._spec(wav_proc)
        mag = model._magnitude(z)
    inputs = {
        'mag': mag.cpu().numpy(),
        'mix': wav_proc.cpu().numpy(),
    }

    # Run ONNX core
    spec_out, time_out = sess.run(None, inputs)

    # Post-process via PyTorch helpers
    out_core = run_core_pre_post(model, wav_proc, spec_out, time_out, orig_length=orig_length)

    # Full PyTorch forward
    out_full = run_pytorch_full(model, wav_proc)

    # Compare
    diff = out_core - out_full
    l2 = np.sqrt((diff ** 2).mean())
    mae = np.abs(diff).mean()
    mx = np.abs(diff).max()
    print(f'L2: {l2:.6f}  MAE: {mae:.6f}  MaxAbs: {mx:.6f}')


if __name__ == '__main__':
    main()
