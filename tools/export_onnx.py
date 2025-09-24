#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from demucs.pretrained import get_model
from demucs.apply import BagOfModels


def load_htdemucs(name: str):
    model = get_model(name)
    # Unwrap bag if needed, use first submodel
    if isinstance(model, BagOfModels):
        model = model.models[0]
    assert hasattr(model, 'cac'), 'Expected HTDemucs-like model'
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Export HTDemucs core to ONNX')
    parser.add_argument('-n', '--name', default='htdemucs', help='Model name or signature')
    parser.add_argument('-o', '--output', type=Path, default=Path('htdemucs_core.onnx'))
    parser.add_argument('--segment', type=float, default=None, help='Override segment length (s)')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes for time frames')
    args = parser.parse_args()

    model = load_htdemucs(args.name)

    # Create representative dummy inputs following the internal pipeline
    B = 1
    S = len(model.sources)
    C = model.audio_channels
    if args.segment is None:
        length = int(model.segment * model.samplerate) if model.use_train_segment else int(10 * model.samplerate)
    else:
        length = int(args.segment * model.samplerate)

    mix = torch.randn(B, C, length)
    with torch.no_grad():
        z = model._spec(mix)
        mag = model._magnitude(z)

    class CoreWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, mag_in, mix_in):
            spec_out, time_out = self.core.forward_core(mag_in, mix_in)
            return spec_out, time_out

    wrapped = CoreWrapper(model).cpu()
    mag = mag.cpu()
    mix = mix.cpu()

    input_names = ['mag', 'mix']
    output_names = ['spec_out', 'time_out']

    dynamic_axes = None
    if args.dynamic:
        # allow dynamic time dimension T and waveform length L
        dynamic_axes = {
            'mag': {0: 'batch', 2: 'freq', 3: 'frames'},
            'mix': {0: 'batch', 2: 'length'},
            'spec_out': {0: 'batch', 3: 'freq', 4: 'frames'},
            'time_out': {0: 'batch', 3: 'length'},
        }

    torch.onnx.export(
        wrapped,
        (mag, mix),
        f=args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f'Exported ONNX core to {args.output}')


if __name__ == '__main__':
    main()
