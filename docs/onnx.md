# HTDemucs → ONNX (Browser)

This repo adds an export path for the HTDemucs v4 hybrid model to ONNX and a parity harness to compare PyTorch vs ONNX (core) outputs. The intended runtime target is `onnxruntime-web` in the browser.

Overview

- We export the network "core" only: everything between magnitude spectrogram input and the decoder outputs from both branches. STFT/iSTFT and Wiener/filtering remain outside the ONNX graph and are implemented in Python for parity and in JS for production.
- This preserves key features: hybrid time–freq architecture, CAC (complex-as-channels) path, cross-transformer, and skip connections. Wiener iterations remain doable on CPU if desired pre-iSTFT.

What is exported

- Inputs:
  - `mag`: `(B, C, F, T)` mixture representation. If `cac=True` (default), this is real/imag stacked as channels, created by `_magnitude(_spec(wav))`.
  - `mix`: `(B, audio_channels, L)` time waveform for the parallel time branch normalization path.
- Outputs:
  - `spec_out`: `(B, S, C_spec, F, T)` spectrogram branch result prior to masking/iSTFT.
  - `time_out`: `(B, S, C_time, L)` time branch result prior to final sum.

Export

- Script: `tools/export_onnx.py`
- Example:

```bash
python -m tools.export_onnx -n htdemucs -o htdemucs_core.onnx --opset 17 --dynamic
```

Parity test

- Script: `tools/compare_onnx.py`
- Compares full PyTorch forward vs ONNX core + Python pre/post.

```bash
pip install onnxruntime torchaudio
python -m tools.compare_onnx -n htdemucs -m htdemucs_core.onnx -i test.mp3 --sr 44100
```

Browser integration

- Use `onnxruntime-web` with WebAssembly + SIMD (and optionally WebGPU as it matures).
- Pre/post-processing in JS:
  - STFT: Hann window, `nfft=4096`, `hop=nfft/4`, pad policy matching `_spec` (reflect, +2 frame crop).
  - Magnitude for CAC: arrange real/imag as channels like `_magnitude`.
  - Masking: if `cac=True`, core `spec_out` is a complex spectrogram already; otherwise Wiener or ratio masking can be applied.
  - iSTFT: mirror `_ispec` conventions (pad, frame alignment), then sum with `time_out`.

Notes

- For exact parity, ensure the same normalization (mean/std per batch-item) as in `forward_core` and the same padding strategy as `_spec/_ispec`.
- Use `--dynamic` to export dynamic time dimensions. Fixed-size export may be simpler for initial integration.

# HTDemucs → ONNX (Browser)

This repo adds an export path for the HTDemucs v4 hybrid model to ONNX and a parity harness to compare PyTorch vs ONNX (core) outputs. The intended runtime target is `onnxruntime-web` in the browser.

Overview

- We export the network "core" only: everything between magnitude spectrogram input and the decoder outputs from both branches. STFT/iSTFT and Wiener/filtering remain outside the ONNX graph and are implemented in Python for parity and in JS for production.
- This preserves key features: hybrid time–freq architecture, CAC (complex-as-channels) path, cross-transformer, and skip connections. Wiener iterations remain doable on CPU if desired pre-iSTFT.

What is exported

- Inputs:
  - `mag`: `(B, C, F, T)` mixture representation. If `cac=True` (default), this is real/imag stacked as channels, created by `_magnitude(_spec(wav))`.
  - `mix`: `(B, audio_channels, L)` time waveform for the parallel time branch normalization path.
- Outputs:
  - `spec_out`: `(B, S, C_spec, F, T)` spectrogram branch result prior to masking/iSTFT.
  - `time_out`: `(B, S, C_time, L)` time branch result prior to final sum.

Export

- Script: `tools/export_onnx.py`
- Example:

```bash
python -m demucs.tools.export_onnx -n htdemucs -o htdemucs_core.onnx --opset 17 --dynamic
```

Parity test

- Script: `tools/compare_onnx.py`
- Compares full PyTorch forward vs ONNX core + Python pre/post.

```bash
pip install onnxruntime torchaudio
python -m demucs.tools.compare_onnx -n htdemucs -m htdemucs_core.onnx -i test.mp3 --sr 44100
```

Browser integration

- Use `onnxruntime-web` with WebAssembly + SIMD (and optionally WebGPU as it matures).
- Pre/post-processing in JS:
  - STFT: Hann window, `nfft=4096`, `hop=nfft/4`, pad policy matching `_spec` (reflect, +2 frame crop).
  - Magnitude for CAC: arrange real/imag as channels like `_magnitude`.
  - Masking: if `cac=True`, core `spec_out` is a complex spectrogram already; otherwise Wiener or ratio masking can be applied.
  - iSTFT: mirror `_ispec` conventions (pad, frame alignment), then sum with `time_out`.

Notes

- For exact parity, ensure the same normalization (mean/std per batch-item) as in `forward_core` and the same padding strategy as `_spec/_ispec`.
- Use `--dynamic` to export dynamic time dimensions. Fixed-size export may be simpler for initial integration.
