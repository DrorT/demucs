# Demucs PyTorch → TensorFlow Migration Plan

## Goals
- Rebuild the Demucs separator in pure TensorFlow/Keras while matching the PyTorch reference implementation bit-for-bit.
- Load weights from existing PyTorch checkpoints and assign them to the TensorFlow model.
- Validate output parity on representative audio mixtures and package the TensorFlow graph for downstream consumers (SavedModel, TFLite, ONNX).

## Proposed TensorFlow Package Layout
```
demucs_tf/
  __init__.py
  layers/
    __init__.py
    conv.py           # Conv1D + ConvTranspose1D wrappers, padding helpers
    norm.py           # GroupNorm (TFA or custom), LayerScale
    recurrent.py      # BLSTM with chunking support
    attention.py      # LocalState attention block
  blocks/
    __init__.py
    dconv.py          # Residual DConv stack with optional LSTM/attention
    encoder.py        # Encoder module definitions
    decoder.py        # Decoder module definitions
  utils/
    __init__.py
    audio.py          # center_trim, unfold emulation, resample helpers
    config.py         # Dataclasses for architecture hyperparameters
  models/
    __init__.py
    demucs.py         # Core DemucsTF model class
    hdemucs.py        # Optional hybrid variant
  checkpoints/
    __init__.py
    loader.py         # PyTorch → TF weight loading utilities
  cli/
    __init__.py
    separate.py       # Entry-point for CLI inference (wraps SavedModel run)
```

## Migration Workstream
1. **Scaffold package**: add the `demucs_tf/` tree with placeholder modules and minimal imports.
2. **Implement layers**: translate Conv/ConvTranspose wrappers, GroupNorm, LayerScale, and shared utilities.
3. **Port residual blocks**: replicate `DConv`, `LocalState`, and BLSTM chunking behaviour.
4. **Build model shell**: wire encoders/decoders using the new layers, respecting stride/padding semantics and optional resampling.
5. **Weight transfer**: implement loader using `torch.load` + NumPy conversions; map PyTorch state_dict keys to TF variables.
6. **Parity validation**: write script to compare PyTorch & TF outputs (relative L2, SI-SDR delta) on curated audio snippets.
7. **Export pathways**: author scripts for SavedModel export, optional TFLite conversion, and TF→ONNX conversion.
8. **Automation**: integrate validation into CI and document usage in README/docs.

## Environment & Tooling
- Use the project virtualenv: `source pydemucs/bin/activate`.
- TensorFlow version target: 2.15+ (confirm compatibility with tf.lite & tf2onnx).
- TensorFlow Addons optional; include fallback implementations where needed.

## Open Questions
- Do we support both classic Demucs and Hybrid (spectrogram) variants in phase 1?
- How closely must we mirror PyTorch numerical parity (exact float match vs tolerances)?
- Should resampling reuse Julius polyphase filters or leverage TF signal processing APIs?

## Milestones
- **M1 – Skeleton Ready**: Package layout committed, placeholder modules pass import checks.
- **M2 – Functional Model**: TF Demucs runs inference with random weights.
- **M3 – Weight Parity**: PyTorch checkpoint successfully transferred; parity tests below tolerance.
- **M4 – Export Suite**: SavedModel & TFLite/ONNX artifacts reproducible with documentation.
