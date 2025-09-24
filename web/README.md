# Demucs Web (ONNX)

A minimal web demo to run the Demucs ONNX core in the browser using onnxruntime-web.

## Quick start

```bash
cd web
npm install
npm run dev
```

Open the printed URL, load an audio file, then click Separate.

## Files

- `public/models/htdemucs_core.onnx`: Place your exported ONNX core here.
- `public/models/meta.json`: Model metadata (samplerate, channels, nfft, cac, segment).
- `src/dsp.ts`: STFT/iSTFT and CAC helpers mirroring Python path.
- `src/index.ts`: Onnx runtime wrapper running pre/post and core.
- `public/index.html` + `src/demo.ts`: Simple UI and wiring.

## Notes

- Execution provider defaults to WASM with optional WebGL when available.
- Browser performance varies; larger segments increase latency but reduce overhead.

## Troubleshooting WASM

If you see an exception thrown from `ort-wasm-*.mjs` like a `throw ad;` inside a function `eb(...)`, try:

- Force single-threaded WASM and enable verbose logs in the `load()` call:

```ts
await demucs.load("/models/htdemucs_core.onnx", "/models/meta.json", {
  device: "wasm",
  wasm: { numThreads: 1, simd: true, proxy: "worker", logLevel: "verbose" },
});
```

- Ensure ORT assets are served under `/ort-wasm/` and COOP/COEP headers are set (see `vite.config.ts`).
- If WebGPU/WebGL fail, use `device: "wasm"`.
