declare global {
  interface Window {
    ort: any;
  }
}
const ort: any = (window as any).ort;

import { iSpecLike, specLike, packCAC, unpackCAC } from "./dsp";

export type ModelMeta = {
  samplerate: number;
  channels: number;
  nfft: number;
  cac: boolean;
  segment: number;
};

export class DemucsOnnx {
  private session!: any;
  private meta!: ModelMeta;
  private _modelUrl!: string;
  private _metaUrl!: string;
  private _opts: any;

  async load(
    modelUrl: string,
    metaUrl: string,
    opts?: {
      device?: "auto" | "cpu" | "webgl" | "wasm" | "webgpu";
      graphOptimizationLevel?: any;
      wasm?: {
        numThreads?: number; // 0=auto
        simd?: boolean;
        proxy?: "none" | "worker" | "wasm";
        initTimeout?: number;
        logLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
      };
    }
  ) {
    this._modelUrl = modelUrl;
    this._metaUrl = metaUrl;
    this._opts = opts ?? {};
    const res = await fetch(metaUrl);
    this.meta = await res.json();
    // Configure ORT WASM environment before creating the session
    try {
      const threads = opts?.wasm?.numThreads ?? 1;
      ort.env.wasm.wasmPaths = "/ort-wasm/";
      ort.env.wasm.numThreads = threads;
      ort.env.wasm.simd = opts?.wasm?.simd ?? true;
      ort.env.wasm.proxy =
        threads === 1 ? "none" : opts?.wasm?.proxy ?? "worker";
      ort.env.logLevel = opts?.wasm?.logLevel ?? "warning";
    } catch {}
    // Build provider list
    const providerList: any[] = [];
    const dev = opts?.device ?? "auto";
    if (dev === "webgpu" || dev === "auto")
      providerList.push({ name: "webgpu" });
    if (dev === "webgl" || dev === "auto") providerList.push({ name: "webgl" });
    if (dev === "wasm" || dev === "cpu" || dev === "auto") {
      const wasmOps: any = { name: "wasm" };
      if (opts?.wasm) {
        wasmOps.wasm = {
          numThreads: opts.wasm.numThreads ?? 0,
          simd: opts.wasm.simd ?? true,
          proxy: opts.wasm.proxy ?? "worker",
          initTimeout: opts.wasm.initTimeout ?? 10000,
        };
      }
      providerList.push(wasmOps);
    }
    const sessOpts: any = {
      executionProviders: (providerList.length
        ? providerList
        : undefined) as any,
      graphOptimizationLevel: opts?.graphOptimizationLevel ?? "all",
      // Point to Vite public path with ORT assets
      wasmPaths: "/ort-wasm/",
      logLevel: opts?.wasm?.logLevel ?? "warning",
    };
    this.session = await ort.InferenceSession.create(modelUrl, sessOpts);
  }

  getMeta(): ModelMeta {
    return this.meta;
  }

  // Run once and produce three sets of stems: sum, spec-only, time-only
  async separateAll(
    pcm: Float32Array,
    opts?: {
      maxSeconds?: number;
      onProgress?: (p: number) => void;
      segmentSeconds?: number;
      logPerf?: boolean;
    }
  ): Promise<{
    results: Record<"sum" | "spec" | "time", Record<string, Float32Array>>;
    provider: string;
  }> {
    const { samplerate, channels, nfft, segment } = this.meta;
    const C = channels;
    const T = Math.floor(pcm.length / C);
    // deinterleave
    const ch: Float32Array[] = new Array(C);
    for (let c = 0; c < C; c++) ch[c] = new Float32Array(T);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++) ch[c][t] = pcm[t * C + c];

    const segSec =
      opts?.segmentSeconds && opts.segmentSeconds > 0
        ? opts.segmentSeconds
        : segment;
    const segLen = Math.floor(segSec * samplerate);
    const stride = Math.max(1, Math.floor(0.75 * segLen));
    const offsets: number[] = [];
    for (let off = 0; off < T; off += stride) offsets.push(off);

    const S = 4; // drums,bass,other,vocals
    const outSum: Float32Array[][] = new Array(S);
    const outSpec: Float32Array[][] = new Array(S);
    const outTime: Float32Array[][] = new Array(S);
    for (let s = 0; s < S; s++) {
      outSum[s] = new Array(C).fill(0).map(() => new Float32Array(T));
      outSpec[s] = new Array(C).fill(0).map(() => new Float32Array(T));
      outTime[s] = new Array(C).fill(0).map(() => new Float32Array(T));
    }

    // triangular weights
    const weight = new Float32Array(segLen);
    for (let i = 0; i < Math.floor(segLen / 2) + 1; i++) weight[i] = i + 1;
    for (let i = Math.floor(segLen / 2) + 1; i < segLen; i++)
      weight[i] = segLen - i;
    let wmax = 0;
    for (let i = 0; i < segLen; i++) wmax = Math.max(wmax, weight[i]);
    for (let i = 0; i < segLen; i++) weight[i] /= wmax;

    const maxT = opts?.maxSeconds
      ? Math.min(T, Math.floor(opts.maxSeconds * samplerate))
      : T;
    const totalChunks = Math.ceil(maxT / stride);
    const perfEnabled = !!opts?.logPerf;
    let preSum = 0,
      inferSum = 0,
      postSum = 0,
      chunkCount = 0;
    const overallStart = performance.now();
    let chunkIdx = 0;

    let F = 0,
      TT = 0;

    for (const off of offsets) {
      if (off >= maxT) break;
      const chunkStart = performance.now();
      if (perfEnabled)
        console.log(
          `[demucs] chunk ${chunkIdx + 1}/${totalChunks} start: off=${off} (${(
            off / samplerate
          ).toFixed(2)}s)`
        );

      // Prepare chunk
      const chunk: Float32Array[] = new Array(C);
      for (let c = 0; c < C; c++) {
        chunk[c] = new Float32Array(segLen);
        const len = Math.min(segLen, T - off);
        chunk[c].set(ch[c].subarray(off, off + len));
      }

      // Pre-processing: STFT + CAC pack
      const tPre0 = performance.now();
      const specMag: Float32Array[] = [];
      for (let c = 0; c < C; c++) {
        const { real, imag } = specLike(chunk[c], nfft);
        F = real[0].length;
        TT = real.length;
        specMag.push(packCAC(real, imag));
      }
      const tPre1 = performance.now();

      // Prepare feeds
      const mag = new Float32Array(C * 2 * F * TT);
      for (let c = 0; c < C; c++) mag.set(specMag[c], c * 2 * F * TT);
      const mixBuf = new Float32Array(C * segLen);
      for (let c = 0; c < C; c++) mixBuf.set(chunk[c], c * segLen);
      const feeds: Record<string, any> = {
        mag: new ort.Tensor("float32", mag, [1, C * 2, F, TT]),
        mix: new ort.Tensor("float32", mixBuf, [1, C, segLen]),
      };

      // Inference
      let res: any;
      const tInfer0 = performance.now();
      try {
        res = await this.session.run(feeds);
      } catch (e) {
        console.error("ORT run failed. Shapes:", {
          mag: [1, C * 2, F, TT],
          mix: [1, C, segLen],
          providerList: (this.session as any).executionProviders,
          wasmEnv: (ort as any).env?.wasm,
        });
        const ep = (this.session as any).executionProviders?.[0]?.name;
        const threads = (ort as any).env?.wasm?.numThreads;
        if (ep === "wasm" && threads && threads > 1) {
          console.warn("Retrying with single-threaded WASM after failure...");
          (ort as any).env.wasm.numThreads = 1;
          const opts2 = Object.assign({}, this._opts, {
            wasm: Object.assign({}, this._opts?.wasm, { numThreads: 1 }),
          });
          await this.load(this._modelUrl, this._metaUrl, opts2);
          res = await this.session.run(feeds);
        } else {
          throw e;
        }
      }
      const tInfer1 = performance.now();

      // Post-processing
      const tPost0 = performance.now();
      const names = this.session.outputNames as string[];
      const a = res[names[0]] as any;
      const b = res[names[1]] as any;
      const pick = (ta: any, tb: any) => {
        const da = ta.dims as number[];
        const db = tb.dims as number[];
        if (da.length === 5 && db.length === 4) return { spec: ta, time: tb };
        if (da.length === 4 && db.length === 5) return { spec: tb, time: ta };
        if (da.length >= 3 && da[2] === C * 2) return { spec: ta, time: tb };
        if (db.length >= 3 && db[2] === C * 2) return { spec: tb, time: ta };
        return { spec: ta, time: tb };
      };
      const { spec: specOut, time: timeOut } = pick(a, b);
      const specBuf = specOut.data as Float32Array; // [1,S,C*2,F,T]
      const timeBuf = timeOut.data as Float32Array; // [1,S,C,T]
      const B = 1,
        S_ = 4,
        C_ = C;
      const Tseg = segLen;
      const idxSpec = (
        b: number,
        s: number,
        cc: number,
        f: number,
        t: number
      ) => {
        const Cspec = C_ * 2;
        return (((b * S_ + s) * Cspec + cc) * F + f) * TT + t;
      };
      for (let s = 0; s < S_; s++) {
        for (let c = 0; c < C_; c++) {
          const plane = new Float32Array(2 * F * TT);
          const realBase = 0;
          const imagBase = F * TT;
          const ccReal = 2 * c;
          const ccImag = 2 * c + 1;
          for (let f = 0; f < F; f++) {
            for (let t = 0; t < TT; t++) {
              plane[realBase + f * TT + t] =
                specBuf[idxSpec(0, s, ccReal, f, t)];
              plane[imagBase + f * TT + t] =
                specBuf[idxSpec(0, s, ccImag, f, t)];
            }
          }
          const { real, imag } = unpackCAC(plane, F, TT);
          const x = iSpecLike(real, imag, Tseg, nfft);
          const tb = timeBuf.subarray(
            (((B - 1) * S_ + s) * C_ + c) * Tseg,
            (((B - 1) * S_ + s) * C_ + c + 1) * Tseg
          );
          for (let t = 0; t < Math.min(Tseg, T - off); t++) {
            const specVal = x[t];
            const timeVal = tb[t];
            outSum[s][c][off + t] += weight[t] * (specVal + timeVal);
            outSpec[s][c][off + t] += weight[t] * specVal;
            outTime[s][c][off + t] += weight[t] * timeVal;
          }
        }
      }
      const tPost1 = performance.now();

      if (perfEnabled) {
        const preMs = tPre1 - tPre0;
        const inferMs = tInfer1 - tInfer0;
        const postMs = tPost1 - tPost0;
        console.log(
          `[demucs] chunk ${
            chunkIdx + 1
          }/${totalChunks} end: pre=${preMs.toFixed(
            1
          )}ms, infer=${inferMs.toFixed(1)}ms, post=${postMs.toFixed(
            1
          )}ms, total=${(tPost1 - chunkStart).toFixed(1)}ms`
        );
        preSum += preMs;
        inferSum += inferMs;
        postSum += postMs;
        chunkCount++;
      }

      chunkIdx++;
      if (opts?.onProgress)
        opts.onProgress(Math.min(1, chunkIdx / totalChunks));
    }

    // Normalize by sum of weights used
    const sumW = new Float32Array(T);
    for (const off of offsets) {
      if (off >= maxT) break;
      const len = Math.min(segLen, T - off);
      for (let t = 0; t < len; t++) sumW[off + t] += weight[t];
    }
    const names = ["drums", "bass", "other", "vocals"];
    const makeStems = (src: Float32Array[][]): Record<string, Float32Array> => {
      const stems: Record<string, Float32Array> = {
        drums: new Float32Array(T * C),
        bass: new Float32Array(T * C),
        other: new Float32Array(T * C),
        vocals: new Float32Array(T * C),
      };
      for (let s = 0; s < S; s++)
        for (let c = 0; c < C; c++)
          for (let t = 0; t < T; t++) {
            const v = sumW[t] > 0 ? src[s][c][t] / sumW[t] : 0;
            stems[names[s]][t * C + c] = v;
          }
      return stems;
    };

    const stemsSum = makeStems(outSum);
    const stemsSpec = makeStems(outSpec);
    const stemsTime = makeStems(outTime);

    const provider =
      (this.session as any).executionProviders?.[0]?.name ?? "wasm";
    if (perfEnabled && chunkCount > 0) {
      const overallEnd = performance.now();
      const totalMs = overallEnd - overallStart;
      console.log(
        `[demucs] summary: chunks=${chunkCount}, pre_total=${preSum.toFixed(
          1
        )}ms, infer_total=${inferSum.toFixed(
          1
        )}ms, post_total=${postSum.toFixed(1)}ms, avg_infer=${(
          inferSum / chunkCount
        ).toFixed(1)}ms, total=${totalMs.toFixed(1)}ms, provider=${provider}`
      );
    }
    return {
      results: { sum: stemsSum, spec: stemsSpec, time: stemsTime },
      provider,
    };
  }

  // Input: Float32Array PCM interleaved stereo at meta.samplerate
  async separate(
    pcm: Float32Array,
    opts?: {
      maxSeconds?: number;
      onProgress?: (p: number) => void;
      segmentSeconds?: number;
      logPerf?: boolean;
      combineMode?: "sum" | "spec" | "time"; // debug: choose how to combine branches
    }
  ): Promise<{ stems: Record<string, Float32Array>; provider: string }> {
    const { samplerate, channels, nfft, segment } = this.meta;
    const C = channels;
    const T = Math.floor(pcm.length / C);
    // deinterleave
    const ch: Float32Array[] = new Array(C);
    for (let c = 0; c < C; c++) ch[c] = new Float32Array(T);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++) ch[c][t] = pcm[t * C + c];

    const segSec =
      opts?.segmentSeconds && opts.segmentSeconds > 0
        ? opts.segmentSeconds
        : segment;
    const segLen = Math.floor(segSec * samplerate);
    const stride = Math.max(1, Math.floor(0.75 * segLen));
    const offsets: number[] = [];
    for (let off = 0; off < T; off += stride) offsets.push(off);

    const S = 4; // drums,bass,other,vocals
    const out: Float32Array[][] = new Array(S);
    for (let s = 0; s < S; s++)
      out[s] = new Array(C).fill(0).map(() => new Float32Array(T));

    // triangular weights
    const weight = new Float32Array(segLen);
    for (let i = 0; i < Math.floor(segLen / 2) + 1; i++) weight[i] = i + 1;
    for (let i = Math.floor(segLen / 2) + 1; i < segLen; i++)
      weight[i] = segLen - i;
    let wmax = 0;
    for (let i = 0; i < segLen; i++) wmax = Math.max(wmax, weight[i]);
    for (let i = 0; i < segLen; i++) weight[i] /= wmax;

    const maxT = opts?.maxSeconds
      ? Math.min(T, Math.floor(opts.maxSeconds * samplerate))
      : T;
    const totalChunks = Math.ceil(maxT / stride);
    const perfEnabled = !!opts?.logPerf;
    let preSum = 0,
      inferSum = 0,
      postSum = 0,
      chunkCount = 0;
    const overallStart = performance.now();
    let chunkIdx = 0;

    for (const off of offsets) {
      if (off >= maxT) break;
      const chunkStart = performance.now();
      if (perfEnabled)
        console.log(
          `[demucs] chunk ${chunkIdx + 1}/${totalChunks} start: off=${off} (${(
            off / samplerate
          ).toFixed(2)}s)`
        );

      // Prepare chunk
      const chunk: Float32Array[] = new Array(C);
      for (let c = 0; c < C; c++) {
        chunk[c] = new Float32Array(segLen);
        const len = Math.min(segLen, T - off);
        chunk[c].set(ch[c].subarray(off, off + len));
      }

      // Pre-processing: STFT + CAC pack
      const tPre0 = performance.now();
      const specMag: Float32Array[] = [];
      let F = 0,
        TT = 0;
      for (let c = 0; c < C; c++) {
        const { real, imag } = specLike(chunk[c], nfft);
        F = real[0].length;
        TT = real.length;
        specMag.push(packCAC(real, imag));
      }
      const tPre1 = performance.now();

      // Prepare feeds
      const mag = new Float32Array(C * 2 * F * TT);
      for (let c = 0; c < C; c++) mag.set(specMag[c], c * 2 * F * TT);

      const mixBuf = new Float32Array(C * segLen);
      for (let c = 0; c < C; c++) mixBuf.set(chunk[c], c * segLen);
      const feeds: Record<string, any> = {
        mag: new ort.Tensor("float32", mag, [1, C * 2, F, TT]),
        mix: new ort.Tensor("float32", mixBuf, [1, C, segLen]),
      };

      // Inference
      let res: any;
      const tInfer0 = performance.now();
      try {
        res = await this.session.run(feeds);
      } catch (e) {
        console.error("ORT run failed. Shapes:", {
          mag: [1, C * 2, F, TT],
          mix: [1, C, segLen],
          providerList: (this.session as any).executionProviders,
          wasmEnv: (ort as any).env?.wasm,
        });
        const ep = (this.session as any).executionProviders?.[0]?.name;
        const threads = (ort as any).env?.wasm?.numThreads;
        if (ep === "wasm" && threads && threads > 1) {
          console.warn("Retrying with single-threaded WASM after failure...");
          (ort as any).env.wasm.numThreads = 1;
          const opts2 = Object.assign({}, this._opts, {
            wasm: Object.assign({}, this._opts?.wasm, { numThreads: 1 }),
          });
          await this.load(this._modelUrl, this._metaUrl, opts2);
          res = await this.session.run(feeds);
        } else {
          throw e;
        }
      }
      const tInfer1 = performance.now();

      // Post-processing
      const tPost0 = performance.now();
      const names = this.session.outputNames as string[];
      const a = res[names[0]] as any;
      const b = res[names[1]] as any;
      const pick = (ta: any, tb: any) => {
        const da = ta.dims as number[];
        const db = tb.dims as number[];
        // spec_out: [B,S,C*2,F,T] (len=5), time_out: [B,S,C,T] (len=4)
        if (da.length === 5 && db.length === 4) return { spec: ta, time: tb };
        if (da.length === 4 && db.length === 5) return { spec: tb, time: ta };
        // Fallback by checking C*2 dimension match
        if (da.length >= 3 && da[2] === C * 2) return { spec: ta, time: tb };
        if (db.length >= 3 && db[2] === C * 2) return { spec: tb, time: ta };
        // As last resort assume original order
        return { spec: ta, time: tb };
      };
      const { spec: specOut, time: timeOut } = pick(a, b);
      const specBuf = specOut.data as Float32Array; // [1,S,C*2,F,T]
      const timeBuf = timeOut.data as Float32Array; // [1,S,C,T]
      const B = 1,
        S_ = 4,
        C_ = C;
      const Tseg = segLen;
      // Helper to compute linear index into spec_out: [B,S,Cspec,F,T]
      const idxSpec = (
        b: number,
        s: number,
        cc: number,
        f: number,
        t: number
      ) => {
        const Cspec = C_ * 2;
        return (((b * S_ + s) * Cspec + cc) * F + f) * TT + t;
      };
      for (let s = 0; s < S_; s++) {
        for (let c = 0; c < C_; c++) {
          // Gather per-channel CAC planes: cc0=2*c (real), cc1=2*c+1 (imag)
          const plane = new Float32Array(2 * F * TT);
          const realBase = 0;
          const imagBase = F * TT;
          const ccReal = 2 * c;
          const ccImag = 2 * c + 1;
          for (let f = 0; f < F; f++) {
            for (let t = 0; t < TT; t++) {
              plane[realBase + f * TT + t] =
                specBuf[idxSpec(0, s, ccReal, f, t)];
              plane[imagBase + f * TT + t] =
                specBuf[idxSpec(0, s, ccImag, f, t)];
            }
          }
          const { real, imag } = unpackCAC(plane, F, TT);
          const x = iSpecLike(real, imag, Tseg, nfft);
          const tb = timeBuf.subarray(
            (((B - 1) * S_ + s) * C_ + c) * Tseg,
            (((B - 1) * S_ + s) * C_ + c + 1) * Tseg
          );
          const mode = opts?.combineMode ?? "sum";
          for (let t = 0; t < Math.min(Tseg, T - off); t++) {
            const specVal = x[t];
            const timeVal = tb[t];
            const v =
              mode === "sum"
                ? specVal + timeVal
                : mode === "spec"
                ? specVal
                : timeVal;
            out[s][c][off + t] += weight[t] * v;
          }
        }
      }
      const tPost1 = performance.now();

      // Perf log per chunk
      if (perfEnabled) {
        const preMs = tPre1 - tPre0;
        const inferMs = tInfer1 - tInfer0;
        const postMs = tPost1 - tPost0;
        console.log(
          `[demucs] chunk ${
            chunkIdx + 1
          }/${totalChunks} end: pre=${preMs.toFixed(
            1
          )}ms, infer=${inferMs.toFixed(1)}ms, post=${postMs.toFixed(
            1
          )}ms, total=${(tPost1 - chunkStart).toFixed(1)}ms`
        );
        preSum += preMs;
        inferSum += inferMs;
        postSum += postMs;
        chunkCount++;
      }

      chunkIdx++;
      if (opts?.onProgress)
        opts.onProgress(Math.min(1, chunkIdx / totalChunks));
    }

    // Normalize by sum of weights used
    const sumW = new Float32Array(T);
    for (const off of offsets) {
      if (off >= maxT) break;
      const len = Math.min(segLen, T - off);
      for (let t = 0; t < len; t++) sumW[off + t] += weight[t];
    }
    if (perfEnabled) {
      let last = T - 1;
      while (last >= 0 && sumW[last] === 0) last--;
      if (last < T - 1) {
        console.warn(
          `[demucs] coverage ends at ${(last / samplerate).toFixed(
            2
          )}s out of ${(T / samplerate).toFixed(2)}s`
        );
      }
    }
    const stems: Record<string, Float32Array> = {
      drums: new Float32Array(T * C),
      bass: new Float32Array(T * C),
      other: new Float32Array(T * C),
      vocals: new Float32Array(T * C),
    };
    const names = ["drums", "bass", "other", "vocals"];
    for (let s = 0; s < S; s++)
      for (let c = 0; c < C; c++)
        for (let t = 0; t < T; t++) {
          const v = sumW[t] > 0 ? out[s][c][t] / sumW[t] : 0;
          stems[names[s]][t * C + c] = v;
        }

    const provider =
      (this.session as any).executionProviders?.[0]?.name ?? "wasm";
    if (perfEnabled && chunkCount > 0) {
      const overallEnd = performance.now();
      const totalMs = overallEnd - overallStart;
      console.log(
        `[demucs] summary: chunks=${chunkCount}, pre_total=${preSum.toFixed(
          1
        )}ms, infer_total=${inferSum.toFixed(
          1
        )}ms, post_total=${postSum.toFixed(1)}ms, avg_infer=${(
          inferSum / chunkCount
        ).toFixed(1)}ms, total=${totalMs.toFixed(1)}ms, provider=${provider}`
      );
    }
    return { stems, provider };
  }
}
