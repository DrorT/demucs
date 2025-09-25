import { DemucsOnnx } from "./index";

function toWavBlob(
  interleaved: Float32Array,
  sampleRate: number,
  numChannels: number
): Blob {
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + interleaved.length * bytesPerSample);
  const view = new DataView(buffer);
  function writeString(offset: number, s: string) {
    for (let i = 0; i < s.length; i++)
      view.setUint8(offset + i, s.charCodeAt(i));
  }
  const pcm16 = new Int16Array(interleaved.length);
  for (let i = 0; i < interleaved.length; i++) {
    const v = Math.max(-1, Math.min(1, interleaved[i]));
    pcm16[i] = v < 0 ? v * 32768 : v * 32767;
  }
  writeString(0, "RIFF");
  view.setUint32(4, 36 + pcm16.byteLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, pcm16.byteLength, true);
  new Int16Array(buffer, 44).set(pcm16);
  return new Blob([buffer], { type: "audio/wav" });
}

const demucs = new DemucsOnnx();

const fileEl = document.getElementById("file") as HTMLInputElement;
const runEl = document.getElementById("run") as HTMLButtonElement;
const deviceEl = document.getElementById("device") as HTMLSelectElement;
const maxsecEl = document.getElementById("maxsec") as HTMLInputElement;
const threadsEl = document.getElementById("threads") as HTMLInputElement;
const segsecEl = document.getElementById("segsec") as HTMLInputElement;
const simdEl = document.getElementById("simd") as HTMLInputElement;
const progEl = document.getElementById("prog") as HTMLProgressElement;
const origEl = document.getElementById("orig") as HTMLAudioElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const stemsEl = document.getElementById("stems") as HTMLDivElement;
const logperfEl = document.getElementById("logperf") as HTMLInputElement;
const allCombinesEl = document.getElementById(
  "allcombines"
) as HTMLInputElement;
const debugstatsEl = document.getElementById("debugstats") as HTMLInputElement;
const mcprojEl = document.getElementById("mcproj") as HTMLInputElement;
const leakSupEl = document.getElementById("leaksup") as HTMLInputElement;
const debugdumpEl = document.getElementById("debugdump") as HTMLInputElement;

let arrayBuffer: ArrayBuffer | null = null;

fileEl.addEventListener("change", async () => {
  const f = fileEl.files?.[0];
  if (!f) return;
  arrayBuffer = await f.arrayBuffer();
  origEl.src = URL.createObjectURL(new Blob([arrayBuffer]));
});

runEl.addEventListener("click", async () => {
  try {
    if (!arrayBuffer) return;
    statusEl.textContent = "Loading model…";
    const numThreads = Math.max(1, parseInt(threadsEl?.value || "1", 10));
    await demucs.load("/models/htdemucs_core.onnx", "/models/meta.json", {
      device: (deviceEl?.value as any) || "wasm",
      wasm: {
        numThreads,
        simd: !!simdEl?.checked,
        proxy: numThreads === 1 ? "none" : "worker",
        logLevel: "info",
      },
    });
    const meta = demucs.getMeta();
    statusEl.textContent = "Decoding audio…";
    const ac = new AudioContext({ sampleRate: meta.samplerate });
    const decoded = await ac.decodeAudioData(arrayBuffer.slice(0));
    const C = decoded.numberOfChannels;
    const T = decoded.length;
    const pcm = new Float32Array(T * C);
    const ch: Float32Array[] = new Array(C);
    for (let c = 0; c < C; c++) ch[c] = decoded.getChannelData(c);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < C; c++) pcm[t * C + c] = ch[c][t];

    statusEl.textContent = "Separating…";
    progEl.style.display = "block";
    progEl.value = 0;
    const t0 = performance.now();
    const runOnce = async (combineMode?: "sum" | "spec" | "time") =>
      demucs.separate(pcm, {
        maxSeconds:
          Math.max(0, parseInt(maxsecEl?.value || "0", 10)) || undefined,
        onProgress: (p) => {
          progEl.value = p;
        },
        segmentSeconds: Math.max(2, parseFloat(segsecEl?.value || "7.8")),
        logPerf: !!logperfEl?.checked,
        combineMode,
      });
    let provider = "";
    let results: Array<{ label: string; stems: Record<string, Float32Array> }> =
      [];
    if (allCombinesEl?.checked) {
      const tA = performance.now();
      const all = await demucs.separateAll(pcm, {
        maxSeconds:
          Math.max(0, parseInt(maxsecEl?.value || "0", 10)) || undefined,
        onProgress: (p) => (progEl.value = p),
        segmentSeconds: Math.max(2, parseFloat(segsecEl?.value || "7.8")),
        logPerf: !!logperfEl?.checked,
        debugDump: !!debugdumpEl?.checked,
      });
      provider = all.provider;
      results.push({ label: "sum", stems: all.results.sum });
      results.push({ label: "spec", stems: all.results.spec });
      results.push({ label: "time", stems: all.results.time });
      if (all.debug && debugdumpEl?.checked) {
        try {
          const toB64 = (f: Float32Array) => {
            const buf = f.buffer.slice(
              f.byteOffset,
              f.byteOffset + f.byteLength
            );
            let binary = "";
            const bytes = new Uint8Array(buf);
            const len = bytes.byteLength;
            for (let i = 0; i < len; i++)
              binary += String.fromCharCode(bytes[i]);
            return btoa(binary);
          };
          const meta = demucs.getMeta();
          const payload = {
            meta: {
              samplerate: meta.samplerate,
              channels: meta.channels,
              nfft: meta.nfft,
              cac: meta.cac,
            },
            shapes: all.debug.shapes,
            tensors: {
              mag: {
                dtype: "float32",
                shape: all.debug.shapes.mag,
                data: toB64(all.debug.mag),
              },
              mix: {
                dtype: "float32",
                shape: all.debug.shapes.mix,
                data: toB64(all.debug.mix),
              },
              spec_out: {
                dtype: "float32",
                shape: all.debug.shapes.spec_out,
                data: toB64(all.debug.spec_out),
              },
              time_out: {
                dtype: "float32",
                shape: all.debug.shapes.time_out,
                data: toB64(all.debug.time_out),
              },
            },
          };
          const blob = new Blob([JSON.stringify(payload, null, 2)], {
            type: "application/json",
          });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "demucs_debug_pack.json";
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        } catch (e) {
          console.error("Failed to create debug pack", e);
        }
      }
      const tB = performance.now();
      console.log(`[demucs] all-combines run took ${(tB - tA).toFixed(0)} ms`);
    } else {
      const single = await runOnce("sum");
      provider = single.provider;
      results.push({ label: "sum", stems: single.stems });
    }
    const t1 = performance.now();
    statusEl.textContent = `Done in ${(t1 - t0).toFixed(
      0
    )} ms (provider: ${provider})`;
    progEl.style.display = "none";

    // Optional branch RMS stats (first 1s) to diagnose residual mix
    if (debugstatsEl?.checked) {
      try {
        const sr = meta.samplerate;
        const names = ["drums", "bass", "other", "vocals"] as const;
        const pick = (label: string) =>
          results.find((r) => r.label === label)?.stems;
        const rsum = pick("sum");
        const rspec = pick("spec");
        const rtime = pick("time");
        if (rsum && rspec && rtime) {
          const oneSecSamples = Math.min(
            sr * meta.channels,
            rsum["drums"].length
          );
          const rms = (buf: Float32Array) => {
            let s = 0;
            for (let i = 0; i < oneSecSamples; i++) s += buf[i] * buf[i];
            return Math.sqrt(s / (oneSecSamples + 1e-9));
          };
          for (const name of names) {
            console.log(
              `[demucs][rms 1s] ${name} | sum=${rms(rsum[name]).toFixed(
                4
              )} spec=${rms(rspec[name]).toFixed(4)} time=${rms(
                rtime[name]
              ).toFixed(4)}`
            );
          }
          // Mixture consistency: sum(stems) vs original mix
          const mix = new Float32Array(pcm.length);
          mix.set(pcm);
          const modes: Array<[string, Record<string, Float32Array>]> = [
            ["sum", rsum],
            ["spec", rspec],
            ["time", rtime],
          ];
          const relRms = (err: Float32Array, ref: Float32Array) => {
            let se = 0,
              sr = 0;
            for (let i = 0; i < err.length; i++) {
              const e = err[i];
              const r = ref[i];
              se += e * e;
              sr += r * r;
            }
            const re = Math.sqrt(se / (err.length + 1e-9));
            const rr = Math.sqrt(sr / (ref.length + 1e-9));
            return re / Math.max(1e-9, rr);
          };
          for (const [label, stems] of modes) {
            // Sum stems back to a mix
            const rec = new Float32Array(pcm.length);
            for (const name of names) {
              const s = stems[name];
              for (let i = 0; i < rec.length; i++) rec[i] += s[i];
            }
            const err = new Float32Array(rec.length);
            for (let i = 0; i < rec.length; i++) err[i] = rec[i] - mix[i];
            const r = relRms(err, mix);
            const db = 20 * Math.log10(Math.max(1e-12, r));
            console.log(
              `[demucs][mix-consistency][pre-mc] mode=${label} relRMS=${r.toFixed(
                6
              )} (${db.toFixed(1)} dB)`
            );
          }
        }
      } catch {}
    }

    // Optional: Mixture-consistency projection (energy-weighted per-sample)
    if (mcprojEl?.checked) {
      const names = ["drums", "bass", "other", "vocals"];
      const C = meta.channels;
      const project = (stems: Record<string, Float32Array>) => {
        const T = stems[names[0]].length / C;
        const mix = new Float32Array(T * C);
        mix.set(pcm);
        const sum = new Float32Array(T * C);
        for (const n of names) {
          const s = stems[n];
          for (let i = 0; i < sum.length; i++) sum[i] += s[i];
        }
        const resid = new Float32Array(T * C);
        for (let i = 0; i < resid.length; i++) resid[i] = mix[i] - sum[i];
        // Compute energy weights per-sample across stems: w_s = |s| / sum(|s|)+eps
        const eps = 1e-6;
        const w: Record<string, Float32Array> = {} as any;
        for (const n of names) w[n] = new Float32Array(T * C);
        for (let i = 0; i < T * C; i++) {
          let denom = eps;
          for (const n of names) denom += Math.abs(stems[n][i]);
          for (const n of names) w[n][i] = Math.abs(stems[n][i]) / denom;
        }
        // Apply correction proportionally
        for (const n of names) {
          const s = stems[n];
          const wn = w[n];
          for (let i = 0; i < s.length; i++) s[i] += resid[i] * wn[i];
        }
      };
      for (const res of results) project(res.stems);
      // Post-projection consistency log
      if (debugstatsEl?.checked) {
        try {
          const mix = new Float32Array(pcm.length);
          mix.set(pcm);
          const names = ["drums", "bass", "other", "vocals"] as const;
          const modes: Array<[string, Record<string, Float32Array>]> =
            results.map((r) => [r.label, r.stems]);
          const relRms = (err: Float32Array, ref: Float32Array) => {
            let se = 0,
              sr = 0;
            for (let i = 0; i < err.length; i++) {
              const e = err[i],
                r = ref[i];
              se += e * e;
              sr += r * r;
            }
            const re = Math.sqrt(se / (err.length + 1e-9));
            const rr = Math.sqrt(sr / (ref.length + 1e-9));
            return re / Math.max(1e-9, rr);
          };
          for (const [label, stems] of modes) {
            const rec = new Float32Array(pcm.length);
            for (const name of names) {
              const s = stems[name];
              for (let i = 0; i < rec.length; i++) rec[i] += s[i];
            }
            const err = new Float32Array(rec.length);
            for (let i = 0; i < rec.length; i++) err[i] = rec[i] - mix[i];
            const r = relRms(err, mix);
            const db = 20 * Math.log10(Math.max(1e-12, r));
            console.log(
              `[demucs][mix-consistency][post-mc] mode=${label} relRMS=${r.toFixed(
                6
              )} (${db.toFixed(1)} dB)`
            );
          }
        } catch {}
      }
    }

    // Optional: Suppress vocal leakage into other stems by per-channel projection
    if (leakSupEl?.checked) {
      const C = meta.channels;
      const names = ["drums", "bass", "other", "vocals"] as const;
      const suppress = (stems: Record<string, Float32Array>) => {
        const V = stems["vocals"];
        const T = V.length / C;
        // For each non-vocal stem S, compute alpha = (S·V)/(V·V) per channel, then S <- S - alpha V
        const dot = (a: Float32Array, b: Float32Array, ch: number) => {
          let s = 0;
          for (let t = ch; t < a.length; t += C) s += a[t] * b[t];
          return s;
        };
        const sub = (
          dst: Float32Array,
          src: Float32Array,
          alpha: number,
          ch: number
        ) => {
          for (let t = ch; t < dst.length; t += C) dst[t] -= alpha * src[t];
        };
        const vv = [dot(V, V, 0), dot(V, V, 1)];
        const eps = 1e-9;
        for (const name of ["drums", "bass", "other"]) {
          const S = stems[name];
          const av = [
            dot(S, V, 0) / Math.max(eps, vv[0]),
            dot(S, V, 1) / Math.max(eps, vv[1]),
          ];
          sub(S, V, av[0], 0);
          sub(S, V, av[1], 1);
        }
      };
      for (const res of results) suppress(res.stems);
      // If MC was also requested, re-project after suppression to maintain sum consistency
      if (mcprojEl?.checked) {
        const names2 = ["drums", "bass", "other", "vocals"] as const;
        const C2 = meta.channels;
        const project2 = (stems: Record<string, Float32Array>) => {
          const T = stems[names2[0]].length / C2;
          const mix = new Float32Array(T * C2);
          mix.set(pcm);
          const sum = new Float32Array(T * C2);
          for (const n of names2) {
            const s = stems[n];
            for (let i = 0; i < sum.length; i++) sum[i] += s[i];
          }
          const resid = new Float32Array(T * C2);
          for (let i = 0; i < resid.length; i++) resid[i] = mix[i] - sum[i];
          const eps = 1e-6;
          const w: Record<string, Float32Array> = {} as any;
          for (const n of names2) w[n] = new Float32Array(T * C2);
          for (let i = 0; i < T * C2; i++) {
            let denom = eps;
            for (const n of names2) denom += Math.abs(stems[n][i]);
            for (const n of names2) w[n][i] = Math.abs(stems[n][i]) / denom;
          }
          for (const n of names2) {
            const s = stems[n],
              wn = w[n];
            for (let i = 0; i < s.length; i++) s[i] += resid[i] * wn[i];
          }
        };
        for (const res of results) project2(res.stems);
      }
    }

    stemsEl.innerHTML = "";
    for (const res of results) {
      const group = document.createElement("div");
      group.innerHTML = `<h3>Combine: ${res.label}</h3>`;
      const grid = document.createElement("div");
      grid.className = "grid";
      for (const name of ["drums", "bass", "other", "vocals"]) {
        const blob = toWavBlob(res.stems[name], meta.samplerate, meta.channels);
        const url = URL.createObjectURL(blob);
        const div = document.createElement("div");
        div.className = "stem";
        div.innerHTML = `<h4>${name}</h4><audio controls src="${url}"></audio>`;
        grid.appendChild(div);
      }
      group.appendChild(grid);
      stemsEl.appendChild(group);
    }
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Error: " + (e as Error).message;
    progEl.style.display = "none";
  }
});
