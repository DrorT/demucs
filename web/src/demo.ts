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
      });
      provider = all.provider;
      results.push({ label: "sum", stems: all.results.sum });
      results.push({ label: "spec", stems: all.results.spec });
      results.push({ label: "time", stems: all.results.time });
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
