// Minimal DSP utils to mirror Demucs STFT/iSTFT conventions in JS
// Implements:
// - Reflect padding like pad1d
// - Hann window
// - STFT-ish framing with FFT.js
// - iSTFT with overlap-add and centering
// - CAC magnitude packing/unpacking

// Lightweight Radix-2 FFT (real input forward, inverse) to avoid external deps.
// For performance, consider swapping with WebFFT or a WASM-backed FFT.
class SimpleFFT {
  private n: number;
  private cos: Float32Array;
  private sin: Float32Array;
  constructor(n: number) {
    if ((n & (n - 1)) !== 0) throw new Error("FFT size must be power of two");
    this.n = n;
    this.cos = new Float32Array(n / 2);
    this.sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
  }
  // Complex FFT in-place arrays (separate real/imag)
  fft(re: Float32Array, im: Float32Array, inverse = false) {
    const n = this.n;
    // bit-reverse
    let j = 0;
    for (let i = 0; i < n; i++) {
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
      let m = n >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
    for (let m = 1; m < n; m <<= 1) {
      const step = m << 1;
      for (let k = 0; k < m; k++) {
        const tw = (n / step) * k;
        const wr = this.cos[tw];
        const wi = inverse ? -this.sin[tw] : this.sin[tw];
        for (let i = k; i < n; i += step) {
          const j = i + m;
          const tr = wr * re[j] - wi * im[j];
          const ti = wr * im[j] + wi * re[j];
          re[j] = re[i] - tr;
          im[j] = im[i] - ti;
          re[i] += tr;
          im[i] += ti;
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
    }
  }
}

export function hannWindow(N: number): Float32Array {
  const w = new Float32Array(N);
  // Match torch.hann_window default (periodic=True): denominator N
  for (let n = 0; n < N; n++) {
    w[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / N));
  }
  return w;
}

export function reflectPad1d(
  x: Float32Array,
  left: number,
  right: number
): Float32Array {
  const L = x.length;
  const maxPad = Math.max(left, right);
  // Emulate Demucs pad1d reflect behavior on small input: if signal is too short
  // compared to padding, insert extra zero padding before reflection so reflect
  // indexing is valid and avoids clamping artifacts.
  let xWork = x;
  let leftPad = left;
  let rightPad = right;
  if (L <= maxPad) {
    const extra = maxPad - L + 1;
    const extraRight = Math.min(right, extra);
    const extraLeft = extra - extraRight;
    const x2 = new Float32Array(extraLeft + L + extraRight);
    // leading zeros
    // center
    x2.set(x, extraLeft);
    // trailing zeros implicitly present
    xWork = x2;
    leftPad = left - extraLeft;
    rightPad = right - extraRight;
  }
  const LW = xWork.length;
  const y = new Float32Array(leftPad + LW + rightPad);
  // left reflect: farthest first -> x[left] ... x[2], x[1]
  for (let i = 0; i < leftPad; i++) {
    const src = 1 + i; // skip edge sample x[0]
    y[leftPad - 1 - i] = xWork[src];
  }
  // center
  y.set(xWork, leftPad);
  // right reflect: x[L-2], x[L-3], ..., x[L-right-1]
  for (let i = 0; i < rightPad; i++) y[leftPad + LW + i] = xWork[LW - 2 - i];
  return y;
}

export function stft(
  x: Float32Array,
  nfft: number,
  hop: number
): { real: Float32Array[]; imag: Float32Array[] } {
  const win = hannWindow(nfft);
  const fft = new SimpleFFT(nfft);
  const scale = 1 / Math.sqrt(nfft); // PyTorch stft(normalized=True)
  const frames: { real: Float32Array; imag: Float32Array }[] = [];
  for (let start = 0; start + nfft <= x.length; start += hop) {
    const buf = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) buf[i] = x[start + i] * win[i];
    // Prepare complex arrays and run FFT
    const re = buf.slice();
    const im = new Float32Array(nfft);
    fft.fft(re, im, false);
    // Keep one-sided spectrum [0..nfft/2]
    const real = new Float32Array(nfft / 2 + 1);
    const imag = new Float32Array(nfft / 2 + 1);
    for (let k = 0; k <= nfft / 2; k++) {
      real[k] = re[k] * scale;
      imag[k] = im[k] * scale;
    }
    frames.push({ real, imag });
  }
  return {
    real: frames.map((f) => f.real),
    imag: frames.map((f) => f.imag),
  };
}

export function istft(
  realFrames: Float32Array[],
  imagFrames: Float32Array[],
  nfft: number,
  hop: number,
  length: number
): Float32Array {
  const win = hannWindow(nfft);
  const fft = new SimpleFFT(nfft);
  const invNorm = Math.sqrt(nfft); // compensate for stft normalized=True
  const out = new Float32Array(length);
  const norm = new Float32Array(length);
  for (let t = 0; t < realFrames.length; t++) {
    const real = realFrames[t];
    const imag = imagFrames[t];
    // Rebuild full complex spectrum by symmetry
    const re = new Float32Array(nfft);
    const im = new Float32Array(nfft);
    for (let k = 0; k <= nfft / 2; k++) {
      re[k] = real[k] * invNorm;
      im[k] = imag[k] * invNorm;
    }
    for (let k = 1; k < nfft / 2; k++) {
      re[nfft - k] = real[k] * invNorm;
      im[nfft - k] = -imag[k] * invNorm;
    }
    fft.fft(re, im, true);
    const time = re;
    const start = t * hop;
    for (let i = 0; i < nfft && start + i < length; i++) {
      out[start + i] += time[i] * win[i];
      norm[start + i] += win[i] * win[i];
    }
  }
  for (let i = 0; i < length; i++) {
    out[i] = norm[i] > 1e-8 ? out[i] / norm[i] : 0;
  }
  return out;
}

export function specLike(
  x: Float32Array,
  nfft: number
): { real: Float32Array[]; imag: Float32Array[]; le: number; pad: number } {
  const hl = nfft >> 2;
  const length = x.length;
  const le = Math.ceil(length / hl);
  const pad = Math.floor((hl >> 1) * 3);
  const padded = reflectPad1d(x, pad, pad + (le * hl - length));
  // Center=True behavior: extra reflect pad of nfft/2 on both sides
  const centerPad = nfft >> 1;
  const centered = reflectPad1d(padded, centerPad, centerPad);
  const st = stft(centered, nfft, hl);
  // Drop last frequency bin to match Python: z = z[..., :-1, :]
  const dropLastFreq = (frame: Float32Array) =>
    frame.subarray(0, frame.length - 1);
  const realAll = st.real.map((f) => new Float32Array(dropLastFreq(f)));
  const imagAll = st.imag.map((f) => new Float32Array(dropLastFreq(f)));
  // slice frames [2 : 2 + le]
  const real = realAll.slice(2, 2 + le);
  const imag = imagAll.slice(2, 2 + le);
  return { real, imag, le, pad };
}

export function iSpecLike(
  realFrames: Float32Array[],
  imagFrames: Float32Array[],
  length: number,
  nfft: number
): Float32Array {
  const hl = nfft >> 2;
  const pad = Math.floor((hl >> 1) * 3);
  const le = Math.floor(hl * Math.ceil(length / hl) + 2 * pad);
  // Add back the last frequency bin (Nyquist) we dropped earlier
  const F = realFrames[0].length;
  const F1 = F + 1;
  const realAdd: Float32Array[] = realFrames.map((fr) => {
    const f1 = new Float32Array(F1);
    f1.set(fr, 0);
    f1[F] = 0;
    return f1;
  });
  const imagAdd: Float32Array[] = imagFrames.map((fi) => {
    const f1 = new Float32Array(F1);
    f1.set(fi, 0);
    f1[F] = 0;
    return f1;
  });

  // Helper to create a reflected frame for padding
  const reflectFrame = (frames: Float32Array[], side: 'left' | 'right', index: number): Float32Array => {
    // side: 'left' reflects from the start of frames, 'right' from the end
    // index: 0 for the first padding frame, 1 for the second
    const sourceFrame = side === 'left' ? frames[index + 1] : frames[frames.length - 2 - index];
    if (!sourceFrame) {
      // Fallback to zero frame if source is unavailable (should not happen with 2 pads)
      return new Float32Array(F1);
    }
    const reflected = new Float32Array(F1);
    for (let i = 0; i < F1; i++) {
      reflected[i] = sourceFrame[F1 - 1 - i];
    }
    return reflected;
  };

  // Insert back missing padding frames in time: 2 left, 2 right, using reflection
  const real = [
    reflectFrame(realAdd, 'left', 1), // Reflect realAdd[1]
    reflectFrame(realAdd, 'left', 0), // Reflect realAdd[0]
    ...realAdd,
    reflectFrame(realAdd, 'right', 0), // Reflect realAdd[realAdd.length - 1]
    reflectFrame(realAdd, 'right', 1), // Reflect realAdd[realAdd.length - 2]
  ];
  const imag = [
    reflectFrame(imagAdd, 'left', 1), // Reflect imagAdd[1]
    reflectFrame(imagAdd, 'left', 0), // Reflect imagAdd[0]
    ...imagAdd,
    reflectFrame(imagAdd, 'right', 0), // Reflect imagAdd[imagAdd.length - 1]
    reflectFrame(imagAdd, 'right', 1), // Reflect imagAdd[imagAdd.length - 2]
  ];
  // Reconstruct with n_fft = 2*F (since frames now have F+1 bins)
  const nfftRec = 2 * F;
  const y = istft(real, imag, nfftRec, hl, le);
  return y.slice(pad, pad + length);
}

export function packCAC(
  real: Float32Array[],
  imag: Float32Array[]
): Float32Array {
  // Returns [C*2, F, T] with C=channels (we only process per-channel in caller)
  // Here we assume shape [F, T] and pack last dim as [2]
  const F = real[0].length;
  const T = real.length;
  const out = new Float32Array(2 * F * T);
  const realBase = 0;
  const imagBase = F * T;
  // Match PyTorch pack: planes [2, F, T] with T contiguous, then reshape to [2*F, T]
  for (let f = 0; f < F; f++) {
    for (let t = 0; t < T; t++) {
      out[realBase + f * T + t] = real[t][f];
      out[imagBase + f * T + t] = imag[t][f];
    }
  }
  return out;
}

export function unpackCAC(
  buf: Float32Array,
  F: number,
  T: number
): { real: Float32Array[]; imag: Float32Array[] } {
  const real: Float32Array[] = new Array(T);
  const imag: Float32Array[] = new Array(T);
  const realBase = 0;
  const imagBase = F * T;
  for (let t = 0; t < T; t++) {
    real[t] = new Float32Array(F);
    imag[t] = new Float32Array(F);
  }
  for (let f = 0; f < F; f++) {
    for (let t = 0; t < T; t++) {
      real[t][f] = buf[realBase + f * T + t];
      imag[t][f] = buf[imagBase + f * T + t];
    }
  }
  return { real, imag };
}

export function interleave(stems: Float32Array[]): Float32Array {
  const T = stems[0].length;
  const C = stems.length;
  const out = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      out[t * C + c] = stems[c][t];
    }
  }
  return out;
}
