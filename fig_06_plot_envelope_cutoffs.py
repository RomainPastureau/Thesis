import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert

def plot_envelope(wav_path, cutoffs=(1, (4, 8), (20, 50)), colors=("#a8c6db", "orange", "red", "#4aa000"),
                  mode="same", legend=True, axes=True, discard_first_s = 0):
    """
    Plot waveform and its amplitude envelope.
    - If an entry in `cutoffs` is a number -> apply LOW-PASS at that cutoff (Hz) to the envelope.
    - If an entry is a 2-tuple (low, high) -> apply BAND-PASS (Hz) to the envelope.
    """

    # Load audio
    fs, x = wavfile.read(wav_path)

    # Convert to mono and float [-1, 1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        peak = max(abs(x.min()), abs(x.max()))
        x = x.astype(np.float32) / (peak if peak > 0 else 1.0)
    else:
        x = x.astype(np.float32)

    # Discard the first N seconds
    n0 = int(discard_first_s * fs)
    x = x[n0:]
    t = np.arange(len(x)) / fs

    # Detrend
    x = x - np.mean(x)

    # Compute analytic envelope
    envelope = np.abs(hilbert(x))
    nyq = fs / 2.0
    envelopes_filtered = []
    labels = []

    # Robust zero-phase filtering padding
    padlen = int(min(len(x) - 1, fs * 0.25)) if len(x) > 1 else 1

    for c in cutoffs:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            lo, hi = float(c[0]), float(c[1])
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError("Band-pass cutoffs must be finite numbers.")
            if lo <= 0 or hi <= 0:
                raise ValueError("Band-pass cutoffs must be > 0 Hz.")
            if not lo < hi:
                raise ValueError(f"Band-pass requires low < high, got {lo} >= {hi}.")
            if hi >= nyq:
                raise ValueError(f"High cutoff must be < Nyquist ({nyq:.2f} Hz). Got {hi} Hz.")

            sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
            env_filt = sosfiltfilt(sos, envelope, padtype="odd", padlen=padlen)
            label = f"Envelope (Hilbert + BP {lo:g}–{hi:g} Hz), scaled"

        else:
            cutoff = float(c)
            if not np.isfinite(cutoff):
                raise ValueError("Low-pass cutoff must be a finite number.")
            if cutoff <= 0:
                raise ValueError("Low-pass cutoff must be > 0 Hz.")
            if cutoff >= nyq:
                raise ValueError(f"Cutoff must be < Nyquist ({nyq:.2f} Hz). Got {cutoff} Hz.")

            sos = butter(4, cutoff / nyq, btype="lowpass", output="sos")
            env_filt = sosfiltfilt(sos, envelope, padtype="odd", padlen=padlen)
            label = f"Envelope (Hilbert + LP {cutoff:g} Hz), scaled"

        # Scale envelope to match waveform amplitude for neat overlay
        wf_peak = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
        env_peak = np.max(env_filt) if np.max(env_filt) > 0 else 1.0
        envelope_scaled = env_filt * (wf_peak / env_peak)

        envelopes_filtered.append(envelope_scaled)
        labels.append(label)

    # Plot
    if mode == "same":
        plt.figure(figsize=(12, 4))
        for i, env in enumerate(envelopes_filtered):
            plt.plot(t, env, linewidth=2.0, label=labels[i], color=colors[(i + 1) % len(colors)])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if axes is False:
            plt.axis("off")
        if legend:
            plt.legend()

    elif mode == "multi":
        fig, ax = plt.subplots(len(envelopes_filtered), 1, figsize=(12, 4 * len(envelopes_filtered)), sharex=True)
        if len(envelopes_filtered) == 1:
            ax = [ax]
        for i, env in enumerate(envelopes_filtered):
            ax[i].plot(t, x, linewidth=0.6, alpha=0.9, label="Waveform (raw)", color=colors[0])
            ax[i].plot(t, env, linewidth=2.0, label=labels[i], color=colors[(i + 1) % len(colors)])
            ax[i].set_ylabel("Amplitude")
            if axes is False:
                ax[i].axis("off")
            if legend:
                ax[i].legend()
        ax[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file = "zut, l'aspirateur.wav"
    discard_first_s = 2.3
    plot_envelope(file, mode="multi", legend=False, axes=False, cutoffs=(1, 8, 50), discard_first_s = discard_first_s)
    plot_envelope(file, mode="same", legend=False, axes=False, cutoffs=((30, 50), (2, 4), (0.01, 1)), colors=("#a8c6db", "#4aa000", "red", "orange"), discard_first_s = discard_first_s)
