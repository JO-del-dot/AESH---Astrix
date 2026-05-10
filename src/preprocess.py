"""
GeoRadar — Preprocessing Pipeline
Team Astrix · AESH 2026 · Green Radar Systems

Stage 2 of the pipeline:
  - Bandpass filtering (remove out-of-band noise)
  - Time-gain correction (compensate for depth-dependent attenuation)
  - Background subtraction (remove direct-wave clutter)
  - Envelope detection (Hilbert transform → instantaneous amplitude)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d


# ── Bandpass filter ────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, f_low: float, f_high: float,
                    order: int = 5) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        signal: 1-D complex or real signal array
        fs:     Sampling frequency (Hz)
        f_low:  Lower cutoff frequency (Hz)
        f_high: Upper cutoff frequency (Hz)
        order:  Filter order

    Returns:
        Filtered signal (same dtype as input)
    """
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")

    if np.iscomplexobj(signal):
        return (sosfiltfilt(sos, signal.real) +
                1j * sosfiltfilt(sos, signal.imag))
    return sosfiltfilt(sos, signal)


# ── Time-gain correction ───────────────────────────────────────────────────────

def time_gain_correction(signal: np.ndarray, t: np.ndarray,
                         gain_exponent: float = 1.5) -> np.ndarray:
    """
    Compensate for geometric spreading and attenuation losses with depth.
    Gain increases as a power of two-way travel time.

    Args:
        signal:        1-D signal array
        t:             Corresponding time axis (s)
        gain_exponent: Power applied to time for gain curve

    Returns:
        Gain-corrected signal
    """
    t_safe = np.where(t > 0, t, t[t > 0].min())   # avoid divide-by-zero at t=0
    gain = t_safe ** gain_exponent
    gain /= gain.max()                              # normalize to [0, 1]
    return signal * gain


# ── Background subtraction ─────────────────────────────────────────────────────

def background_subtraction(b_scan: np.ndarray, window: int = None) -> np.ndarray:
    """
    Remove direct-wave clutter by subtracting the mean trace from a B-scan
    (2-D array: rows = traces, cols = time samples).

    Args:
        b_scan: 2-D array (n_traces, n_samples)
        window: If set, use a sliding-window mean instead of global mean

    Returns:
        Clutter-removed B-scan
    """
    if window is None:
        mean_trace = b_scan.mean(axis=0, keepdims=True)
    else:
        mean_trace = uniform_filter1d(b_scan, size=window, axis=0)
    return b_scan - mean_trace


# ── Envelope detection ─────────────────────────────────────────────────────────

def envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the instantaneous amplitude (envelope) via the Hilbert transform.
    Works on real or complex input.
    """
    if np.iscomplexobj(signal):
        signal = signal.real
    return np.abs(hilbert(signal))


# ── Full preprocessing pipeline ────────────────────────────────────────────────

def preprocess(
    signal: np.ndarray,
    t: np.ndarray,
    fs: float,
    f_low: float = 150e6,
    f_high: float = 1.1e9,
    tgc_exponent: float = 1.5,
    return_envelope: bool = True,
) -> np.ndarray:
    """
    Run the full preprocessing chain on a single A-scan.

    Args:
        signal:          Raw received signal (complex)
        t:               Time axis (s)
        fs:              Sampling frequency (Hz)
        f_low:           Bandpass lower cutoff (Hz)
        f_high:          Bandpass upper cutoff (Hz)
        tgc_exponent:    Time-gain correction exponent
        return_envelope: If True, return instantaneous amplitude

    Returns:
        Preprocessed 1-D signal (real)
    """
    out = bandpass_filter(signal, fs, f_low, f_high)
    out = time_gain_correction(out, t, tgc_exponent)
    if return_envelope:
        out = envelope(out)
    elif np.iscomplexobj(out):
        out = out.real
    return out


def preprocess_batch(
    X: np.ndarray,
    fs: float,
    f_low: float = 150e6,
    f_high: float = 1.1e9,
    tgc_exponent: float = 1.5,
) -> np.ndarray:
    """
    Preprocess a batch of A-scans.

    Args:
        X: Array of shape (n_samples, n_time_steps, 2)  [real, imag channels]

    Returns:
        Preprocessed array of shape (n_samples, n_time_steps, 1)  [envelope]
    """
    n_samples, n_time, _ = X.shape
    t = np.linspace(0, n_time / fs, n_time)
    out = np.zeros((n_samples, n_time, 1), dtype=np.float32)

    for i in range(n_samples):
        sig = X[i, :, 0] + 1j * X[i, :, 1]
        env = preprocess(sig, t, fs, f_low, f_high, tgc_exponent)
        # Normalize each scan to [0, 1]
        env_max = env.max()
        if env_max > 0:
            env = env / env_max
        out[i, :, 0] = env.astype(np.float32)

    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from signal_gen import simulate_scan

    # Demo: generate one scan and show before/after preprocessing
    layers = [
        {"depth": 1.0,  "eps_r": 3.0,  "sigma": 0.001, "label": 0},
        {"depth": 4.5,  "eps_r": 25.0, "sigma": 0.05,  "label": 1},
        {"depth": 8.0,  "eps_r": 6.5,  "sigma": 0.0005,"label": 2},
    ]
    fs = 10e9
    scan = simulate_scan(layers, fs=fs)
    t = scan["time"]
    raw = scan["rx_signal"]

    proc = preprocess(raw, t, fs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t * 1e9, raw.real, lw=0.5, color="steelblue")
    axes[0].set_title("Raw received signal (real part)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t * 1e9, proc, lw=0.8, color="darkorange")
    axes[1].set_title("After preprocessing (envelope, TGC, bandpass)")
    axes[1].set_xlabel("Two-way travel time (ns)")
    axes[1].set_ylabel("Normalized amplitude")

    plt.tight_layout()
    plt.savefig("../results/preprocessing_demo.png", dpi=150)
    print("Saved: results/preprocessing_demo.png")
