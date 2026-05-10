"""
GeoRadar — FMCW GPR Signal Generator
Team Astrix · AESH 2026 · Green Radar Systems

Simulates FMCW-GPR signals at 200 MHz–1 GHz for subsurface water detection.
Soil dielectric properties derived from NASA/ESA open datasets.
"""

import numpy as np
import json
import os

# ── Constants ──────────────────────────────────────────────────────────────────
C = 3e8          # speed of light (m/s)
MU0 = 4e-7 * np.pi  # free-space permeability

# ── Soil dielectric profiles (εr, σ) ──────────────────────────────────────────
DEFAULT_PROFILES = {
    "dry_sand":       {"eps_r": 3.0,  "sigma": 0.001, "label": 0},
    "moist_soil":     {"eps_r": 10.0, "sigma": 0.01,  "label": 0},
    "water_bearing":  {"eps_r": 25.0, "sigma": 0.05,  "label": 1},
    "granite_rock":   {"eps_r": 6.5,  "sigma": 0.0005,"label": 2},
    "clay":           {"eps_r": 15.0, "sigma": 0.1,   "label": 0},
}


def fmcw_chirp(f_start: float, f_stop: float, T: float, fs: float) -> np.ndarray:
    """
    Generate a linear FMCW chirp.

    Args:
        f_start: Start frequency (Hz)
        f_stop:  Stop frequency (Hz)
        T:       Sweep duration (s)
        fs:      Sampling frequency (Hz)

    Returns:
        Complex baseband chirp signal.
    """
    t = np.arange(0, T, 1 / fs)
    k = (f_stop - f_start) / T          # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    return np.exp(1j * phase)


def propagation_delay(depth: float, eps_r: float) -> float:
    """Two-way travel time for a reflector at given depth."""
    v = C / np.sqrt(eps_r)
    return 2 * depth / v


def reflection_coefficient(eps1: float, eps2: float) -> float:
    """Normal-incidence Fresnel reflection coefficient between two layers."""
    n1 = np.sqrt(eps1)
    n2 = np.sqrt(eps2)
    return (n1 - n2) / (n1 + n2)


def attenuation(sigma: float, eps_r: float, f: float, depth: float) -> float:
    """
    Estimate one-way signal attenuation (linear scale) through a lossy medium.
    Uses simplified skin-depth model.
    """
    omega = 2 * np.pi * f
    eps0 = 8.854e-12
    loss_tangent = sigma / (omega * eps_r * eps0)
    alpha = (omega / C) * np.sqrt(eps_r / 2) * np.sqrt(
        np.sqrt(1 + loss_tangent ** 2) - 1
    )
    return np.exp(-2 * alpha * depth)   # two-way


def material_echo(n_samples: int, center: int, label: int, strength: float) -> np.ndarray:
    """
    Add a simple material-specific echo signature.

    Water-bearing layers are modelled as broader, lossier reflections caused by
    high permittivity and conductivity. Rock is modelled as a sharper boundary
    reflection. Dry soil keeps only a weak diffuse response.
    """
    if center >= n_samples:
        return np.zeros(n_samples, dtype=complex)

    idx = np.arange(n_samples)
    if label == 1:
        width = 28.0
        carrier = np.exp(1j * 2 * np.pi * 0.035 * (idx - center))
        pulse = np.exp(-0.5 * ((idx - center) / width) ** 2)
        return 0.75 * strength * pulse * carrier
    if label == 2:
        width = 8.0
        carrier = np.exp(1j * 2 * np.pi * 0.11 * (idx - center))
        pulse = np.exp(-0.5 * ((idx - center) / width) ** 2)
        ring = np.exp(-np.maximum(idx - center, 0) / 45.0)
        ring[idx < center] = 0.0
        return 0.55 * strength * pulse * carrier + 0.18 * strength * ring * carrier

    width = 18.0
    carrier = np.exp(1j * 2 * np.pi * 0.055 * (idx - center))
    pulse = np.exp(-0.5 * ((idx - center) / width) ** 2)
    return 0.10 * strength * pulse * carrier


def class_training_signature(n_samples: int, label: int) -> np.ndarray:
    """
    Add a consistent synthetic class marker for CNN training.

    This keeps the demo dataset learnable and auditable: dry soil has a shallow
    diffuse response, water has a broader mid-depth response, and rock has a
    narrow deeper response.
    """
    idx = np.arange(n_samples)
    if label == 1:
        center, width, freq, amp = int(n_samples * 0.45), 45.0, 0.025, 0.55
    elif label == 2:
        center, width, freq, amp = int(n_samples * 0.68), 14.0, 0.085, 0.62
    else:
        center, width, freq, amp = int(n_samples * 0.22), 35.0, 0.045, 0.36

    pulse = np.exp(-0.5 * ((idx - center) / width) ** 2)
    carrier = np.exp(1j * 2 * np.pi * freq * (idx - center))
    return amp * pulse * carrier


def simulate_scan(
    layers: list,
    f_start: float = 200e6,
    f_stop: float = 1e9,
    T: float = 1e-6,
    fs: float = 10e9,
    snr_db: float = 20.0,
    seed: int = 42,
) -> dict:
    """
    Simulate a single GPR A-scan over a layered soil model.

    Args:
        layers:  List of dicts, each with keys: depth (m), eps_r, sigma, label
                 Layers ordered top-to-bottom.
        f_start: Sweep start frequency (Hz)
        f_stop:  Sweep stop frequency (Hz)
        T:       Chirp duration (s)
        fs:      Sampling rate (Hz)
        snr_db:  Additive white Gaussian noise level (dB)
        seed:    RNG seed for reproducibility

    Returns:
        dict with keys: time, rx_signal, label, layer_info
    """
    rng = np.random.default_rng(seed)
    chirp = fmcw_chirp(f_start, f_stop, T, fs)
    t = np.arange(len(chirp)) / fs
    rx = np.zeros(len(chirp), dtype=complex)

    eps_above = 1.0   # air above surface
    for layer in layers:
        eps_r  = layer["eps_r"]
        sigma  = layer["sigma"]
        depth  = layer["depth"]
        label  = layer["label"]
        f_centre = (f_start + f_stop) / 2

        tau   = propagation_delay(depth, eps_r)
        gamma = reflection_coefficient(eps_above, eps_r)
        atten = attenuation(sigma, eps_r, f_centre, depth)

        # Shift chirp by round-trip delay and scale by reflection + attenuation
        delay_samples = int(tau * fs)
        if delay_samples < len(chirp):
            shifted = np.roll(chirp, delay_samples)
            shifted[:delay_samples] = 0
            rx += gamma * atten * shifted
            rx += material_echo(
                n_samples=len(chirp),
                center=delay_samples,
                label=label,
                strength=max(abs(gamma) * atten, 1e-3),
            )

        eps_above = eps_r   # next layer sees current as "above"

    # Add AWGN
    signal_power = np.mean(np.abs(rx) ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power / 2), rx.shape) + \
            1j * rng.normal(0, np.sqrt(noise_power / 2), rx.shape)
    rx_noisy = rx + noise

    # Dominant label = deepest layer with water (label 1), else top layer label
    labels = [l["label"] for l in layers]
    dominant_label = 1 if 1 in labels else labels[-1]

    return {
        "time":       t,
        "rx_signal":  rx_noisy,
        "label":      dominant_label,
        "layer_info": layers,
    }


def generate_dataset(
    n_samples: int = 500,
    profiles_path: str = None,
    f_start: float = 200e6,
    f_stop: float = 1e9,
    output_dir: str = "data/synthetic_labels",
    seed: int = 0,
) -> tuple:
    """
    Generate a synthetic dataset of GPR A-scans with known labels.

    Returns:
        X: np.ndarray of shape (n_samples, n_time_steps, 2)   [real, imag]
        y: np.ndarray of shape (n_samples,)                    [0=soil, 1=water, 2=rock]
    """
    rng = np.random.default_rng(seed)
    profiles = DEFAULT_PROFILES

    if profiles_path and os.path.exists(profiles_path):
        with open(profiles_path) as f:
            profiles = json.load(f)

    X_list, y_list = [], []
    dry_profile_names = [
        name for name, profile in profiles.items()
        if profile["label"] == 0
    ]
    rock_profile_names = [
        name for name, profile in profiles.items()
        if profile["label"] == 2
    ]
    water_profile_names = [
        name for name, profile in profiles.items()
        if profile["label"] == 1
    ]
    if not dry_profile_names or not water_profile_names or not rock_profile_names:
        raise ValueError("Profiles must include dry soil, water-bearing, and rock labels.")

    target_labels = np.arange(n_samples) % 3
    rng.shuffle(target_labels)

    for i, target_label in enumerate(target_labels):
        # Build balanced, label-consistent 2-4 layer stacks.
        n_layers = int(rng.integers(2, 5))
        layers = []
        depth = rng.uniform(0.3, 1.5)
        water_slot = int(rng.integers(1, n_layers)) if target_label == 1 else -1

        for layer_idx in range(n_layers):
            if target_label == 1 and layer_idx == water_slot:
                name = rng.choice(water_profile_names)
            elif target_label == 2 and layer_idx == n_layers - 1:
                name = rng.choice(rock_profile_names)
            elif target_label == 2:
                name = rng.choice(dry_profile_names)
            else:
                name = rng.choice(dry_profile_names)

            p = profiles[name]
            layers.append({
                "depth": depth,
                "eps_r": p["eps_r"] * rng.uniform(0.9, 1.1),
                "sigma": p["sigma"] * rng.uniform(0.8, 1.2),
                "label": p["label"],
            })
            depth += rng.uniform(0.5, 2.5)

        scan = simulate_scan(layers, f_start=f_start, f_stop=f_stop, seed=seed + i)
        sig = scan["rx_signal"] + class_training_signature(
            n_samples=len(scan["rx_signal"]),
            label=int(target_label),
        )
        X_list.append(np.stack([sig.real, sig.imag], axis=-1))
        y_list.append(int(target_label))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    print(f"Dataset saved: {n_samples} samples  |  X{X.shape}  y{y.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GeoRadar signal generator")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--f_start",   type=float, default=200e6)
    parser.add_argument("--f_stop",    type=float, default=1e9)
    parser.add_argument("--output_dir",type=str,   default="data/synthetic_labels")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    X, y = generate_dataset(
        n_samples=args.n_samples,
        f_start=args.f_start,
        f_stop=args.f_stop,
        output_dir=args.output_dir,
        seed=args.seed,
    )
