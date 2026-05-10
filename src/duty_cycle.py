"""
GeoRadar — Adaptive Duty-Cycle Controller
Team Astrix · AESH 2026 · Green Radar Systems

Green Operation core:
  Activates the radar only when entropy-based environmental sensing
  detects subsurface anomaly conditions — achieving ~70% reduction in
  active transmission time vs. always-on GPR systems.

Algorithm overview:
  1. A low-power environmental sensor (soil moisture proxy, surface
     permittivity estimate) computes a Shannon entropy score for the
     current terrain window.
  2. If entropy exceeds a threshold, subsurface heterogeneity is likely
     → radar activates for one sweep.
  3. Otherwise, radar stays off → significant power saved.
  4. Power metrics are logged for comparison against conventional GPR.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Power model constants (estimated from literature) ─────────────────────────
RADAR_TX_POWER_W   = 0.5    # active transmit power (W) — low-power GPR module
RADAR_RX_POWER_W   = 0.1    # active receive power (W)
RADAR_IDLE_POWER_W = 0.02   # idle/standby power (W)
SENSOR_POWER_W     = 0.005  # entropy sensor power (W) — always on

CONVENTIONAL_GPR_W = 2.0    # always-on conventional GPR reference (W)


# ── Entropy sensor ─────────────────────────────────────────────────────────────

def shannon_entropy(signal: np.ndarray, n_bins: int = 32) -> float:
    """
    Estimate Shannon entropy of a signal as a proxy for terrain heterogeneity.

    High entropy → complex subsurface structure → activate radar.
    Low entropy  → homogeneous medium → skip sweep, save power.
    """
    counts, _ = np.histogram(signal, bins=n_bins, density=False)
    probs = counts / counts.sum()
    probs = probs[probs > 0]       # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def surface_entropy_signal(n_positions: int, seed: int = 42,
                            water_fraction: float = 0.30) -> np.ndarray:
    """
    Simulate a sequence of surface entropy readings along a survey transect.
    Positions over water-bearing terrain have higher entropy.
    """
    rng = np.random.default_rng(seed)
    entropy_seq = rng.normal(2.5, 0.4, n_positions)    # baseline arid terrain

    # Inject water-zone anomaly patches (higher entropy)
    n_water = int(n_positions * water_fraction)
    water_idx = rng.choice(n_positions, n_water, replace=False)
    entropy_seq[water_idx] += rng.uniform(1.0, 2.5, n_water)

    return np.clip(entropy_seq, 0, None)


# ── Duty-cycle gate ────────────────────────────────────────────────────────────

@dataclass
class DutyCycleLog:
    n_positions:       int   = 0
    n_activated:       int   = 0
    n_skipped:         int   = 0
    georadar_energy_J: float = 0.0
    conventional_energy_J: float = 0.0
    activation_map:    List[bool] = field(default_factory=list)


def run_duty_cycle(
    entropy_sequence: np.ndarray,
    threshold: float = 3.5,
    sweep_duration_s: float = 1e-6,
    position_interval_s: float = 0.1,    # time between position steps
) -> Tuple[np.ndarray, DutyCycleLog]:
    """
    Simulate the adaptive duty-cycle gate over a survey transect.

    Args:
        entropy_sequence:  1-D array of surface entropy values per position
        threshold:         Entropy threshold above which radar activates
        sweep_duration_s:  Duration of one FMCW radar sweep (s)
        position_interval_s: Time step between sensor readings (s)

    Returns:
        activation_flags: bool array — True where radar fired
        log:              DutyCycleLog with energy and activation stats
    """
    log = DutyCycleLog(n_positions=len(entropy_sequence))
    activation_flags = np.zeros(len(entropy_sequence), dtype=bool)

    for i, h in enumerate(entropy_sequence):
        # Sensor always running
        sensor_energy = SENSOR_POWER_W * position_interval_s

        if h >= threshold:
            # Activate radar for one sweep
            radar_energy = (RADAR_TX_POWER_W + RADAR_RX_POWER_W) * sweep_duration_s
            log.n_activated  += 1
            activation_flags[i] = True
        else:
            # Radar stays in idle
            radar_energy = RADAR_IDLE_POWER_W * position_interval_s
            log.n_skipped += 1

        log.georadar_energy_J += sensor_energy + radar_energy
        log.conventional_energy_J += CONVENTIONAL_GPR_W * position_interval_s

    log.activation_map = activation_flags.tolist()
    return activation_flags, log


def power_reduction_percent(log: DutyCycleLog) -> float:
    """Calculate percentage power saving vs conventional always-on GPR."""
    return (1 - log.georadar_energy_J / log.conventional_energy_J) * 100


def print_power_report(log: DutyCycleLog) -> None:
    reduction = power_reduction_percent(log)
    duty_ratio = log.n_activated / log.n_positions * 100
    print("=" * 52)
    print("  GeoRadar Power Report")
    print("=" * 52)
    print(f"  Total positions surveyed : {log.n_positions}")
    print(f"  Radar activations        : {log.n_activated}  ({duty_ratio:.1f}% duty cycle)")
    print(f"  Radar skips (saved)      : {log.n_skipped}")
    print(f"  GeoRadar energy          : {log.georadar_energy_J*1000:.3f} mJ")
    print(f"  Conventional GPR energy  : {log.conventional_energy_J*1000:.3f} mJ")
    print(f"  Power reduction          : {reduction:.1f}%")
    print("=" * 52)


if __name__ == "__main__":
    n_pos = 1000
    entropy_seq = surface_entropy_signal(n_positions=n_pos, water_fraction=0.30)
    flags, log = run_duty_cycle(entropy_seq, threshold=3.5)
    print_power_report(log)

    # Save results
    import csv, os
    os.makedirs("../results", exist_ok=True)
    with open("../results/power_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "energy_mJ", "duty_cycle_pct", "power_reduction_pct"])
        writer.writerow([
            "GeoRadar (adaptive)",
            f"{log.georadar_energy_J*1000:.3f}",
            f"{log.n_activated/log.n_positions*100:.1f}",
            f"{power_reduction_percent(log):.1f}",
        ])
        writer.writerow([
            "Conventional GPR (always-on)",
            f"{log.conventional_energy_J*1000:.3f}",
            "100.0",
            "0.0",
        ])
    print("Saved: results/power_comparison.csv")
