"""
GeoRadar - Soil Image Water-Potential Estimator
Team Astrix · AESH 2026 · Green Radar Systems

Estimates whether a surface soil image shows visual indicators that can be
associated with possible subsurface moisture or groundwater.

Important:
  This is an image-based screening heuristic, not a physical groundwater
  detector. True underground water confirmation still requires GPR, resistivity,
  drilling logs, or field measurements.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image


SOIL_CONFIDENCE_THRESHOLD = 0.32
MIN_SOIL_EDGE_DENSITY = 0.018
MAX_SOIL_EDGE_DENSITY = 0.55


@dataclass
class SoilImageResult:
    image_path: str
    is_soil: bool
    soil_confidence: float
    water_probability: float
    risk_level: str
    surface_condition: str
    analysis_brief: str
    recommendation: str
    features: Dict[str, float]


def _load_rgb(image_path: str, max_size: int = 768) -> np.ndarray:
    """Load an image as RGB float array in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported image format for {image_path}")
    return arr


def _normalized_channel_features(rgb: np.ndarray) -> Dict[str, float]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=2)

    # Soil that is visibly moist is often darker and slightly less red/yellow.
    dark_pixel_ratio = float((brightness < 0.34).mean())
    mean_brightness = float(brightness.mean())

    # Green vegetation near soil can indicate shallow moisture, but it is only
    # supportive evidence and should not dominate the decision.
    green_excess = 2.0 * g - r - b
    vegetation_ratio = float((green_excess > 0.12).mean())

    # Very bright, red/yellow, low-green soil is often drier at the surface.
    dry_bare_ratio = float(((brightness > 0.58) & (r > g * 1.05) & (g > b * 0.95)).mean())

    # Surface texture and color variation can indicate mixed sediments, cracks,
    # organic matter, or damp patches. We use Shannon entropy on brightness.
    hist, _ = np.histogram(brightness, bins=32, range=(0.0, 1.0))
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)) / 5.0)  # normalize by log2(32)

    # Estimate local contrast with simple finite differences.
    grad_x = np.abs(np.diff(brightness, axis=1)) if brightness.shape[1] > 1 else np.zeros_like(brightness)
    grad_y = np.abs(np.diff(brightness, axis=0)) if brightness.shape[0] > 1 else np.zeros_like(brightness)
    dx = grad_x.mean() if grad_x.size else 0.0
    dy = grad_y.mean() if grad_y.size else 0.0
    texture_contrast = float(np.clip((dx + dy) * 4.0, 0.0, 1.0))

    # Soil should have visible granular texture: too few edges is often sky,
    # walls, paper, or blurred images; too many can be vegetation/clutter.
    edge_x = grad_x > 0.075 if grad_x.size else np.zeros((brightness.shape[0], 0), dtype=bool)
    edge_y = grad_y > 0.075 if grad_y.size else np.zeros((0, brightness.shape[1]), dtype=bool)
    edge_density = float(
        (edge_x.sum() + edge_y.sum()) / max(edge_x.size + edge_y.size, 1)
    )

    # Blue/gray shadows or wet-looking patches can increase the moisture cue,
    # but this can also be lighting, so it carries moderate weight.
    cool_dark_ratio = float(((brightness < 0.42) & (b >= r * 0.9)).mean())
    channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    low_saturation_ratio = float((channel_spread < 0.08).mean())
    red_yellow_ratio = float(((r > b * 1.18) & (g > b * 1.05)).mean())

    return {
        "mean_brightness": mean_brightness,
        "dark_pixel_ratio": dark_pixel_ratio,
        "vegetation_ratio": vegetation_ratio,
        "dry_bare_soil_ratio": dry_bare_ratio,
        "brightness_entropy": entropy,
        "texture_contrast": texture_contrast,
        "edge_density": edge_density,
        "cool_dark_patch_ratio": cool_dark_ratio,
        "low_saturation_ratio": low_saturation_ratio,
        "red_yellow_soil_ratio": red_yellow_ratio,
    }


def estimate_water_probability(features: Dict[str, float]) -> float:
    """
    Convert visual soil cues into a calibrated screening probability.

    The weights are conservative because surface photos cannot directly image
    underground water. A high score means "worth scanning with GeoRadar/GPR",
    not "water is confirmed."
    """
    wet_score = (
        0.28 * features["dark_pixel_ratio"]
        + 0.18 * features["vegetation_ratio"]
        + 0.16 * features["cool_dark_patch_ratio"]
        + 0.14 * features["brightness_entropy"]
        + 0.10 * features["texture_contrast"]
    )
    dry_penalty = (
        0.20 * features["dry_bare_soil_ratio"]
        + 0.10 * max(features["mean_brightness"] - 0.55, 0.0) / 0.45
    )

    # Keep the image-only estimate modest unless several cues agree.
    probability = 0.18 + wet_score - dry_penalty
    return float(np.clip(probability, 0.03, 0.92))


def estimate_soil_confidence(features: Dict[str, float]) -> float:
    """
    Estimate whether the image is likely soil/ground.

    Soil photos usually contain earthy red/yellow/brown tones, moderate to low
    saturation, visible texture, and limited pure vegetation/sky/water area.
    This is a validation gate before water-potential scoring.
    """
    earthy_score = (
        0.34 * features["red_yellow_soil_ratio"]
        + 0.24 * features["low_saturation_ratio"]
        + 0.18 * features["brightness_entropy"]
        + 0.08 * features["texture_contrast"]
        + 0.04 * min(features["edge_density"] / MIN_SOIL_EDGE_DENSITY, 1.0)
        + 0.12 * features["dark_pixel_ratio"]
    )
    non_soil_penalty = (
        0.34 * max(features["vegetation_ratio"] - 0.35, 0.0) / 0.65
        + 0.20 * max(features["cool_dark_patch_ratio"] - 0.65, 0.0) / 0.35
    )
    return float(np.clip(earthy_score - non_soil_penalty, 0.0, 1.0))


def classify_probability(probability: float) -> str:
    if probability >= 0.62:
        return "high"
    if probability >= 0.38:
        return "moderate"
    return "low"


def classify_soil_validity(features: Dict[str, float], soil_confidence: float) -> bool:
    edge_density = features["edge_density"]
    return (
        soil_confidence >= SOIL_CONFIDENCE_THRESHOLD
        and MIN_SOIL_EDGE_DENSITY <= edge_density <= MAX_SOIL_EDGE_DENSITY
    )


def describe_surface_condition(features: Dict[str, float], probability: float) -> str:
    if probability >= 0.62 and features["dark_pixel_ratio"] > 0.35:
        return "wet-looking or strongly moisture-indicative surface"
    if probability >= 0.38:
        return "possibly moist or mixed soil surface"
    if features["dry_bare_soil_ratio"] > 0.35 or features["mean_brightness"] > 0.56:
        return "dry, bright bare-soil surface"
    return "mostly dry or inconclusive surface"


def build_analysis_brief(features: Dict[str, float], probability: float) -> str:
    """Create a concise natural-language description of what the image suggests."""
    condition = describe_surface_condition(features, probability)
    cues: List[str] = []

    if features["dark_pixel_ratio"] > 0.45:
        cues.append("large dark soil regions")
    elif features["mean_brightness"] > 0.56:
        cues.append("bright exposed soil")

    if features["vegetation_ratio"] > 0.08:
        cues.append("visible green vegetation")
    if features["cool_dark_patch_ratio"] > 0.18:
        cues.append("cool dark patches that may be damp or shaded")
    if features["dry_bare_soil_ratio"] > 0.25:
        cues.append("dry bare-soil color cues")
    if features["brightness_entropy"] > 0.42 or features["texture_contrast"] > 0.18:
        cues.append("mixed texture and color variation")
    if features["edge_density"] > MIN_SOIL_EDGE_DENSITY:
        cues.append("soil-like granular edges")

    if not cues:
        cues.append("limited strong moisture cues")

    cue_text = ", ".join(cues[:4])
    return (
        f"The image appears to show a {condition}. Key visual cues: {cue_text}. "
        "Treat this as surface evidence only; confirm with the GPR/CNN scan outputs."
    )


def build_not_soil_brief(features: Dict[str, float], soil_confidence: float) -> str:
    cues: List[str] = []
    if features["vegetation_ratio"] > 0.35:
        cues.append("too much green vegetation")
    if features["red_yellow_soil_ratio"] < 0.12:
        cues.append("few earthy soil-colored pixels")
    if features["low_saturation_ratio"] < 0.12:
        cues.append("high color saturation")
    if features["brightness_entropy"] < 0.08 and features["texture_contrast"] < 0.04:
        cues.append("too little soil-like texture")
    if features["edge_density"] < MIN_SOIL_EDGE_DENSITY:
        cues.append("edge density below the soil threshold")
    if features["edge_density"] > MAX_SOIL_EDGE_DENSITY:
        cues.append("edge density too high for a clean soil photo")
    if features["cool_dark_patch_ratio"] > 0.65:
        cues.append("large cool/dark areas that may be water, shadow, or non-soil")
    if not cues:
        cues.append("weak soil evidence")
    return (
        f"This image does not look enough like a soil/ground photo "
        f"(soil confidence {soil_confidence * 100:.1f}%). "
        f"Main reason: {', '.join(cues[:3])}. Upload a close, clear image of exposed soil."
    )


def recommendation_for(level: str) -> str:
    if level == "high":
        return "Prioritize this location for GPR scanning or a resistivity survey."
    if level == "moderate":
        return "Collect more images and run a GeoRadar/GPR scan before selecting a drilling point."
    return "Surface image evidence is weak; scan only if other field indicators support it."


def analyze_soil_image(image_path: str) -> SoilImageResult:
    rgb = _load_rgb(image_path)
    return analyze_soil_array(rgb, image_path=str(image_path))


def analyze_soil_array(rgb: np.ndarray, image_path: str = "uploaded_image") -> SoilImageResult:
    """Analyze an already-loaded RGB image array."""
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32) / 255.0
    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    features = _normalized_channel_features(rgb)
    soil_confidence = estimate_soil_confidence(features)
    is_soil = classify_soil_validity(features, soil_confidence)
    probability = estimate_water_probability(features)
    level = classify_probability(probability)
    surface_condition = describe_surface_condition(features, probability)
    analysis_brief = (
        build_analysis_brief(features, probability)
        if is_soil
        else build_not_soil_brief(features, soil_confidence)
    )
    return SoilImageResult(
        image_path=image_path,
        is_soil=is_soil,
        soil_confidence=soil_confidence,
        water_probability=probability,
        risk_level=level if is_soil else "invalid",
        surface_condition=surface_condition,
        analysis_brief=analysis_brief,
        recommendation=recommendation_for(level) if is_soil else "Upload a valid soil image before estimating groundwater potential.",
        features=features,
    )


def analyze_many(image_paths: Iterable[str]) -> List[SoilImageResult]:
    return [analyze_soil_image(path) for path in image_paths]


def save_report(results: List[SoilImageResult], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)


def print_result(result: SoilImageResult) -> None:
    print("=" * 64)
    print(f"Image: {result.image_path}")
    print(f"Soil image valid: {'YES' if result.is_soil else 'NO'} ({result.soil_confidence * 100:.1f}% confidence)")
    if not result.is_soil:
        print(f"Brief: {result.analysis_brief}")
        print(f"Recommendation: {result.recommendation}")
        return
    print(f"Possible underground-water indicator: {result.water_probability * 100:.1f}%")
    print(f"Level: {result.risk_level.upper()}")
    print(f"Surface condition: {result.surface_condition}")
    print(f"Brief: {result.analysis_brief}")
    print(f"Recommendation: {result.recommendation}")
    print("Features:")
    for name, value in result.features.items():
        print(f"  {name:24s}: {value:.3f}")


def expand_image_inputs(paths: List[str]) -> List[str]:
    image_paths: List[str] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"[WARNING] Skipping missing path: {raw_path}")
            continue
        if path.is_dir():
            image_paths.extend(
                str(child) for child in sorted(path.iterdir())
                if child.suffix.lower() in extensions
            )
        else:
            image_paths.append(str(path))
    return image_paths


def average_water_probability(results: List[SoilImageResult]) -> float:
    valid_results = [result for result in results if result.is_soil]
    if not valid_results:
        return 0.0
    return float(np.mean([result.water_probability for result in valid_results]))


def image_probability_to_water_fraction(probability: float) -> float:
    """
    Convert image-screening probability into a conservative simulated survey
    water-zone fraction for the rest of the GeoRadar pipeline.
    """
    return float(np.clip(0.10 + probability * 0.45, 0.10, 0.55))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate possible underground-water indicators from soil images."
    )
    parser.add_argument("images", nargs="+", help="Image file(s) or folder(s) to analyze")
    parser.add_argument(
        "--output",
        default="results/soil_image_report.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    image_paths = expand_image_inputs(args.images)
    if not image_paths:
        raise SystemExit(
            "No supported images found. Use a real JPG/PNG path, for example:\n"
            "  python soil_image_detector.py C:\\Users\\mahmo\\Desktop\\soil.jpg\n"
            "or place images in a folder and pass that folder path."
        )

    results = analyze_many(image_paths)
    for result in results:
        print_result(result)
    save_report(results, args.output)
    print("=" * 64)
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
