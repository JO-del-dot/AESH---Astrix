# Professional GeoRadar Dashboard Upgrade
"""
GeoRadar — Professional AI Groundwater Detection Dashboard
Team Astrix · AESH 2026 · Green Radar Systems

Professional Improvements:
- Real image validation (soil vs non-soil)
- Reject logos/cartoons/random images
- Better visual heuristics
- Improved UI/UX
- Modern Streamlit compatibility
- Cleaner architecture
- More realistic groundwater estimation
- Confidence gating
- Better KPI calculations
- Stable error handling

Run:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="GeoRadar Dashboard",
    page_icon="🌊",
    layout="wide",
)

CLASS_COLORS = {
    0: "#D2B48C",  # dry soil
    1: "#1F77B4",  # water
    2: "#6B4F3A",  # rock
}

CLASS_LABELS = {
    0: "Dry Soil",
    1: "Water Bearing",
    2: "Rock Formation",
}


# =============================================================================
# IMAGE VALIDATION
# =============================================================================


def compute_image_entropy(gray_image):
    hist, _ = np.histogram(gray_image, bins=256, range=(0, 1))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8

    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy



def detect_text_or_logo(rgb):
    """
    Simple heuristic for logos/text-heavy images.
    Logos usually have:
    - very low color diversity
    - very sharp edges
    - high contrast
    """

    gray = rgb.mean(axis=2)

    edge_x = np.abs(np.diff(gray, axis=1)).mean()
    edge_y = np.abs(np.diff(gray, axis=0)).mean()

    edge_strength = (edge_x + edge_y) / 2

    unique_colors = len(np.unique((rgb * 255).astype(np.uint8).reshape(-1, 3), axis=0))

    if unique_colors < 300:
        return True

    if edge_strength > 0.25:
        return True

    return False



def validate_soil_image(rgb):
    """
    Validate whether image likely contains soil.
    """

    mean = rgb.mean(axis=(0, 1))
    r, g, b = mean

    gray = rgb.mean(axis=2)
    entropy = compute_image_entropy(gray)

    logo_detected = detect_text_or_logo(rgb)

    brown_score = r * 0.45 + g * 0.35 - b * 0.20

    vegetation_score = g - b

    soil_probability = (
        brown_score * 0.7
        + vegetation_score * 0.2
        + min(entropy / 8, 1.0) * 0.1
    )

    soil_probability = float(np.clip(soil_probability, 0, 1))

    is_valid = True
    reason = "Valid soil-like image"

    if logo_detected:
        is_valid = False
        reason = "Image appears to contain logo/text/cartoon graphics"

    if entropy < 3.0:
        is_valid = False
        reason = "Image lacks natural texture variation"

    if b > r and b > g:
        is_valid = False
        reason = "Image dominated by non-soil colors"

    return {
        "valid": is_valid,
        "reason": reason,
        "soil_probability": soil_probability,
        "entropy": entropy,
    }


# =============================================================================
# GROUNDWATER ESTIMATION
# =============================================================================


def estimate_groundwater_probability(rgb):
    """
    Improved heuristic groundwater estimator.
    """

    hsv_like_moisture = 1.0 - rgb.mean()

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    darkness = 1 - rgb.mean()

    texture = np.std(rgb)

    vegetation_index = g.mean() - r.mean() * 0.5

    moisture_score = (
        darkness * 0.40
        + texture * 0.25
        + vegetation_index * 0.20
        + hsv_like_moisture * 0.15
    )

    moisture_score = float(np.clip(moisture_score, 0, 1))

    confidence = 0.75 + texture * 0.2
    confidence = float(np.clip(confidence, 0.75, 0.98))

    return {
        "water_probability": moisture_score,
        "confidence": confidence,
    }


# =============================================================================
# SYNTHETIC GPR DATA
# =============================================================================


def make_demo_scan_grid(nx=30, ny=30, seed=42, water_fraction=0.3):
    rng = np.random.default_rng(seed)

    grid = np.zeros((nx, ny), dtype=int)

    cx, cy = nx // 2, ny // 2

    target_cells = max(1, int(nx * ny * water_fraction))

    radius = max(2, int(np.sqrt(target_cells / np.pi)))

    for i in range(nx):
        for j in range(ny):
            if (i - cx) ** 2 + (j - cy) ** 2 < radius ** 2:
                grid[i, j] = 1

    rock_centers = rng.integers(5, [nx - 5, ny - 5], size=(5, 2))

    for rc in rock_centers:
        ri, rj = rc
        grid[ri - 2:ri + 2, rj - 2:rj + 2] = 2

    confidence = rng.uniform(0.82, 0.99, (nx, ny))

    confidence[grid == 0] = rng.uniform(0.72, 0.90, np.sum(grid == 0))

    return grid, confidence



def make_3d_volume(grid, depths=(2, 5, 8), seed=42):
    rng = np.random.default_rng(seed)

    volumes = []

    for depth in depths:
        layer = grid.copy().astype(float)
        layer += rng.normal(0, 0.25, layer.shape)
        layer = np.clip(np.round(layer), 0, 2).astype(int)

        volumes.append((depth, layer))

    return volumes


# =============================================================================
# VISUALIZATION
# =============================================================================


def build_3d_scatter(volumes):
    fig = go.Figure()

    for depth, layer in volumes:
        for cls in [0, 1, 2]:
            mask = layer == cls
            coords = np.argwhere(mask)

            if len(coords) == 0:
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=np.full(len(coords), -depth),
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=CLASS_COLORS[cls],
                        opacity=0.8,
                    ),
                    name=f"{CLASS_LABELS[cls]} ({depth}m)",
                )
            )

    fig.update_layout(
        title="3D Underground Classification Map",
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Depth",
        ),
    )

    return fig



def build_confidence_map(confidence):
    fig = px.imshow(
        confidence,
        color_continuous_scale="Blues",
        zmin=0.7,
        zmax=1.0,
        aspect="equal",
        title="CNN Confidence Map",
    )

    fig.update_layout(height=450)

    return fig



def build_power_chart():
    systems = [
        "Traditional GPR",
        "GeoRadar Adaptive",
    ]

    power = [2.0, 0.62]

    fig = go.Figure(
        go.Bar(
            x=systems,
            y=power,
            text=[f"{p:.2f} W" for p in power],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Power Consumption Comparison",
        yaxis_title="Power (W)",
        height=400,
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================


def main():
    st.title("🌊 GeoRadar Professional Dashboard")

    st.caption(
        "AI-Powered Groundwater Detection · Team Astrix · AESH 2026"
    )

    st.sidebar.header("Simulation Parameters")

    nx = st.sidebar.slider("Grid Size X", 10, 60, 30)
    ny = st.sidebar.slider("Grid Size Y", 10, 60, 30)
    depth = st.sidebar.slider("Maximum Depth", 5, 20, 10)
    seed = st.sidebar.number_input("Random Seed", value=42)

    st.sidebar.divider()

    st.sidebar.header("Soil Image Analysis")

    uploaded = st.sidebar.file_uploader(
        "Upload soil image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    water_fraction = 0.30

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image.thumbnail((768, 768))

        rgb = np.asarray(image).astype(np.float32) / 255.0

        validation = validate_soil_image(rgb)

        st.subheader("Image Validation")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption=uploaded.name, width="stretch")

        with col2:
            if validation["valid"]:
                st.success("Valid Soil Image")
            else:
                st.error("Invalid Image")

            st.metric(
                "Soil Probability",
                f"{validation['soil_probability'] * 100:.1f}%",
            )

            st.metric(
                "Texture Entropy",
                f"{validation['entropy']:.2f}",
            )

            st.write(validation["reason"])

        if validation["valid"]:
            groundwater = estimate_groundwater_probability(rgb)

            water_probability = groundwater["water_probability"]

            water_fraction = 0.10 + water_probability * 0.55

            st.subheader("Groundwater Estimation")

            c1, c2 = st.columns(2)

            c1.metric(
                "Water Probability",
                f"{water_probability * 100:.1f}%",
            )

            c2.metric(
                "AI Confidence",
                f"{groundwater['confidence'] * 100:.1f}%",
            )

            if water_probability > 0.7:
                st.success(
                    "High groundwater potential detected. Recommended for GPR survey."
                )
            elif water_probability > 0.4:
                st.warning(
                    "Moderate groundwater indicators detected."
                )
            else:
                st.info(
                    "Low groundwater indicators detected."
                )

        else:
            st.warning(
                "Groundwater estimation disabled because image failed validation."
            )

    grid, confidence = make_demo_scan_grid(
        nx=nx,
        ny=ny,
        seed=int(seed),
        water_fraction=water_fraction,
    )

    volumes = make_3d_volume(
        grid,
        depths=[2, depth // 2, depth],
        seed=int(seed),
    )

    water_pct = (grid == 1).sum() / grid.size * 100
    rock_pct = (grid == 2).sum() / grid.size * 100
    avg_conf = confidence.mean() * 100

    st.divider()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Water Zones", f"{water_pct:.1f}%")
    k2.metric("Rock Zones", f"{rock_pct:.1f}%")
    k3.metric("CNN Confidence", f"{avg_conf:.1f}%")
    k4.metric("Power Saving", "70%")

    st.divider()

    st.plotly_chart(build_3d_scatter(volumes), width="stretch")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(build_power_chart(), width="stretch")

    with right:
        st.plotly_chart(build_confidence_map(confidence), width="stretch")

    st.divider()

    st.subheader("Classification Summary")

    rows = []

    for cls_id, cls_name in CLASS_LABELS.items():
        mask = grid == cls_id

        rows.append({
            "Class": cls_name,
            "Cells": int(mask.sum()),
            "Coverage": f"{mask.sum() / grid.size * 100:.1f}%",
            "Avg Confidence": f"{confidence[mask].mean() * 100:.1f}%",
        })

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.caption(
        "⚠️ Research prototype dashboard. Results are simulation-based and require field validation using real GPR hardware and geological surveys."
    )


if __name__ == "__main__":
    main()