# GeoRadar — Underground Water Detection
### Low-Power Green GPR for Drought Zone Sustainability
**Team Astrix · AESH 2026 · Green Radar Systems Track**

[![SDG 6](https://img.shields.io/badge/SDG%206-Clean%20Water-0a97d9)](https://sdgs.un.org/goals/goal6)
[![SDG 9](https://img.shields.io/badge/SDG%209-Innovation-fd6925)](https://sdgs.un.org/goals/goal9)
[![SDG 13](https://img.shields.io/badge/SDG%2013-Climate%20Action-3f7e44)](https://sdgs.un.org/goals/goal13)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

GeoRadar is a software-simulated Ground Penetrating Radar (GPR) system designed to detect subsurface water reservoirs in drought-affected regions of North Africa and the Middle East. It introduces two core innovations over conventional GPR:

- **Green Operation** — An entropy-based adaptive duty-cycling algorithm that activates the radar only when terrain conditions suggest subsurface anomalies, reducing power consumption by up to **70%** compared to always-on GPR systems.
- **AI-Powered Detection** — CNN training plus an operational physics-feature classifier for distinguishing water-bearing layers, dry soil, and rock in the synthetic GPR traces.
- **Soil Image Screening** — A photo-based estimator that accepts JPG/PNG soil images and scores visual indicators that may justify a follow-up GPR scan.

The full pipeline runs in simulation: soil image screening → FMCW signal generation → preprocessing → duty-cycle control → CNN classification → 3D interactive groundwater map.

---

## 2. Problem Statement

Over 2 billion people live in water-stressed regions. Traditional water prospecting (drilling, chemical surveys) is destructive, expensive, and environmentally harmful. Conventional radar systems are also power-hungry — transmitting continuously regardless of environmental conditions, wasting energy and contributing to electromagnetic pollution.

GeoRadar replaces destructive surveys with a non-invasive, low-power radar simulation framework that can be deployed computationally before any physical hardware is committed.

---

## 3. System Architecture

```
Soil Photo(s) + Soil Input Data (NASA/ESA dielectric datasets)
        │
        ▼
┌─────────────────────┐
│ Soil Image Screen   │  Pillow + NumPy — visual moisture cues become survey prior
│ (soil_image_detector.py) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FMCW Signal Gen    │  Python — 200 MHz–1 GHz sweep, configurable soil layers
│  (signal_gen.py)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │  SciPy — bandpass filter, noise removal, time-gain correction
│  (preprocess.py)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Duty-Cycle Gate    │  Entropy-based sensor — activates radar only on anomaly
│  (duty_cycle.py)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CNN Classifier     │  Keras/TensorFlow — classifies: water / dry soil / rock
│  (cnn_model.py)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3D Visualization   │  Plotly + Streamlit — interactive groundwater map dashboard
│  (dashboard.py)     │
└─────────────────────┘
```

---

## 4. Repository Structure

```
georadar/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE
│
├── docs/
│   ├── Astrix_GeoRadar_Phase1_ConceptDocument.pdf
│   ├── Astrix_GeoRadar_Phase1_Presentation.pdf
│   ├── architecture_diagram.png
│   └── assumptions.md                 # All simulation assumptions documented
│
├── soil_image_detector.py             # Soil photo screening and survey prior
├── signal_gen.py                      # FMCW GPR waveform simulation
├── preprocess.py                      # Bandpass filter + noise removal pipeline
├── duty_cycle.py                      # Adaptive entropy-based duty cycling
├── cnn_model.py                       # CNN architecture + training script
├── dashboard.py                       # Streamlit + Plotly 3D map dashboard
│
├── run_pipeline.py                    # End-to-end pipeline runner
├── soil_profiles.json                 # Configurable soil layer parameters
│
├── data/
│   ├── dielectric_profiles/           # NASA/ESA-derived soil dielectric datasets
│   └── synthetic_labels/              # Ground truth labels for CNN training
│
├── results/
│   ├── cnn_accuracy_report.txt        # Classification report — water/soil/rock
│   ├── soil_image_report.json         # Image screening output
│   ├── power_comparison.csv           # GeoRadar vs conventional power budget
│   ├── power_comparison_chart.png
│   └── sample_groundwater_map.html    # Interactive Plotly output (open in browser)
│
└── assets/
    ├── demo_thumbnail.png
    └── pipeline_flow.png
```

---

## 5. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python run_pipeline.py
```

### Run the full pipeline with soil images
```bash
python run_pipeline.py --soil_images C:\Users\mahmo\Desktop\soil.jpg
python run_pipeline.py --soil_images C:\Users\mahmo\Desktop\soil_images
```

### Launch the dashboard
```bash
streamlit run dashboard.py
```

The dashboard is the main one-line interface. It includes soil image upload, the image brief, the image-informed survey prior, 3D subsurface classification, CNN confidence heatmap, power comparison, and saved classifier accuracy metrics.

### Analyze soil images
```bash
python soil_image_detector.py path/to/soil_photo.jpg
python soil_image_detector.py path/to/image_folder --output results/soil_image_report.json
```

The image score is a screening estimate only. Surface photos can show moisture-related cues such as darker soil, vegetation, and texture variation, but they cannot directly prove underground water. Use high-scoring locations to prioritize GPR, resistivity, or field validation.

### Train the CNN classifier
```bash
python cnn_model.py --train --epochs 50 --data_dir data/
```

### Run individual notebooks
Open `simulation/notebooks/` in Jupyter. Run notebooks in order (01 → 04).

---

## 6. Key Results

| Metric | Value |
|---|---|
| CNN classification accuracy | >85% (water / dry soil / rock) |
| Power reduction vs conventional GPR | ~70% |
| Simulated subsurface penetration depth | Up to 10 m |
| Operating frequency range | 200 MHz – 1 GHz |
| Duty-cycle activation rate (arid terrain) | ~30% of scan time |

See `results/` for full classification report, power comparison CSV, and sample groundwater map.

---

## 7. Assumptions and Limitations

Full assumptions are documented in `docs/assumptions.md`. Key points:

- Dielectric constant profiles are derived from open NASA/ESA datasets — not field-measured values.
- CNN training data is synthetically generated from simulation; real-world validation would require field deployment.
- Soil image analysis is heuristic unless retrained on labelled field images from the target region.
- Power consumption comparison is modeled, not measured on physical hardware.
- Duty-cycle savings of 70% assume arid terrain with low ambient anomaly density; savings vary by environment.
- The 3D groundwater map is a visualization of simulated outputs, not real survey data.

---

## 8. AI Usage Disclosure

Portions of boilerplate code structure and documentation were drafted with AI assistance (GitHub Copilot, Claude). All generated outputs were reviewed, validated, and modified by team members. The CNN architecture, signal simulation logic, duty-cycle algorithm, and all numerical results were developed and verified by the team.

---

## 9. Contact

**Team Astrix** · AESH Sustainability Hackathon 2026  
Challenge Track: Green Radar Systems  
