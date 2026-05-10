# GeoRadar — Demo Video Script
**Team Astrix · AESH 2026 · Target duration: 2–3 minutes**

---

## Segment breakdown

| Time | What to show on screen | Script |
|---|---|---|
| 0:00 – 0:20 | Title slide: "GeoRadar — Underground Water Detection" + team name + SDG badges | "We are Team Astrix. GeoRadar is a software-simulated Ground Penetrating Radar system that locates hidden groundwater in drought-affected regions — using 70% less power than conventional always-on GPR." |
| 0:20 – 0:45 | Show the pipeline architecture diagram | "Our system starts with a soil photo screening step. The image score becomes a survey prior, then the simulator generates FMCW GPR scans, preprocesses them, and uses the adaptive duty-cycle gate to activate radar only where anomaly evidence is stronger." |
| 0:45 – 1:15 | Live terminal: run `python run_pipeline.py --soil_images results/test_soil_image.png --epochs 3` | "Here we run the full pipeline with a soil image. The image module scores visual moisture cues, writes a JSON report, and passes that probability into the duty-cycle simulation before the GPR and CNN classification stage." |
| 1:15 – 1:45 | Show power_comparison.csv / power comparison bar chart | "The power comparison tells the core story. Conventional GPR draws 2 watts continuously. GeoRadar activates for roughly 30% of positions, bringing average consumption down to about 0.6 watts — a 70% reduction." |
| 1:45 – 2:15 | Launch Streamlit dashboard: `streamlit run dashboard.py` — upload a soil image, then show 3D map rotating | "The dashboard ties the same image score to the survey map. Uploading a soil photo updates the image-informed survey prior and changes the simulated water-bearing coverage before the 3D scan view is rendered." |
| 2:15 – 2:45 | Show image screening panel, CNN confidence heatmap, and classification report | "The image panel is not claiming proof of underground water. It ranks where to scan first. The CNN confidence map then shows the simulated GPR classifier's water, dry soil, and rock predictions." |
| 2:45 – 3:00 | Closing slide with GitHub link + SDG alignment | "All code, notebooks, datasets, and results are in our public repository. We're honest that these are simulated results — but the pipeline is complete, reproducible, and directly applicable to arid regions across North Africa and the Middle East." |

---

## Recording tips

- Run `python run_pipeline.py --soil_images results/test_soil_image.png --epochs 3` live if TensorFlow is installed. Use `--skip_train` only for a fast backup demo.
- Launch the dashboard before recording and have it ready to switch to at the 1:45 mark.
- Rotate the 3D map slowly with the mouse during the dashboard segment — motion makes it readable.
- Keep the `results/power_comparison.csv` and `results/cnn_accuracy_report.txt` open in a text editor as a fallback visual.
- Add subtitles if recording in a noisy environment.

## File name

```
Astrix_GeoRadar_Phase1_Video.mp4
```
