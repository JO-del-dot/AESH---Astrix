# GeoRadar — Presentation Deck Outline
**Team Astrix · AESH 2026 · 10 slides**

File name: `Astrix_GeoRadar_Phase1_Presentation.pdf`

---

| Slide | Title | Key content |
|---|---|---|
| 1 | GeoRadar — Team Astrix | Project title · Team name · Challenge track · SDG 6 / 9 / 13 badges |
| 2 | The Problem | 2B people in water-stressed regions · Destructive drilling · Power-hungry always-on GPR · Visual: map of arid zones in North Africa / Middle East |
| 3 | Baseline: Conventional GPR | Always-on 2 W consumption · No terrain sensing · Spectrum congestion · Visual: always-on vs adaptive activation diagram |
| 4 | Our Solution: GeoRadar | Two innovations: (1) adaptive duty-cycling, (2) CNN classifier · One-line value prop per innovation |
| 5 | System Architecture | Block diagram: Signal Gen → Preprocess → Duty-Cycle Gate → CNN → 3D Map Dashboard · Tools per block |
| 6 | Green Operation — Duty-Cycle Algorithm | Entropy sensor logic · Flow: entropy > threshold → radar ON, else IDLE · Power model table |
| 7 | AI Detection — CNN Classifier | Architecture diagram (Conv1D blocks) · Input: envelope · Output: water / dry soil / rock · Training setup |
| 8 | Results | Power comparison bar chart (2.0 W vs 0.6 W → 70% reduction) · CNN accuracy (>85%) · Screenshot of 3D dashboard |
| 9 | Limitations & Honest Assessment | Synthetic data only · No hardware measurement · Dielectric values from literature · Field validation needed |
| 10 | Impact, Repository & Next Steps | SDG alignment · GitHub link · 3 next steps: (1) real soil datasets, (2) hardware prototype, (3) field trial in arid region |

---

## Design notes

- Slides 1–2: use one strong image (arid landscape / cracked earth) to anchor the problem emotionally.
- Slide 5: the architecture diagram is the most important technical visual — keep it clean and label every arrow.
- Slide 8: results slide should have two charts side-by-side: power bar chart (left) + CNN accuracy summary (right).
- Slide 9: honesty on limitations is a scoring criterion — don't hide assumptions.
- Slide 10: show the GitHub folder tree screenshot to prove the repository is real and organized.

## Scoring alignment

| Criterion (points) | Slide |
|---|---|
| Technical Innovation (25) | 4, 6, 7 |
| Technical Feasibility (20) | 5, 6, 7, 8 |
| Sustainability Impact (20) | 2, 3, 10 |
| Implementation Evidence (15) | 8, 9 |
| Documentation Quality (10) | 9, 10 |
| Presentation Quality (10) | All |
