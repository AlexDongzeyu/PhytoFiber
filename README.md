# PhytoFiber

PhytoFiber is a reproducible analysis repository for a color-changing fiber spoilage project. The repo keeps raw records, standardized working images, analysis code, and generated figures separated so the full workflow can be rerun from source data.

## What Is Here

- `data/raw/`: active CSV inputs and raw photo archives.
- `images/raw/`: standardized calibration and spoilage images used by the CV pipeline.
- `data/processed/`: generated intermediate tables and model outputs.
- `visualizations/`: generated board-ready and advanced figures.
- `scripts/`: runnable pipeline entry points.
- `src/phytofiber_analysis/`: reusable analysis and plotting code.
- `docs/RAW_DATA_ENTRY_GUIDE.md`: detailed instructions for filling the raw CSV files.
- `docs/DELIVERABLES_CHECKLIST.md`: short checklist for what the repo can produce for reports or fair submission.
- `img/classification/`: scanned reference material and handout PDFs, kept as supporting records rather than machine-readable inputs.

## Main Workflow

Run the pipeline in this order:

```powershell
python scripts/01_cv_extraction.py
python scripts/02_biomechanics_anova.py
python scripts/03_predictive_models.py
python scripts/04_advanced_ml_augmentation.py
python scripts/build_figures.py
```

The numbered scripts and the `run_*` scripts are equivalent entry points. A root-level convenience wrapper is also available as `python 04_advanced_ml_augmentation.py`. `build_figures.py` now generates the core figures and, when the latency, stability, digestibility, and economics tables are present, also runs the advanced analysis layer.

## How The Files Link Together

1. CV extraction reads standardized images from `images/raw/` and writes color features into `data/processed/`.
2. Statistics reads `tensile_raw.csv`, computes tensile stress, then writes descriptives, assumption checks, ANOVA, Tukey, and effect sizes.
3. Predictive modeling joins spoilage measurements with extracted color data, fits the calibration and spoilage models, then writes labeled data and ML metrics.
4. Figure building reads the processed outputs and renders the final figures into `visualizations/`.

## Raw Inputs

Core inputs:

- `data/raw/tensile_raw.csv`
- `data/raw/calibration_raw.csv`
- `data/raw/spoilage_raw.csv`

Supporting inputs for the advanced layer:

- `data/raw/latency_data.csv`
- `data/raw/stability_data.csv`
- `data/raw/digestibility_data.csv`
- `data/raw/economics_data.csv`

Use `docs/RAW_DATA_ENTRY_GUIDE.md` for field-level guidance.

## Generated Outputs

Core processed outputs include:

- CV tables such as `color_data_final.csv` and `cv_extracted_spoilage.csv`
- biomechanics outputs such as `tensile_with_mpa.csv`, `assumption_checks.csv`, `anova_results.csv`, `tukey_results.csv`, and `effect_sizes.csv`
- ML outputs such as `spoilage_labeled.csv`, `classifier_predictions.csv`, `model_comparison.csv`, and calibration summaries

Core figures include:

- `tensile_strength_violin.png`
- `spoilage_regplot.png`
- `calibration_curve.png`
- `confusion_matrix_logistic.png`
- `roc_curve_logistic.png`
- `predictive_analysis_dashboard.png`

Advanced figures generated when the extra tables are present include:

- `weibull_probability_plot.png`
- `tensile_raincloud.png`
- `tensile_raincloud_monte_carlo.png`
- `bayesian_tensile_forest.png`
- `bayesian_tensile_superiority_heatmap.png`
- `calibration_4pl_curve.png`
- `spoilage_density_cloud.png`
- `spoilage_response_surface_3d.png`
- `latency_raincloud.png`
- `bayesian_latency_forest.png`
- `bayesian_latency_superiority_heatmap.png`
- `stability_timeseries.png`
- `digestibility_mass_loss.png`
- `economics_breakdown.png`
- `formulation_optimization_radar.png`

Generated outputs in `data/processed/` and `visualizations/` are reproducible artifacts. The repo is set up to keep most of those generated files out of git.

## Notes On Source Material

- `images/raw/` is the curated image set the pipeline is meant to work from.
- `data/raw/*_img/` folders are raw evidence archives and may contain extra reference shots.
- `img/classification/` PDFs are useful for documentation, but not as a substitute for structured CSV measurements.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
