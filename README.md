# PhytoFiber Advanced Data Pipeline

This repository is a reproducible analysis pipeline for the project “PhytoFiber: Engineering an Anthocyanin-Functionalized Lignocellulosic Bio-Composite for Automated, Machine Learning-Assisted Spoilage Transduction.” It is organized for science-fair judging standards: raw data in one place, deterministic scripts, processed outputs separated from source data, and board-ready figures generated from code.

## What The Pipeline Does

- Runs unsupervised computer vision on fiber images using K-means clustering with `n_clusters=2`.
- Extracts fiber-only RGB statistics automatically and writes a processed color table.
- Converts tensile measurements to stress in MPa and performs Shapiro-Wilk, Levene, one-way ANOVA, and Tukey HSD.
- Fits a degree-2 polynomial calibration model from fiber color to pH and reports `R²`.
- Quantifies the chicken spoilage relationship with Pearson’s `r`.
- Trains a G-channel logistic regression spoilage classifier and compares it against a random forest benchmark.
- Produces high-resolution figures for a paper, board, or GitHub portfolio.

## Workspace Layout

```text
data/
  raw/
    spoilage_images/
    calibration_raw.csv
    calibration_data.csv
    tensile_raw.csv
    tensile_data.csv
    spoilage_raw.csv
    spoilage_data.csv
  processed/
images/
  raw/
img/                    # legacy archive of photos and scanned PDFs
docs/
  RAW_DATA_ENTRY_GUIDE.md
scripts/
  01_cv_extraction.py
  02_biomechanics_anova.py
  03_predictive_models.py
  run_cv_extraction.py
  run_statistics.py
  run_ml.py
  build_figures.py
src/
  phytofiber_analysis/
visualizations/
```

## Environment Setup

Python 3.10+ is the intended target. In this workspace the environment is already configured, but the standard setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Recommended Run Order

```powershell
python scripts/01_cv_extraction.py
python scripts/02_biomechanics_anova.py
python scripts/03_predictive_models.py
python scripts/build_figures.py
```

Legacy script names still work if you prefer them.

## Inputs Supported

The pipeline accepts both the original repo filenames and the simpler handout filenames.

### Mechanical data

- `tensile_raw.csv` with `sample_id,Group,Diameter_mm,Force_N`
- `tensile_data.csv` with `sample_id,group,force_n,diameter_mm`

### Calibration data

- `calibration_raw.csv` with `pH_Level,Image_Filename`
- `calibration_data.csv` with `sample_id,pH,R,G,B`

### Spoilage data

- `spoilage_raw.csv` with `sample_id,Time_Hours,Meat_pH,Image_Filename`
- `spoilage_data.csv` with `sample_id,time_h,meat_surface_ph`

### Image sources

The CV stage checks these in order:

1. [data/raw/spoilage_images](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw/spoilage_images)
2. [images/raw](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/images/raw)
3. [img](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/img)

For automated spoilage matching, image filenames should follow `<sample_id>_t<time_h>.<ext>`, for example `S01_t12.jpg`.

## Main Outputs

Processed tables are written to [data/processed](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/processed). Key artifacts include:

- `color_data_final.csv`
- `cv_extracted_spoilage.csv`
- `tensile_with_mpa.csv`
- `assumption_checks.csv`
- `anova_results.csv`
- `tukey_results.csv`
- `calibration_model.json`
- `calibration_predictions.csv`
- `pearson_results.json`
- `spoilage_labeled.csv`
- `model_comparison.csv`
- `ml_metrics.json`

Figures are written to [visualizations](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/visualizations):

- `tensile_strength_boxplot.png`
- `spoilage_regplot.png`
- `calibration_curve.png`
- `confusion_matrix_logistic.png`
- `confusion_matrix_random_forest.png`
- `predictive_analysis_dashboard.png`

## Important Data Integrity Note

The scanned PDFs in [img/classification](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/img/classification) and the handout PDF are useful as experimental references, but they are not a trustworthy substitute for structured numeric CSV data. This repository intentionally does not invent measurements from scanned sheets. To get valid statistical results, enter the real measurements into the raw CSV files.

## Why This Version Is Stronger

- The CV extraction is unsupervised rather than manual.
- The tensile workflow checks assumptions before interpreting ANOVA.
- The pH calibration is explicitly modeled as non-linear.
- The spoilage claim is supported by both correlation and classification.
- The figures are generated from code and are reproducible.

## Next Data You Should Add

If you want the full pipeline to produce final science-fair results immediately, the highest-value files to complete are:

1. [data/raw/tensile_raw.csv](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw/tensile_raw.csv)
2. [data/raw/calibration_raw.csv](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw/calibration_raw.csv)
3. [data/raw/spoilage_raw.csv](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw/spoilage_raw.csv)

Once those contain real measurements and matching image filenames, the scripts can generate defensible final outputs end-to-end.
