# Raw Data Entry Guide

Use the active files in `data/raw/`. The older template files are no longer the working source of truth.

## Files To Fill

Core pipeline inputs:

- `tensile_raw.csv`
- `calibration_raw.csv`
- `spoilage_raw.csv`

Supporting advanced-analysis inputs:

- `latency_data.csv`
- `stability_data.csv`
- `digestibility_data.csv`
- `economics_data.csv`

If time is limited, fill the first three first. They drive the main CV, statistics, and ML workflow.

## Image Sources

- `images/raw/`: standardized working images used by the CV pipeline.
- `data/raw/*_img/`: raw photo archives for calibration, spoilage, latency, stability, digestibility, and tensile reference records.
- `img/classification/`: scanned supporting PDFs and record sheets.

The current scripted CV workflow is intended to run from the curated image set in `images/raw/`. The larger raw archives are kept as evidence and reference material.

## Spoilage Naming Rule

For automatic spoilage matching, use filenames like:

- `S01_t0.jpg`
- `S01_t6.jpg`
- `S01_t12.jpg`
- `S02_t24.heic`

Pattern:

- `<sample_id>_t<time_h>.<ext>`

That lets the parser recover `sample_id` and `time_h` directly from the file name.

## What Each File Should Contain

`tensile_raw.csv`

- `sample_id`: unique fiber ID such as `T01`
- `Group`: formulation group such as `A`, `B`, or `C`
- `Diameter_mm`: fiber diameter in millimeters
- `Force_N`: breaking force in newtons

`calibration_raw.csv`

- `pH_Level`: known pH value
- `Image_Filename`: exact file name of the calibration image

`spoilage_raw.csv`

- `sample_id`: sample ID such as `S01`
- `Time_Hours`: elapsed time
- `Meat_pH`: measured chicken pH
- `Image_Filename`: exact image file name

`latency_data.csv`

- response timing measurements for the halochromic response

`stability_data.csv`

- timepoint color values and color retention measurements

`digestibility_data.csv`

- mass change or structure-degradation measurements

`economics_data.csv`

- ingredient, amount, unit, and cost records

## Analysis Rules Used In The Repo

- CV extraction uses unsupervised K-means and keeps the darker fiber cluster.
- Tensile stress is computed automatically from diameter and breaking force.
- Statistics include assumption checks before ANOVA interpretation.
- The main calibration model is polynomial and the advanced layer adds a 4-parameter logistic fit.
- Spoilage labeling uses `pH >= 6.8` as the spoiled threshold.
- The core classifier uses logistic regression; the advanced layer adds an SVM surface.

## Run Order

```powershell
python scripts/01_cv_extraction.py
python scripts/02_biomechanics_anova.py
python scripts/03_predictive_models.py
python scripts/build_figures.py
```

The `run_*` scripts are equivalent alternatives. `build_figures.py` also runs the advanced figure layer when the supporting advanced-analysis tables are present.

## Important Limitation

The scanned PDFs in `img/classification/` are supporting records, not machine-readable datasets. The repo should only make statistical claims from the structured CSV measurements.
