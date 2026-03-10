# Raw Data Entry Guide

This project now supports both the original repo filenames and the simpler handout filenames. Use one naming style consistently.

## 1) Supported CSV Inputs

Recommended files in [data/raw](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw):

- `tensile_raw.csv` or `tensile_data.csv`
- `calibration_raw.csv` or `calibration_data.csv`
- `spoilage_raw.csv` or `spoilage_data.csv`
- Optional: `latency_data.csv`, `stability_data.csv`, `digestibility_data.csv`, `economics_data.csv`

The pipeline automatically prefers the more detailed repo files when both exist.

## 2) Image Locations

Preferred active image folder:

- [data/raw/spoilage_images](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/data/raw/spoilage_images)

Also supported:

- [images/raw](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/images/raw)
- [img](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/img)

The CV script scans `data/raw/spoilage_images` first, then `images/raw`, then the legacy [img](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/img) archive.

## 3) Spoilage Image Naming

For automatic matching between pH measurements and photos, use:

- `S01_t0.jpg`
- `S01_t6.jpg`
- `S01_t12.jpg`
- `S02_t24.heic`

Pattern rule:

- `<sample_id>_t<time_h>.<ext>`

The parser extracts `sample_id` and `time_h` directly from the filename. This is required for fully automated spoilage analysis.

## 4) Required Columns

### `tensile_raw.csv`

- Required: `sample_id,Group,Diameter_mm,Force_N`
- The script normalizes these to internal analysis columns.

### `tensile_data.csv`

- Required: `sample_id,group,force_n,diameter_mm`

### `calibration_raw.csv`

- Required: `pH_Level,Image_Filename`
- The CV pipeline fills RGB values by matching the image filename.

### `calibration_data.csv`

- Required: `sample_id,pH,R,G,B`

### `spoilage_raw.csv`

- Required: `sample_id,Time_Hours,Meat_pH`
- Strongly recommended: `Image_Filename`

### `spoilage_data.csv`

- Required: `sample_id,time_h,meat_surface_ph`

## 5) Analysis Rules Used by the Pipeline

- Computer vision: unsupervised K-means with `n_clusters=2`, darker cluster retained as fiber.
- Tensile physics: radius, area, and stress are computed automatically.
- Assumption checks: Shapiro-Wilk and Levene are run before ANOVA reporting.
- Calibration model: degree-2 polynomial regression with reported `R²`.
- Spoilage threshold: `pH >= 6.8` is labeled as spoiled.
- Classifier: logistic regression uses G-channel only; random forest is included as an additional benchmark.

## 6) Run Order

Use either the numbered scripts or the original runner scripts.

1. `python scripts/01_cv_extraction.py`
2. `python scripts/02_biomechanics_anova.py`
3. `python scripts/03_predictive_models.py`
4. `python scripts/build_figures.py`

Equivalent legacy commands still work:

1. `python scripts/run_cv_extraction.py`
2. `python scripts/run_statistics.py`
3. `python scripts/run_ml.py`
4. `python scripts/build_figures.py`

## 7) Important Limitation

The scanned PDFs in [img/classification](c:/Users/dongz/OneDrive/Desktop/Project%20Code/PhytoFiber/img/classification) are not reliable structured datasets. They are useful as experimental records, but the pipeline does not invent numeric values from them. For publishable statistics, enter the real measurements into the CSV files above.
