# Raw Data Entry Guide (ISEF Workflow)

This guide ensures your raw files map directly into the analysis pipeline without breaking any scripts.

## 1) File Naming (Required)

Create these files in `data/raw/` by duplicating the templates:

- `calibration_data.csv`
- `tensile_data.csv`
- `latency_data.csv`
- `stability_data.csv`
- `spoilage_data.csv`
- Optional: `digestibility_data.csv`, `economics_data.csv`

## 2) Spoilage Image Naming (Critical for ML)

Place all spoilage photos in `data/raw/spoilage_images/`.

Use this exact pattern:

- `S01_t0.jpg`
- `S01_t6.jpg`
- `S01_t12.jpg`
- `S02_t0.jpg`

Pattern rule:

- `<sample_id>_t<time_h>.<ext>`
- Example: `S03_t24.png`

The parser extracts:

- `sample_id` -> `S03`
- `time_h` -> `24`

## 3) Column Rules by File

### `calibration_data.csv`
- Required: `sample_id,pH,R,G,B`
- Fill with RGB values from ImageJ or CV extraction references.

### `tensile_data.csv`
- Required: `sample_id,group,force_n,diameter_mm`
- `group` should be consistent (e.g., `A_5`, `B_10`, `C_15`).
- `force_n` and `diameter_mm` must be numeric.

### `spoilage_data.csv`
- Required: `sample_id,time_h,meat_surface_ph`
- `sample_id` and `time_h` must match image names.
- Example row: `S01,12,6.8,...`

## 4) Recommended Data Quality Checks

- No blank values in required numeric fields.
- Use one unit system only (N, mm, hours, pH).
- Keep sample IDs consistent across every file.
- Keep replicate counts balanced (target `n=5` per group).

## 5) Run Order

1. `python scripts/run_cv_extraction.py`
2. `python scripts/run_statistics.py`
3. `python scripts/run_ml.py`
4. `python scripts/build_figures.py`

## 6) Output Artifacts

- Processed data: `data/processed/`
- Figures: `visualizations/`
- Board-ready charts include:
  - tensile boxplot + stripplot
  - spoilage regplot + pH threshold
  - calibration curve
  - confusion matrix heatmaps
