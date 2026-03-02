# Data Analysis for Science Fair Project "PhytoFiber: Engineering an Anthocyanin-Functionalized Lignocellulosic Bio-Composite for Automated, Machine Learning-Assisted Spoilage Transduction"

## Project Goals

- Standardize colorimetric and halochromic measurements with reproducible computer vision.
- Validate mechanical and thermodynamic performance statistically.
- Train a predictive spoilage classifier from RGB-derived features.
- Generate publication-style figures for display board and paper.

## Repository Structure

```text
data/
  raw/                  # manually collected raw CSV + image folders
  processed/            # cleaned outputs and analysis artifacts
notebooks/
  01_cv_extraction.ipynb
  02_statistical_tests.ipynb
  03_ml_prediction.ipynb
scripts/
  run_cv_extraction.py
  run_statistics.py
  run_ml.py
  build_figures.py
src/
  phytofiber_analysis/
    __init__.py
    config.py
    io_utils.py
    cv_extraction.py
    statistical_tests.py
    ml_prediction.py
    visualization.py
visualizations/         # board-ready high DPI figures
requirements.txt
```

## Quick Start

1. Create and activate virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Put your data in `data/raw/`:
   - CSV files (use templates provided)
   - photos for spoilage/cv extraction
4. Run:
   - `python scripts/run_cv_extraction.py`
   - `python scripts/run_statistics.py`
   - `python scripts/run_ml.py`
   - `python scripts/build_figures.py`

## Input Data Templates

Templates are in `data/raw/` with `_template.csv` suffix. Duplicate each template and rename without `_template` before analysis.

Detailed entry rules are in:

- `docs/RAW_DATA_ENTRY_GUIDE.md`

### Naming convention for spoilage photos

The ML pipeline matches rows using `sample_id` + `time_h` extracted from the image filename:

- Required image pattern: `<sample_id>_t<time_h>.jpg`
- Example: `S01_t12.jpg`

Store images in:

- `data/raw/spoilage_images/`

## Statistical Protocols

- Normality: Shapiro-Wilk (`scipy.stats.shapiro`)
- Homogeneity of variance: Levene (`scipy.stats.levene`)
- Group comparison: One-way ANOVA (`scipy.stats.f_oneway`)
- Post-hoc pairwise comparison: Tukey HSD (`statsmodels.stats.multicomp.pairwise_tukeyhsd`)

## Machine Learning Protocol

- Target label from spoilage pH:
  - `0` = safe (`pH < 6.5`)
  - `1` = spoiled (`pH >= 6.5`)
- Features: RGB channels plus optional engineered ratio feature.
- Baseline classifier: Logistic Regression
- Optional model: Random Forest
- Outputs:
  - confusion matrix CSV
  - metrics JSON
  - model comparison CSV

## Reproducibility Notes

- All processing scripts are deterministic with `random_state=42`.
- Figure generation uses `seaborn-v0_8-whitegrid` and high-resolution export (`dpi=300`).
- Pipeline writes all derived files into `data/processed/` or `visualizations/`.

## Current Figure Outputs

After running all scripts, expected visuals include:

- `visualizations/tensile_strength_boxplot.png`
- `visualizations/spoilage_regplot.png`
- `visualizations/calibration_curve.png` (generated when calibration data is present)
- `visualizations/confusion_matrix_logistic.png` (generated when ML outputs are present)
- `visualizations/confusion_matrix_random_forest.png` (generated when ML outputs are present)
