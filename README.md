# PhytoFiber

Reproducible analysis repository for the PhytoFiber project: engineering anthocyanin-cellulose fibers for smart food packaging, with computer-vision color extraction, biomechanics testing, machine-learning spoilage classification, Bayesian inference, and blinded validation.

---

## Directory Structure

```
PhytoFiber/
│
├── data/
│   ├── raw/                        # Source measurement CSVs + raw evidence images
│   │   ├── tensile_raw.csv         # Tensile force + diameter measurements
│   │   ├── calibration_raw.csv     # pH calibration measurements
│   │   ├── spoilage_raw.csv        # Chicken spoilage time-series measurements
│   │   ├── latency_data.csv        # Halochromic response latency timings
│   │   ├── stability_data.csv      # Color-stability timepoint values
│   │   ├── digestibility_data.csv  # Digestibility mass-change measurements
│   │   ├── economics_data.csv      # Ingredient cost breakdown
│   │   ├── calibration_img/        # Raw calibration photos (PXL_*.jpg)
│   │   ├── tensil_img/             # Raw tensile test photos (IMG_*.HEIC, PXL_*.jpg)
│   │   ├── spoilage_img/           # Raw spoilage timeline photos (PXL_*.jpg)
│   │   ├── latency_img/            # Raw latency test photos (PXL_*.jpg)
│   │   ├── stability_img/          # Raw stability test photos (PXL_*.jpg)
│   │   └── digestibility_img/      # Raw digestibility photos (PXL_*.jpg)
│   └── processed/                  # Generated outputs from pipeline (not committed to git)
│
├── images/
│   └── raw/                        # Curated canonical images used by CV pipeline
│       ├── pH2.jpg … pH10.jpg      # Calibration reference images (pH 2, 4, 6, 8, 10)
│       └── chicken_0h.jpg … _36h   # Spoilage time-series images (0, 6, 12, 18, 24, 36 h)
│
├── img/
│   └── classification/             # Scanned supporting PDFs (reference records only)
│       ├── Chicken.pdf             # Chicken spoilage test record
│       ├── GAS.pdf                 # Latency / gas detection record
│       ├── My pH for fibre.pdf     # pH calibration record
│       ├── Tensile Strength Test.pdf
│       ├── cook and colour stability.pdf
│       └── digest.pdf              # Digestibility record
│
├── scripts/                        # Main pipeline entry points
│   ├── 01_cv_extraction.py         # Computer-vision RGB color extraction
│   ├── 02_biomechanics_anova.py    # Tensile stress calculation + one-way ANOVA
│   ├── 03_predictive_models.py     # Calibration regression + spoilage classifier
│   ├── 04_advanced_ml_augmentation.py  # Monte Carlo tensile/spoilage augmentation
│   ├── build_figures.py            # Master figure builder (core + advanced)
│   ├── run_advanced_analysis.py    # Advanced layer: Weibull, Bayesian, 4PL, SVM, radar
│   ├── run_cv_extraction.py        # Alias for 01_cv_extraction.py
│   ├── run_statistics.py           # Alias for 02_biomechanics_anova.py
│   └── run_ml.py                   # Alias for 03_predictive_models.py
│
├── src/
│   └── phytofiber_analysis/        # Reusable Python library used by all scripts
│       ├── config.py               # All file-path constants for the project
│       ├── cv_extraction.py        # Image loading, K-means clustering, color feature extraction
│       ├── statistical_tests.py    # Normality, Levene, ANOVA, Tukey, effect sizes
│       ├── ml_prediction.py        # Calibration models, logistic + RF spoilage classifiers
│       ├── advanced_analysis.py    # Weibull, Bayesian posteriors, 4PL, SVM, Monte Carlo
│       ├── visualization.py        # All figure-generation functions
│       ├── io_utils.py             # CSV/JSON read-write helpers
│       └── __init__.py
│
├── notebooks/                      # Jupyter notebook alternatives to the pipeline scripts
│   ├── 01_cv_extraction.ipynb
│   ├── 02_statistical_tests.ipynb
│   └── 03_ml_prediction.ipynb
│
├── visualizations/                 # Generated figures — reproduced by build_figures.py
│   ├── tensile_strength_violin.png
│   ├── tensile_raincloud.png
│   ├── bayesian_tensile_forest.png
│   ├── bayesian_tensile_superiority_heatmap.png
│   ├── calibration_curve.png
│   ├── calibration_4pl_curve.png
│   ├── confusion_matrix_logistic.png
│   ├── roc_curve_logistic.png
│   ├── predictive_analysis_dashboard.png
│   ├── spoilage_regplot.png
│   ├── weibull_probability_plot.png
│   ├── latency_raincloud.png
│   ├── bayesian_latency_forest.png
│   ├── bayesian_latency_superiority_heatmap.png
│   ├── stability_timeseries.png
│   ├── digestibility_mass_loss.png
│   ├── economics_breakdown.png
│   └── formulation_optimization_radar.png
│
├── PhytoFiber_Validation_Phase/    # Independent blinded-validation dataset and scripts
│   ├── data/
│   │   ├── raw/                    # Validation input CSVs
│   │   │   ├── tensile_validation.csv
│   │   │   ├── calibration_validation.csv
│   │   │   └── spoilage_validation.csv
│   │   └── processed/              # Validation analysis outputs
│   │       ├── tensile_validation_anova.csv
│   │       ├── tensile_validation_summary.csv
│   │       ├── tensile_validation_processed.csv
│   │       ├── calibration_validation_summary.csv
│   │       ├── calibration_validation_long.csv
│   │       ├── spoilage_validation_labeled.csv
│   │       ├── spoilage_validation_confusion_matrix.csv
│   │       └── spoilage_validation_classification_report.csv
│   ├── figures/                    # Validation plots (committed to git)
│   │   ├── tensile_validation_plot.png   — tensile ANOVA replication
│   │   ├── calibration_validation_plot.png — pH calibration replication
│   │   └── blinded_spoilage_cm.png       — blinded spoilage confusion matrix
│   └── scripts/
│       ├── 01_tensile_validation.py
│       ├── 02_calibration_validation.py
│       └── 03_spoilage_validation.py
│
├── docs/
│   ├── RAW_DATA_ENTRY_GUIDE.md     # Field-by-field guide for filling raw CSVs
│   └── DELIVERABLES_CHECKLIST.md   # Checklist of code, data, figure, and board outputs
│
├── 04_advanced_ml_augmentation.py  # Root-level convenience wrapper → scripts/04_...py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Run Order

Run the main pipeline in this order. Each step reads from `data/raw/` and writes to `data/processed/`.

```powershell
# 1. Extract RGB color features from calibration and spoilage images
python scripts/01_cv_extraction.py

# 2. Compute tensile stress, ANOVA, Tukey HSD, and effect sizes
python scripts/02_biomechanics_anova.py

# 3. Fit calibration regression and train spoilage classifiers
python scripts/03_predictive_models.py

# 4. Monte Carlo augmentation for tensile and spoilage distributions
python scripts/04_advanced_ml_augmentation.py

# 5. Generate all figures into visualizations/
python scripts/build_figures.py
```

`build_figures.py` automatically runs the advanced analysis layer (Weibull reliability, Bayesian posteriors, 4PL calibration, SVM surface, latency/stability/digestibility/economics charts) when the supporting input tables are present.

The `run_*.py` scripts in `scripts/` are exact aliases for steps 1–3.
The root-level `04_advanced_ml_augmentation.py` is a convenience entry point for step 4.

---

## Core Input Files

| File | Test |
|---|---|
| `data/raw/tensile_raw.csv` | Tensile strength (sample_id, Group, Diameter_mm, Force_N) |
| `data/raw/calibration_raw.csv` | pH calibration (pH_Level, Image_Filename) |
| `data/raw/spoilage_raw.csv` | Chicken spoilage (sample_id, Time_Hours, Meat_pH, Image_Filename) |
| `data/raw/latency_data.csv` | Halochromic response latency |
| `data/raw/stability_data.csv` | Color-stability timeseries |
| `data/raw/digestibility_data.csv` | In-vitro digestibility mass change |
| `data/raw/economics_data.csv` | Cost-per-meter ingredient breakdown |

See `docs/RAW_DATA_ENTRY_GUIDE.md` for field-level guidance on each file.

---

## Images

| Folder | Contents |
|---|---|
| `images/raw/` | Curated canonical images for CV pipeline — pH2…pH10 calibration series; chicken_0h…36h spoilage series |
| `data/raw/calibration_img/` | Full raw calibration photo archive |
| `data/raw/tensil_img/` | Full raw tensile test photo archive |
| `data/raw/spoilage_img/` | Full raw spoilage photo archive |
| `data/raw/latency_img/` | Full raw latency test photo archive |
| `data/raw/stability_img/` | Full raw stability photo archive |
| `data/raw/digestibility_img/` | Full raw digestibility photo archive |
| `img/classification/` | Scanned supporting PDFs (reference records, not model inputs) |

The CV pipeline reads from `images/raw/`. The `data/raw/*_img/` folders are raw evidence archives.

---

## Generated Outputs

`data/processed/` and `visualizations/` are reproducible build artifacts and are not committed to git (except for the validation phase outputs in `PhytoFiber_Validation_Phase/`).

Key processed tables:

| File | Produced by |
|---|---|
| `color_data_final.csv`, `cv_extracted_spoilage.csv` | `01_cv_extraction.py` |
| `tensile_with_mpa.csv`, `anova_results.csv`, `tukey_results.csv`, `effect_sizes.csv` | `02_biomechanics_anova.py` |
| `spoilage_labeled.csv`, `calibration_predictions.csv`, `ml_metrics.json`, `model_comparison.csv` | `03_predictive_models.py` |
| `weibull_summary.csv`, `bayesian_tensile_summary.csv`, `bayesian_latency_summary.csv`, `advanced_metrics.json` | `build_figures.py` (advanced layer) |
| `augmented_tensile_monte_carlo.csv`, `augmented_spoilage_monte_carlo.csv` | `04_advanced_ml_augmentation.py` |

---

## Validation Phase

`PhytoFiber_Validation_Phase/` is an independent blinded-validation study. Run its scripts separately:

```powershell
python PhytoFiber_Validation_Phase/scripts/01_tensile_validation.py
python PhytoFiber_Validation_Phase/scripts/02_calibration_validation.py
python PhytoFiber_Validation_Phase/scripts/03_spoilage_validation.py
```

Validation figures (committed to git):

| Figure | Test |
|---|---|
| `figures/tensile_validation_plot.png` | Tensile ANOVA replication |
| `figures/calibration_validation_plot.png` | pH calibration replication |
| `figures/blinded_spoilage_cm.png` | Blinded spoilage confusion matrix |
