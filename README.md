# PhytoFiber

PhytoFiber is a reproducible analysis project for anthocyanin-cellulose fiber testing, spoilage sensing, and validation.

## Structure

- `data/raw/`: source CSV inputs and raw evidence image folders.
- `data/processed/`: generated analysis outputs (ignored by git).
- `images/raw/`: canonical images used by CV extraction.
- `scripts/`: main pipeline entry points.
- `src/phytofiber_analysis/`: reusable analysis utilities.
- `visualizations/`: generated figures (ignored by git).
- `PhytoFiber_Validation_Phase/`: independent validation dataset, scripts, and validation figures.
- `docs/`: data-entry and deliverable guidance.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Order

```powershell
python scripts/01_cv_extraction.py
python scripts/02_biomechanics_anova.py
python scripts/03_predictive_models.py
python scripts/04_advanced_ml_augmentation.py
python scripts/build_figures.py
```

## Input Files

Core:

- `data/raw/tensile_raw.csv`
- `data/raw/calibration_raw.csv`
- `data/raw/spoilage_raw.csv`

Advanced:

- `data/raw/latency_data.csv`
- `data/raw/stability_data.csv`
- `data/raw/digestibility_data.csv`
- `data/raw/economics_data.csv`

Validation phase:

- `PhytoFiber_Validation_Phase/data/raw/tensile_validation.csv`
- `PhytoFiber_Validation_Phase/data/raw/calibration_validation.csv`
- `PhytoFiber_Validation_Phase/data/raw/spoilage_validation.csv`

## Notes

- `data/processed/` and `visualizations/` are reproducible build artifacts.
- `img/classification/` contains supporting scanned references, not machine-readable model inputs.
