from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VIS_DIR = PROJECT_ROOT / "visualizations"

# Core data files expected after collection
CALIBRATION_CSV = DATA_RAW_DIR / "calibration_data.csv"
TENSILE_CSV = DATA_RAW_DIR / "tensile_data.csv"
LATENCY_CSV = DATA_RAW_DIR / "latency_data.csv"
STABILITY_CSV = DATA_RAW_DIR / "stability_data.csv"
SPOILAGE_CSV = DATA_RAW_DIR / "spoilage_data.csv"

# Common generated files
CV_EXTRACTED_CSV = DATA_PROCESSED_DIR / "cv_extracted_spoilage.csv"
ANOVA_RESULTS_CSV = DATA_PROCESSED_DIR / "anova_results.csv"
TUKEY_RESULTS_CSV = DATA_PROCESSED_DIR / "tukey_results.csv"
ML_METRICS_JSON = DATA_PROCESSED_DIR / "ml_metrics.json"
MODEL_COMPARISON_CSV = DATA_PROCESSED_DIR / "model_comparison.csv"

