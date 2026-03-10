from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VIS_DIR = PROJECT_ROOT / "visualizations"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_RAW_DIR = IMAGES_DIR / "raw"
LEGACY_IMAGE_DIR = PROJECT_ROOT / "img"
CV_DIAGNOSTICS_DIR = DATA_PROCESSED_DIR / "cv_diagnostics"

# Core data files expected after collection
CALIBRATION_CSV = DATA_RAW_DIR / "calibration_data.csv"
CALIBRATION_RAW_CSV = DATA_RAW_DIR / "calibration_raw.csv"
TENSILE_CSV = DATA_RAW_DIR / "tensile_data.csv"
TENSILE_RAW_CSV = DATA_RAW_DIR / "tensile_raw.csv"
LATENCY_CSV = DATA_RAW_DIR / "latency_data.csv"
STABILITY_CSV = DATA_RAW_DIR / "stability_data.csv"
SPOILAGE_CSV = DATA_RAW_DIR / "spoilage_data.csv"
SPOILAGE_RAW_CSV = DATA_RAW_DIR / "spoilage_raw.csv"
SPOILAGE_IMAGE_DIR = DATA_RAW_DIR / "spoilage_images"

# Common generated files
CV_EXTRACTED_CSV = DATA_PROCESSED_DIR / "cv_extracted_spoilage.csv"
COLOR_DATA_FINAL_CSV = DATA_PROCESSED_DIR / "color_data_final.csv"
IMAGE_INVENTORY_CSV = DATA_PROCESSED_DIR / "image_inventory.csv"
TENSILE_PROCESSED_CSV = DATA_PROCESSED_DIR / "tensile_with_mpa.csv"
ASSUMPTION_RESULTS_CSV = DATA_PROCESSED_DIR / "assumption_checks.csv"
EFFECT_SIZE_CSV = DATA_PROCESSED_DIR / "effect_sizes.csv"
CALIBRATION_MODEL_JSON = DATA_PROCESSED_DIR / "calibration_model.json"
CALIBRATION_PREDICTIONS_CSV = DATA_PROCESSED_DIR / "calibration_predictions.csv"
PEARSON_RESULTS_JSON = DATA_PROCESSED_DIR / "pearson_results.json"
SPOILAGE_LABELED_CSV = DATA_PROCESSED_DIR / "spoilage_labeled.csv"
ANOVA_RESULTS_CSV = DATA_PROCESSED_DIR / "anova_results.csv"
TUKEY_RESULTS_CSV = DATA_PROCESSED_DIR / "tukey_results.csv"
ML_METRICS_JSON = DATA_PROCESSED_DIR / "ml_metrics.json"
MODEL_COMPARISON_CSV = DATA_PROCESSED_DIR / "model_comparison.csv"


def first_existing_path(*paths: Path) -> Path:
	for path in paths:
		if path.exists():
			return path
	return paths[0]

