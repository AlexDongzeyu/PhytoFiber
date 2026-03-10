from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import CALIBRATION_CSV, CALIBRATION_MODEL_JSON, CALIBRATION_PREDICTIONS_CSV, CALIBRATION_RAW_CSV, COLOR_DATA_FINAL_CSV, CV_EXTRACTED_CSV, DATA_PROCESSED_DIR, ML_METRICS_JSON, MODEL_COMPARISON_CSV, PEARSON_RESULTS_JSON, SPOILAGE_CSV, SPOILAGE_LABELED_CSV, SPOILAGE_RAW_CSV
from phytofiber_analysis.io_utils import choose_existing_file, maybe_rename_columns, read_csv_checked, write_csv, write_json
from phytofiber_analysis.ml_prediction import evaluate_classifier, fit_polynomial_calibration, prepare_spoilage_labels, run_pearson_correlation


def _load_spoilage() -> pd.DataFrame:
    spoilage_path = choose_existing_file(SPOILAGE_CSV, SPOILAGE_RAW_CSV)
    spoilage = pd.read_csv(spoilage_path)
    spoilage = maybe_rename_columns(
        spoilage,
        {
            "Time_Hours": "time_h",
            "Meat_pH": "meat_surface_ph",
            "Fiber_G_Channel": "G",
            "Image_Filename": "image_name",
        },
    )
    required = ["time_h", "meat_surface_ph"]
    missing = [col for col in required if col not in spoilage.columns]
    if missing:
        raise ValueError(f"Spoilage file is missing required columns: {missing}")
    if spoilage["meat_surface_ph"].dropna().empty:
        raise ValueError("Spoilage file exists, but pH measurements have not been entered yet.")
    return spoilage


def _load_calibration(rgb_table: pd.DataFrame) -> pd.DataFrame | None:
    try:
        calibration_path = choose_existing_file(CALIBRATION_CSV, CALIBRATION_RAW_CSV)
    except FileNotFoundError:
        return None

    calibration = pd.read_csv(calibration_path)
    calibration = maybe_rename_columns(
        calibration,
        {
            "pH_Level": "pH",
            "Image_Filename": "image_name",
        },
    )
    if "image_name" in calibration.columns:
        calibration["image_name"] = calibration["image_name"].fillna("").astype(str).str.strip()
    if "pH" in calibration.columns and calibration["pH"].dropna().empty:
        return None
    if "G" not in calibration.columns and "image_name" in calibration.columns and (calibration["image_name"] == "").all():
        return None
    if "G" not in calibration.columns and "image_name" in calibration.columns:
        calibration = calibration.merge(rgb_table[["image_name", "R", "G", "B"]], on="image_name", how="left")
    if {"pH", "G"}.issubset(calibration.columns):
        return calibration
    return None


def main() -> None:
    spoilage = _load_spoilage()
    rgb_path = choose_existing_file(CV_EXTRACTED_CSV, COLOR_DATA_FINAL_CSV)
    rgb = read_csv_checked(rgb_path, required_columns=["image_name", "R", "G", "B"])
    rgb = maybe_rename_columns(rgb, {"Fiber_G_Channel": "G"})

    calibration = _load_calibration(rgb)
    calibration_payload = None
    calibration_predictions = None
    if calibration is not None:
        calibration_payload, calibration_predictions = fit_polynomial_calibration(calibration, feature_col="G", target_col="pH", degree=2)
        write_json(calibration_payload, CALIBRATION_MODEL_JSON)
        write_csv(calibration_predictions, CALIBRATION_PREDICTIONS_CSV)

    join_cols = []
    if {"sample_id", "time_h"}.issubset(spoilage.columns) and {"sample_id", "time_h"}.issubset(rgb.columns):
        spoilage["sample_id"] = spoilage["sample_id"].fillna("").astype(str).str.strip()
        rgb["sample_id"] = rgb["sample_id"].fillna("").astype(str).str.strip()
        spoilage["time_h"] = spoilage["time_h"].astype(float)
        rgb["time_h"] = rgb["time_h"].astype(float)
        join_cols = ["sample_id", "time_h"]
    elif "image_name" in spoilage.columns and "image_name" in rgb.columns:
        spoilage["image_name"] = spoilage["image_name"].fillna("").astype(str).str.strip()
        rgb["image_name"] = rgb["image_name"].fillna("").astype(str).str.strip()
        join_cols = ["image_name"]
    else:
        raise ValueError("No common join key found between spoilage measurements and color extraction.")

    merged = spoilage.merge(rgb, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError(
            "No matched spoilage rows found. Check that sample IDs, times, or image filenames align."
        )

    labeled = prepare_spoilage_labels(merged, ph_col="meat_surface_ph", threshold=6.8)
    pearson_payload = run_pearson_correlation(labeled, x_col="G", y_col="meat_surface_ph")

    feature_cols = ["G"]
    logistic_metrics, logistic_cm = evaluate_classifier(
        labeled, feature_cols=feature_cols, model_type="logistic"
    )
    rf_metrics, rf_cm = evaluate_classifier(
        labeled, feature_cols=["R", "G", "B"], model_type="random_forest"
    )

    comparison = pd.DataFrame([logistic_metrics, rf_metrics]).sort_values(
        by="accuracy", ascending=False
    )
    write_csv(labeled, SPOILAGE_LABELED_CSV)
    write_csv(logistic_cm, DATA_PROCESSED_DIR / "confusion_matrix_logistic.csv")
    write_csv(rf_cm, DATA_PROCESSED_DIR / "confusion_matrix_random_forest.csv")
    write_csv(comparison, MODEL_COMPARISON_CSV)
    write_json(pearson_payload, PEARSON_RESULTS_JSON)
    write_json(
        {
            "calibration": calibration_payload,
            "pearson": pearson_payload,
            "logistic": logistic_metrics,
            "random_forest": rf_metrics,
        },
        ML_METRICS_JSON,
    )
    print("Saved ML outputs to data/processed/")


if __name__ == "__main__":
    main()

