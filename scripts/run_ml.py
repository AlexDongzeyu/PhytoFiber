from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import SPOILAGE_CSV, CV_EXTRACTED_CSV, DATA_PROCESSED_DIR, ML_METRICS_JSON, MODEL_COMPARISON_CSV
from phytofiber_analysis.io_utils import read_csv_checked, write_csv, write_json
from phytofiber_analysis.ml_prediction import prepare_spoilage_labels, evaluate_classifier


def main() -> None:
    spoilage = read_csv_checked(
        SPOILAGE_CSV,
        required_columns=["sample_id", "time_h", "meat_surface_ph"],
    )
    rgb = read_csv_checked(
        CV_EXTRACTED_CSV,
        required_columns=["image_name", "sample_id", "time_h", "R", "G", "B"],
    )

    if rgb["sample_id"].isna().any() or rgb["time_h"].isna().any():
        raise ValueError(
            "Image filename metadata parse failed. Use naming format like 'S01_t12.jpg'."
        )

    spoilage["time_h"] = spoilage["time_h"].astype(float)
    rgb["time_h"] = rgb["time_h"].astype(float)
    merged = spoilage.merge(rgb, on=["sample_id", "time_h"], how="inner")
    if merged.empty:
        raise ValueError(
            "No matched spoilage rows found. Check that sample_id/time_h in CSV align with image names."
        )

    labeled = prepare_spoilage_labels(merged, ph_col="meat_surface_ph", threshold=6.5)

    feature_cols = ["R", "G", "B"]
    logistic_metrics, logistic_cm = evaluate_classifier(
        labeled, feature_cols=feature_cols, model_type="logistic"
    )
    rf_metrics, rf_cm = evaluate_classifier(
        labeled, feature_cols=feature_cols, model_type="random_forest"
    )

    comparison = pd.DataFrame([logistic_metrics, rf_metrics]).sort_values(
        by="accuracy", ascending=False
    )
    write_csv(labeled, DATA_PROCESSED_DIR / "spoilage_labeled.csv")
    write_csv(logistic_cm, DATA_PROCESSED_DIR / "confusion_matrix_logistic.csv")
    write_csv(rf_cm, DATA_PROCESSED_DIR / "confusion_matrix_random_forest.csv")
    write_csv(comparison, MODEL_COMPARISON_CSV)
    write_json({"logistic": logistic_metrics, "random_forest": rf_metrics}, ML_METRICS_JSON)
    print("Saved ML outputs to data/processed/")


if __name__ == "__main__":
    main()

