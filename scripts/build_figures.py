from pathlib import Path
import json
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import CALIBRATION_MODEL_JSON, CALIBRATION_PREDICTIONS_CSV, DATA_PROCESSED_DIR, ML_METRICS_JSON, PEARSON_RESULTS_JSON, SPOILAGE_LABELED_CSV, TENSILE_PROCESSED_CSV, VIS_DIR
from phytofiber_analysis.io_utils import read_csv_checked
from phytofiber_analysis.visualization import (
    save_analysis_dashboard,
    save_calibration_curve,
    save_confusion_matrix_heatmap,
    save_spoilage_regplot,
    save_tensile_boxplot,
)


def main() -> None:
    tensile = read_csv_checked(
        TENSILE_PROCESSED_CSV,
        required_columns=["group", "tensile_mpa"],
    )
    spoilage = read_csv_checked(
        SPOILAGE_LABELED_CSV,
        required_columns=["G", "meat_surface_ph"],
    )

    anova_p = None
    anova_path = DATA_PROCESSED_DIR / "anova_results.csv"
    if anova_path.exists():
        anova_df = pd.read_csv(anova_path)
        if not anova_df.empty and "p_value" in anova_df.columns:
            anova_p = float(anova_df.loc[0, "p_value"])

    pearson_r = None
    if PEARSON_RESULTS_JSON.exists():
        with PEARSON_RESULTS_JSON.open("r", encoding="utf-8") as handle:
            pearson_r = json.load(handle).get("pearson_r")

    save_tensile_boxplot(tensile, VIS_DIR / "tensile_strength_boxplot.png", anova_p=anova_p)
    save_spoilage_regplot(spoilage, VIS_DIR / "spoilage_regplot.png", threshold=6.8, pearson_r=pearson_r)

    if CALIBRATION_PREDICTIONS_CSV.exists():
        calibration = read_csv_checked(CALIBRATION_PREDICTIONS_CSV, required_columns=["pH", "G", "predicted_pH"])
        calibration_r2 = None
        if CALIBRATION_MODEL_JSON.exists():
            with CALIBRATION_MODEL_JSON.open("r", encoding="utf-8") as handle:
                calibration_r2 = json.load(handle).get("r2")
        if not calibration.empty:
            save_calibration_curve(calibration, VIS_DIR / "calibration_curve.png", prediction_col="predicted_pH", r2=calibration_r2)

    logistic_cm_path = DATA_PROCESSED_DIR / "confusion_matrix_logistic.csv"
    rf_cm_path = DATA_PROCESSED_DIR / "confusion_matrix_random_forest.csv"
    if logistic_cm_path.exists():
        logistic_cm = pd.read_csv(logistic_cm_path)
        save_confusion_matrix_heatmap(
            logistic_cm,
            title="Confusion Matrix (Logistic Regression)",
            out_path=VIS_DIR / "confusion_matrix_logistic.png",
        )
    if rf_cm_path.exists():
        rf_cm = pd.read_csv(rf_cm_path)
        save_confusion_matrix_heatmap(
            rf_cm,
            title="Confusion Matrix (Random Forest)",
            out_path=VIS_DIR / "confusion_matrix_random_forest.png",
        )

    if (DATA_PROCESSED_DIR / "model_comparison.csv").exists():
        comparison = pd.read_csv(DATA_PROCESSED_DIR / "model_comparison.csv")
        if not comparison.empty:
            save_analysis_dashboard(
                spoilage_df=spoilage,
                comparison_df=comparison,
                out_path=VIS_DIR / "predictive_analysis_dashboard.png",
                pearson_r=pearson_r,
            )
    print("Saved board-ready figures to visualizations/")


if __name__ == "__main__":
    main()

