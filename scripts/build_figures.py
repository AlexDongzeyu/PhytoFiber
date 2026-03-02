from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import CALIBRATION_CSV, DATA_PROCESSED_DIR, VIS_DIR
from phytofiber_analysis.io_utils import read_csv_checked
from phytofiber_analysis.visualization import (
    save_calibration_curve,
    save_confusion_matrix_heatmap,
    save_spoilage_regplot,
    save_tensile_boxplot,
)


def main() -> None:
    tensile = read_csv_checked(
        DATA_PROCESSED_DIR / "tensile_with_mpa.csv",
        required_columns=["group", "tensile_mpa"],
    )
    spoilage = read_csv_checked(
        DATA_PROCESSED_DIR / "spoilage_labeled.csv",
        required_columns=["G", "meat_surface_ph"],
    )

    save_tensile_boxplot(tensile, VIS_DIR / "tensile_strength_boxplot.png")
    save_spoilage_regplot(spoilage, VIS_DIR / "spoilage_regplot.png")

    if CALIBRATION_CSV.exists():
        calibration = read_csv_checked(CALIBRATION_CSV, required_columns=["pH", "G"])
        calibration = calibration.dropna(subset=["pH", "G"])
        if not calibration.empty:
            save_calibration_curve(calibration, VIS_DIR / "calibration_curve.png")

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
    print("Saved board-ready figures to visualizations/")


if __name__ == "__main__":
    main()

