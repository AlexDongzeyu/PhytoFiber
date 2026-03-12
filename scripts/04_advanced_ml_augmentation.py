from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.advanced_analysis import (
    fit_spoilage_response_surface,
    simulate_spoilage_monte_carlo,
    simulate_tensile_monte_carlo,
)
from phytofiber_analysis.config import (
    AUGMENTATION_SUMMARY_JSON,
    AUGMENTED_SPOILAGE_CSV,
    AUGMENTED_TENSILE_CSV,
    AUGMENTED_TENSILE_SUMMARY_CSV,
    COLOR_DATA_FINAL_CSV,
    CV_EXTRACTED_CSV,
    SPOILAGE_LABELED_CSV,
    SPOILAGE_RAW_CSV,
    TENSILE_PROCESSED_CSV,
    TENSILE_RAW_CSV,
    VIS_DIR,
)
from phytofiber_analysis.io_utils import choose_existing_file, maybe_rename_columns, read_csv_checked, write_csv, write_json
from phytofiber_analysis.ml_prediction import prepare_spoilage_labels
from phytofiber_analysis.statistical_tests import compute_tensile_stress
from phytofiber_analysis.visualization import save_raincloud_plot, save_spoilage_density_cloud, save_spoilage_response_surface


def _load_tensile() -> pd.DataFrame:
    tensile_path = choose_existing_file(TENSILE_PROCESSED_CSV, TENSILE_RAW_CSV)
    tensile = pd.read_csv(tensile_path)
    tensile = maybe_rename_columns(
        tensile,
        {
            "Group": "group",
            "Force_N": "force_n",
            "Diameter_mm": "diameter_mm",
            "Sample_ID": "sample_id",
            "stress_mpa": "tensile_mpa",
        },
    )
    if "tensile_mpa" not in tensile.columns:
        tensile = compute_tensile_stress(tensile)
    return tensile


def _load_spoilage() -> pd.DataFrame:
    if SPOILAGE_LABELED_CSV.exists():
        spoilage = pd.read_csv(SPOILAGE_LABELED_CSV)
        rename_map = {}
        if "time_h_x" in spoilage.columns:
            rename_map["time_h_x"] = "time_h"
        elif "time_h_y" in spoilage.columns and "time_h" not in spoilage.columns:
            rename_map["time_h_y"] = "time_h"
        if "sample_id_x" in spoilage.columns:
            rename_map["sample_id_x"] = "sample_id"
        spoilage = maybe_rename_columns(spoilage, rename_map)
        required = {"time_h", "meat_surface_ph", "G"}
        if required.issubset(spoilage.columns):
            return spoilage

    spoilage = pd.read_csv(SPOILAGE_RAW_CSV)
    spoilage = maybe_rename_columns(
        spoilage,
        {
            "Time_Hours": "time_h",
            "Meat_pH": "meat_surface_ph",
            "Image_Filename": "image_name",
            "Sample_ID": "sample_id",
        },
    )
    rgb_path = choose_existing_file(CV_EXTRACTED_CSV, COLOR_DATA_FINAL_CSV)
    rgb = read_csv_checked(rgb_path, required_columns=["image_name", "G"])

    join_cols = []
    has_spoilage_sample_time = {"sample_id", "time_h"}.issubset(spoilage.columns)
    has_rgb_sample_time = {"sample_id", "time_h"}.issubset(rgb.columns)
    rgb_has_populated_sample_time = False
    if has_rgb_sample_time:
        rgb_sample_id = rgb["sample_id"].fillna("").astype(str).str.strip()
        rgb_has_populated_sample_time = rgb_sample_id.ne("").any() and rgb["time_h"].notna().any()

    if has_spoilage_sample_time and has_rgb_sample_time and rgb_has_populated_sample_time:
        spoilage["sample_id"] = spoilage["sample_id"].fillna("").astype(str).str.strip()
        rgb["sample_id"] = rgb["sample_id"].fillna("").astype(str).str.strip()
        spoilage["time_h"] = spoilage["time_h"].astype(float)
        rgb["time_h"] = rgb["time_h"].astype(float)
        join_cols = ["sample_id", "time_h"]
    else:
        spoilage["image_name"] = spoilage["image_name"].fillna("").astype(str).str.strip()
        rgb["image_name"] = rgb["image_name"].fillna("").astype(str).str.strip()
        join_cols = ["image_name"]

    merged = spoilage.merge(rgb, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError("No matched spoilage rows found for augmentation. Run CV extraction first and verify join keys.")
    return prepare_spoilage_labels(merged, ph_col="meat_surface_ph", threshold=6.8)


def main() -> None:
    tensile = _load_tensile()
    spoilage = _load_spoilage()

    augmented_tensile, tensile_summary = simulate_tensile_monte_carlo(
        tensile,
        group_col="group",
        value_col="tensile_mpa",
        draws_per_group=1000,
    )
    augmented_spoilage, spoilage_summary = simulate_spoilage_monte_carlo(
        spoilage,
        time_col="time_h",
        ph_col="meat_surface_ph",
        signal_col="G",
        draws=1000,
    )
    surface_summary, surface_df = fit_spoilage_response_surface(
        spoilage,
        time_col="time_h",
        ph_col="meat_surface_ph",
        signal_col="G",
        grid_size=60,
    )

    write_csv(augmented_tensile, AUGMENTED_TENSILE_CSV)
    write_csv(tensile_summary, AUGMENTED_TENSILE_SUMMARY_CSV)
    write_csv(augmented_spoilage, AUGMENTED_SPOILAGE_CSV)
    write_json(
        {
            "tensile_monte_carlo": tensile_summary.to_dict(orient="records"),
            "spoilage_monte_carlo": spoilage_summary,
            "response_surface": surface_summary,
        },
        AUGMENTATION_SUMMARY_JSON,
    )

    save_raincloud_plot(
        augmented_tensile,
        VIS_DIR / "tensile_raincloud_monte_carlo.png",
        group_col="group",
        value_col="tensile_mpa",
        title="Monte Carlo Raincloud of Tensile Strength by Formulation",
        xlabel="Simulated Tensile Strength (MPa)",
        disclosure_note="In-Silico Visualization: Augmented via Monte Carlo (n=1000) based on physical sample variance (n=15).",
    )
    save_spoilage_density_cloud(
        augmented_spoilage,
        spoilage,
        VIS_DIR / "spoilage_density_cloud.png",
        ph_col="meat_surface_ph",
        signal_col="G",
    )
    save_spoilage_response_surface(
        surface_df,
        spoilage,
        VIS_DIR / "spoilage_response_surface_3d.png",
        time_col="time_h",
        ph_col="meat_surface_ph",
        signal_col="G",
        r2=surface_summary["r2"],
    )
    print("Saved augmentation outputs to data/processed/ and visualizations/")


if __name__ == "__main__":
    main()