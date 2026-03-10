from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.advanced_analysis import (
    apply_4pl_calibration,
    build_formulation_radar_scores,
    compute_bland_altman,
    estimate_bayesian_group_posteriors,
    fit_4pl_calibration,
    fit_svm_classifier,
    fit_weibull_reliability,
    summarize_latency,
)
from phytofiber_analysis.config import (
    ADVANCED_METRICS_JSON,
    BAYESIAN_LATENCY_PAIRWISE_CSV,
    BAYESIAN_LATENCY_SUMMARY_CSV,
    BAYESIAN_TENSILE_PAIRWISE_CSV,
    BAYESIAN_TENSILE_SUMMARY_CSV,
    BLAND_ALTMAN_CSV,
    CALIBRATION_4PL_JSON,
    CALIBRATION_4PL_PREDICTIONS_CSV,
    CALIBRATION_PREDICTIONS_CSV,
    COLOR_DATA_FINAL_CSV,
    ECONOMICS_CSV,
    FORMULATION_RADAR_CSV,
    LATENCY_CSV,
    LATENCY_SUMMARY_CSV,
    SPOILAGE_LABELED_CSV,
    STABILITY_CSV,
    TENSILE_PROCESSED_CSV,
    TENSILE_RAW_CSV,
    VIS_DIR,
    WEIBULL_PLOT_CSV,
    WEIBULL_SUMMARY_CSV,
    SVM_PREDICTIONS_CSV,
    DIGESTIBILITY_CSV,
    CALIBRATION_RAW_CSV,
)
from phytofiber_analysis.io_utils import choose_existing_file, maybe_rename_columns, read_csv_checked, write_csv, write_json
from phytofiber_analysis.statistical_tests import compute_group_descriptives, compute_tensile_stress
from phytofiber_analysis.visualization import (
    save_4pl_calibration_plot,
    save_bayesian_forest_plot,
    save_bland_altman_plot,
    save_digestibility_bars,
    save_economics_breakdown,
    save_formulation_radar,
    save_raincloud_plot,
    save_stability_timeseries,
    save_superiority_heatmap,
    save_weibull_probability_plot,
)


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
        },
    )
    if "tensile_mpa" not in tensile.columns:
        tensile = compute_tensile_stress(tensile)
    return tensile


def _load_calibration_for_4pl() -> pd.DataFrame:
    if CALIBRATION_PREDICTIONS_CSV.exists():
        calibration = pd.read_csv(CALIBRATION_PREDICTIONS_CSV)
        if {"G", "pH"}.issubset(calibration.columns):
            return calibration[["G", "pH"]].copy()

    calibration = pd.read_csv(CALIBRATION_RAW_CSV)
    calibration = maybe_rename_columns(calibration, {"pH_Level": "pH", "Image_Filename": "image_name"})
    rgb = read_csv_checked(COLOR_DATA_FINAL_CSV, required_columns=["image_name", "G"])
    merged = calibration.merge(rgb[["image_name", "G"]], on="image_name", how="left")
    return merged[["G", "pH"]].dropna().copy()


def _load_spoilage_for_advanced() -> pd.DataFrame:
    spoilage = read_csv_checked(SPOILAGE_LABELED_CSV, required_columns=["G", "meat_surface_ph", "target_spoiled"])
    rename_map = {}
    if "time_h_x" in spoilage.columns:
        rename_map["time_h_x"] = "time_h"
    elif "time_h_y" in spoilage.columns and "time_h" not in spoilage.columns:
        rename_map["time_h_y"] = "time_h"
    if "sample_id_x" in spoilage.columns:
        rename_map["sample_id_x"] = "sample_id"
    spoilage = maybe_rename_columns(spoilage, rename_map)
    return spoilage


def _compute_cost_per_meter(economics: pd.DataFrame) -> float:
    length_row = economics[economics["ingredient"] == "total_fiber_length_m"]
    if length_row.empty:
        total_cost = float(economics["cost_usd"].sum())
        total_length = 1.0
    else:
        total_cost = float(length_row["cost_usd"].iloc[0])
        total_length = float(length_row["amount_used"].iloc[0])
    return total_cost / total_length if total_length else float("nan")


def main() -> None:
    tensile = _load_tensile()
    weibull_summary, weibull_points = fit_weibull_reliability(tensile, group_col="group", value_col="tensile_mpa")
    tensile_descriptives = compute_group_descriptives(tensile, group_col="group", value_col="tensile_mpa")
    tensile_bayes_summary, tensile_bayes_pairwise = estimate_bayesian_group_posteriors(
        tensile,
        group_col="group",
        value_col="tensile_mpa",
        better_direction="higher",
    )
    write_csv(weibull_summary, WEIBULL_SUMMARY_CSV)
    write_csv(weibull_points, WEIBULL_PLOT_CSV)
    write_csv(tensile_bayes_summary, BAYESIAN_TENSILE_SUMMARY_CSV)
    write_csv(tensile_bayes_pairwise, BAYESIAN_TENSILE_PAIRWISE_CSV)

    save_raincloud_plot(
        tensile,
        VIS_DIR / "tensile_raincloud.png",
        group_col="group",
        value_col="tensile_mpa",
        title="Raincloud Plot of Tensile Strength by Formulation",
        xlabel="Tensile Strength (MPa)",
    )
    save_weibull_probability_plot(weibull_points, weibull_summary, VIS_DIR / "weibull_probability_plot.png")
    save_bayesian_forest_plot(
        tensile_bayes_summary,
        VIS_DIR / "bayesian_tensile_forest.png",
        title="Bayesian Posterior Tensile Strength by Formulation",
        xlabel="Posterior Median Tensile Strength (MPa)",
    )
    save_superiority_heatmap(
        tensile_bayes_pairwise,
        VIS_DIR / "bayesian_tensile_superiority_heatmap.png",
        title="Bayesian Superiority Probability: Higher Tensile Strength",
        better_label="P(row stronger than column)",
    )

    calibration = _load_calibration_for_4pl()
    calibration_4pl_payload, calibration_4pl_predictions = fit_4pl_calibration(calibration, feature_col="G", target_col="pH")
    write_json(calibration_4pl_payload, CALIBRATION_4PL_JSON)
    write_csv(calibration_4pl_predictions, CALIBRATION_4PL_PREDICTIONS_CSV)
    save_4pl_calibration_plot(calibration_4pl_predictions, VIS_DIR / "calibration_4pl_curve.png", calibration_4pl_payload)

    spoilage = _load_spoilage_for_advanced()
    spoilage = apply_4pl_calibration(spoilage, calibration_4pl_payload, feature_col="G", out_col="predicted_pH_4pl")
    bland_altman_metrics, bland_altman_points = compute_bland_altman(spoilage, reference_col="meat_surface_ph", candidate_col="predicted_pH_4pl")
    write_csv(bland_altman_points, BLAND_ALTMAN_CSV)

    svm_metrics, svm_predictions, svm_model = fit_svm_classifier(spoilage, feature_cols=["time_h", "G"], target_col="target_spoiled")
    write_csv(svm_predictions, SVM_PREDICTIONS_CSV)

    latency = pd.read_csv(LATENCY_CSV)
    latency = maybe_rename_columns(latency, {"Group": "group", "Response_Time_s": "response_time_s"})
    latency_summary = summarize_latency(latency, group_col="group", value_col="response_time_s")
    latency_bayes_summary, latency_bayes_pairwise = estimate_bayesian_group_posteriors(
        latency,
        group_col="group",
        value_col="response_time_s",
        better_direction="lower",
    )
    write_csv(latency_summary, LATENCY_SUMMARY_CSV)
    write_csv(latency_bayes_summary, BAYESIAN_LATENCY_SUMMARY_CSV)
    write_csv(latency_bayes_pairwise, BAYESIAN_LATENCY_PAIRWISE_CSV)
    save_raincloud_plot(
        latency,
        VIS_DIR / "latency_raincloud.png",
        group_col="group",
        value_col="response_time_s",
        title="Raincloud Plot of Halochromic Response Latency",
        xlabel="Response Time (s)",
    )
    save_bayesian_forest_plot(
        latency_bayes_summary,
        VIS_DIR / "bayesian_latency_forest.png",
        title="Bayesian Posterior Response Time by Formulation",
        xlabel="Posterior Median Response Time (s)",
    )
    save_superiority_heatmap(
        latency_bayes_pairwise,
        VIS_DIR / "bayesian_latency_superiority_heatmap.png",
        title="Bayesian Superiority Probability: Faster Response",
        better_label="P(row faster than column)",
    )

    stability = pd.read_csv(STABILITY_CSV)
    stability = maybe_rename_columns(stability, {"Time_Hours": "time_h"})
    save_stability_timeseries(stability, VIS_DIR / "stability_timeseries.png")

    digestibility = pd.read_csv(DIGESTIBILITY_CSV)
    save_digestibility_bars(digestibility, VIS_DIR / "digestibility_mass_loss.png")

    economics = pd.read_csv(ECONOMICS_CSV)
    cost_per_meter = _compute_cost_per_meter(economics)
    save_economics_breakdown(economics, VIS_DIR / "economics_breakdown.png", cost_per_meter=cost_per_meter)

    radar_scores = build_formulation_radar_scores(tensile_descriptives, latency_summary, weibull_summary)
    write_csv(radar_scores, FORMULATION_RADAR_CSV)
    save_formulation_radar(radar_scores, VIS_DIR / "formulation_optimization_radar.png")

    write_json(
        {
            "weibull": weibull_summary.to_dict(orient="records"),
            "bayesian_tensile": {
                "summary": tensile_bayes_summary.to_dict(orient="records"),
                "pairwise": tensile_bayes_pairwise.to_dict(orient="records"),
            },
            "calibration_4pl": calibration_4pl_payload,
            "bland_altman": bland_altman_metrics,
            "svm": svm_metrics,
            "latency_summary": latency_summary.to_dict(orient="records"),
            "bayesian_latency": {
                "summary": latency_bayes_summary.to_dict(orient="records"),
                "pairwise": latency_bayes_pairwise.to_dict(orient="records"),
            },
            "cost_per_meter_usd": cost_per_meter,
            "radar_scores": radar_scores.to_dict(orient="records"),
        },
        ADVANCED_METRICS_JSON,
    )
    print("Saved advanced research-grade outputs to data/processed/ and visualizations/")


if __name__ == "__main__":
    main()