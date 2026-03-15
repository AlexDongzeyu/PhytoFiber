from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / "calibration_validation.csv")

    required = ["Target_pH", "Actual_pH", "Rep_1_G", "Rep_2_G", "Rep_3_G"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration_validation.csv: {missing}")

    df_melted = df.melt(
        id_vars=["Target_pH", "Actual_pH"],
        value_vars=["Rep_1_G", "Rep_2_G", "Rep_3_G"],
        var_name="Replicate",
        value_name="G_Channel",
    )

    df_melted["Actual_pH"] = df_melted["Actual_pH"].astype(float)
    df_melted["G_Channel"] = df_melted["G_Channel"].astype(float)

    summary = (
        df_melted.groupby("Actual_pH", as_index=False)["G_Channel"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
        .sort_values("Actual_pH")
    )

    df_melted.to_csv(PROCESSED_DIR / "calibration_validation_long.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "calibration_validation_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="Actual_pH",
        y="G_Channel",
        data=df_melted,
        marker="o",
        color="#2c5aa0",
        linewidth=2.5,
        errorbar=("ci", 95),
    )

    plt.title("Validation Study: Micro-Calibration Response Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Measured pH")
    plt.ylabel("Green Channel Intensity (0-255)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axvline(x=6.8, color="red", linestyle="--", label="Spoilage Threshold (pH 6.8)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_validation_plot.png", dpi=300)
    plt.close()
    print("Saved calibration outputs to data/processed/ and figures/")


if __name__ == "__main__":
    main()
