from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / "tensile_validation.csv")

    required = ["Sample", "Batch", "Group", "Avg_Diam_mm", "Force_N", "Fracture_Loc", "Valid_YN"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tensile_validation.csv: {missing}")

    df["Valid_YN"] = df["Valid_YN"].astype(str).str.strip().str.upper()
    df_valid = df[df["Valid_YN"] == "Y"].copy()
    if df_valid.empty:
        raise ValueError("No valid tensile rows found after filtering Valid_YN == 'Y'.")

    df_valid["Avg_Diam_mm"] = df_valid["Avg_Diam_mm"].astype(float)
    df_valid["Force_N"] = df_valid["Force_N"].astype(float)

    df_valid["Radius_mm"] = df_valid["Avg_Diam_mm"] / 2.0
    df_valid["Area_mm2"] = np.pi * (df_valid["Radius_mm"] ** 2)
    df_valid["Stress_MPa"] = df_valid["Force_N"] / df_valid["Area_mm2"]

    groups = sorted(df_valid["Group"].astype(str).unique())
    group_data = [df_valid.loc[df_valid["Group"] == g, "Stress_MPa"].values for g in groups]
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least two groups.")

    f_stat, p_val = stats.f_oneway(*group_data)
    anova_df = pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "groups": len(groups),
                "n_valid": int(len(df_valid)),
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "significant_alpha_0_05": bool(p_val < 0.05),
            }
        ]
    )

    summary_df = (
        df_valid.groupby("Group", as_index=False)["Stress_MPa"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    df_valid.to_csv(PROCESSED_DIR / "tensile_validation_processed.csv", index=False)
    anova_df.to_csv(PROCESSED_DIR / "tensile_validation_anova.csv", index=False)
    summary_df.to_csv(PROCESSED_DIR / "tensile_validation_summary.csv", index=False)

    print(f"Validation ANOVA: F={f_stat:.3f}, p-value={p_val:.6f}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Group", y="Stress_MPa", data=df_valid, inner=None, color="0.85")
    sns.stripplot(
        x="Group",
        y="Stress_MPa",
        data=df_valid,
        size=7,
        jitter=True,
        alpha=0.85,
        hue="Group",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.title("Validation Study: Tensile Strength by Formulation", fontsize=14, fontweight="bold")
    plt.ylabel("Tensile Stress (MPa)")
    plt.xlabel("Formulation Group")
    plt.text(0.01, 0.98, f"ANOVA p = {p_val:.5f}", transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tensile_validation_plot.png", dpi=300)
    plt.close()
    print("Saved tensile outputs to data/processed/ and figures/")


if __name__ == "__main__":
    main()
