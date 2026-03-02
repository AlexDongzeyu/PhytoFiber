from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")
    sns.set_palette("deep")


def save_tensile_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df, x="group", y="tensile_mpa", width=0.55)
    sns.stripplot(data=df, x="group", y="tensile_mpa", color="black", size=6, alpha=0.7)
    plt.xlabel("Anthocyanin Concentration Group")
    plt.ylabel("Tensile Strength (MPa)")
    plt.title("Tensile Strength Distribution by Group")
    sns.despine(ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_spoilage_regplot(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(8, 6))
    ax = sns.regplot(
        data=df,
        x="G",
        y="meat_surface_ph",
        ci=95,
        scatter_kws={"s": 60, "alpha": 0.8},
        line_kws={"linewidth": 2.2},
    )
    plt.xlabel("Fiber Green Channel Intensity")
    plt.ylabel("Meat Surface pH")
    plt.title("Spoilage Kinetics: Fiber Color vs Meat pH")
    plt.axhline(6.5, linestyle="--", linewidth=1.2, color="crimson", alpha=0.85, label="Spoilage Threshold pH 6.5")
    plt.legend(frameon=False)
    sns.despine(ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_calibration_curve(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(8, 6))
    work = df.sort_values("pH")
    ax = sns.regplot(
        data=work,
        x="pH",
        y="G",
        ci=95,
        scatter_kws={"s": 75, "alpha": 0.9},
        line_kws={"linewidth": 2.2},
    )
    plt.xlabel("pH")
    plt.ylabel("Green Channel Intensity")
    plt.title("Colorimetric Calibration Curve (pH vs G-channel)")
    sns.despine(ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_confusion_matrix_heatmap(cm_df: pd.DataFrame, title: str, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(6.5, 5.5))
    matrix = cm_df.set_index("actual").loc[["actual_safe", "actual_spoiled"], ["pred_safe", "pred_spoiled"]]
    ax = sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, linewidths=0.5, linecolor="white")
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    sns.despine(ax=ax, left=True, bottom=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

