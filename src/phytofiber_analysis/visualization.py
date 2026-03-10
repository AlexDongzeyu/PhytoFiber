from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.facecolor": "#f7f3eb",
            "axes.facecolor": "#fffaf2",
            "axes.edgecolor": "#3a312a",
            "axes.labelcolor": "#2b2521",
            "text.color": "#2b2521",
            "xtick.color": "#4f463f",
            "ytick.color": "#4f463f",
            "grid.color": "#d8cfc2",
            "axes.titleweight": "bold",
            "font.size": 12,
        },
    )


def _finish_figure(out_path: Path, dpi: int = 320) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_tensile_boxplot(df: pd.DataFrame, out_path: Path, anova_p: float | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(9, 6.5))
    order = list(pd.Series(df["group"]).dropna().astype(str).unique())
    palette = ["#244c5a", "#d17a22", "#7a9e48"][: len(order)]
    ax = sns.boxplot(data=df, x="group", y="tensile_mpa", width=0.55, order=order, palette=palette, linewidth=1.5)
    sns.swarmplot(data=df, x="group", y="tensile_mpa", order=order, color="#1f1f1f", size=6, alpha=0.85)
    means = df.groupby("group")["tensile_mpa"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, color="#9a1f40", linewidth=2.5, marker="D", markersize=7, label="Group mean")
    plt.xlabel("Anthocyanin Concentration Group")
    plt.ylabel("Tensile Strength (MPa)")
    plt.title("Biomechanics of PhytoFiber Across Formulation Groups")
    if anova_p is not None:
        ax.text(0.02, 0.98, f"ANOVA p = {anova_p:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    plt.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_spoilage_regplot(df: pd.DataFrame, out_path: Path, threshold: float = 6.8, pearson_r: float | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(9, 6.5))
    ax = sns.regplot(
        data=df,
        x="G",
        y="meat_surface_ph",
        ci=95,
        scatter_kws={"s": 80, "alpha": 0.85, "color": "#244c5a", "edgecolor": "white", "linewidth": 0.6},
        line_kws={"linewidth": 2.8, "color": "#9a1f40"},
    )
    plt.xlabel("Fiber Green Channel Intensity")
    plt.ylabel("Meat Surface pH")
    plt.title("Real-World Spoilage Signal: Fiber Color vs Chicken pH")
    plt.axhline(threshold, linestyle="--", linewidth=1.4, color="#d34d4d", alpha=0.9, label=f"Spoilage threshold pH {threshold:.1f}")
    if pearson_r is not None:
        ax.text(0.02, 0.98, f"Pearson r = {pearson_r:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    plt.legend(frameon=False)
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_calibration_curve(df: pd.DataFrame, out_path: Path, prediction_col: str = "predicted_pH", r2: float | None = None) -> None:
    set_publication_style()
    work = df.sort_values("G")
    plt.figure(figsize=(9, 6.5))
    ax = plt.gca()
    ax.scatter(work["G"], work["pH"], s=110, color="#d17a22", edgecolors="white", linewidths=0.8, zorder=3, label="Measured pH cups")
    if prediction_col in work.columns:
        ax.plot(work["G"], work[prediction_col], color="#244c5a", linewidth=2.8, label="Polynomial fit (degree 2)")
    plt.xlabel("Fiber Green Channel Intensity")
    plt.ylabel("pH")
    plt.title("Colorimetric Calibration Curve for Anthocyanin Fiber")
    if r2 is not None:
        ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    plt.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_confusion_matrix_heatmap(cm_df: pd.DataFrame, title: str, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(6.5, 5.5))
    matrix = cm_df.set_index("actual").loc[["actual_safe", "actual_spoiled"], ["pred_safe", "pred_spoiled"]]
    ax = sns.heatmap(matrix, annot=True, fmt="g", cmap="YlOrBr", cbar=False, linewidths=1.0, linecolor="white", annot_kws={"fontsize": 14, "fontweight": "bold"})
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    sns.despine(ax=ax, left=True, bottom=True)
    _finish_figure(out_path)


def save_analysis_dashboard(
    spoilage_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_path: Path,
    pearson_r: float | None = None,
) -> None:
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax_left, ax_right = axes
    sns.regplot(
        data=spoilage_df,
        x="G",
        y="meat_surface_ph",
        ci=95,
        scatter_kws={"s": 70, "alpha": 0.8, "color": "#244c5a", "edgecolor": "white", "linewidth": 0.6},
        line_kws={"linewidth": 2.6, "color": "#9a1f40"},
        ax=ax_left,
    )
    ax_left.axhline(6.8, linestyle="--", linewidth=1.3, color="#d34d4d")
    ax_left.set_title("Chicken Spoilage Signal")
    ax_left.set_xlabel("Fiber G-channel")
    ax_left.set_ylabel("Meat pH")
    if pearson_r is not None:
        ax_left.text(0.03, 0.96, f"r = {pearson_r:.3f}", transform=ax_left.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})

    model_order = comparison_df.sort_values("accuracy", ascending=True)
    ax_right.barh(model_order["model_type"], model_order["accuracy"], color=["#7a9e48", "#d17a22"][: len(model_order)])
    for idx, (_, row) in enumerate(model_order.iterrows()):
        ax_right.text(row["accuracy"] + 0.01, idx, f"{row['accuracy']:.2%}", va="center", fontsize=11)
    ax_right.set_xlim(0, min(1.0, max(0.1, model_order["accuracy"].max() + 0.15)))
    ax_right.set_title("Classifier Accuracy")
    ax_right.set_xlabel("Cross-validated accuracy")
    ax_right.set_ylabel("")

    fig.suptitle("PhytoFiber Predictive Analysis Dashboard", fontsize=20, fontweight="bold", y=1.02)
    sns.despine(ax=ax_left)
    sns.despine(ax=ax_right)
    _finish_figure(out_path)

