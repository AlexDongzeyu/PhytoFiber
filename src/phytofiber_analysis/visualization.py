from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import auc, roc_curve


GROUP_PALETTE = ["#244c5a", "#d17a22", "#7a9e48", "#b64d6b"]
SAFE_COLOR = "#2f7d32"
SAFE_FILL = "#dfeedd"
SPOILED_COLOR = "#b53737"
SPOILED_FILL = "#f7d9d3"
DISCLOSURE_TEXT = "In-Silico Visualization: Augmented via Monte Carlo (n=1000) based on physical sample variance (n=15)."


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(
        context="talk",
        font_scale=1.28,
        style="whitegrid",
        rc={
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "figure.facecolor": "#f5f1e8",
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#2b2b2b",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "xtick.color": "#4a4a4a",
            "ytick.color": "#4a4a4a",
            "grid.color": "#ddd6ca",
            "axes.titleweight": "bold",
            "font.size": 13,
            "axes.titlepad": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
            "grid.linewidth": 0.8,
        },
    )


def _add_axis_note(ax, note: str, *, x: float = 0.02, y: float = 0.02, fontsize: float = 10.5) -> None:
    ax.text(
        x,
        y,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontstyle="italic",
        color="#3d342d",
        bbox={"facecolor": "#fffaf2", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35", "alpha": 0.96},
    )


def _add_figure_note(fig, note: str, *, y: float = 0.02, fontsize: float = 10.5) -> None:
    fig.text(
        0.02,
        y,
        note,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontstyle="italic",
        color="#3d342d",
        bbox={"facecolor": "#fffaf2", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35", "alpha": 0.96},
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
    palette = GROUP_PALETTE[: len(order)]
    ax = sns.boxplot(data=df, x="group", y="tensile_mpa", hue="group", width=0.55, order=order, palette=palette, linewidth=1.5, dodge=False, legend=False)
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


def save_tensile_violinplot(df: pd.DataFrame, out_path: Path, anova_p: float | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(9.4, 6.8))
    order = list(pd.Series(df["group"]).dropna().astype(str).unique())
    palette = GROUP_PALETTE[: len(order)]
    ax = sns.violinplot(data=df, x="group", y="tensile_mpa", hue="group", order=order, palette=palette, inner=None, linewidth=1.4, cut=0, legend=False)
    sns.swarmplot(data=df, x="group", y="tensile_mpa", order=order, color="#111111", size=5.6, alpha=0.9)
    means = df.groupby("group")["tensile_mpa"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, color="#9a1f40", linewidth=2.2, marker="D", markersize=6)
    ax.set_title("Tensile Strength Distribution and Replicate Density")
    ax.set_xlabel("Anthocyanin Concentration Group")
    ax.set_ylabel("Tensile Strength (MPa)")
    if anova_p is not None:
        ax.text(0.02, 0.98, f"ANOVA p = {anova_p:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_bayesian_forest_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    set_publication_style()
    work = df.sort_values("posterior_median", ascending=True).reset_index(drop=True)
    plt.figure(figsize=(9.8, 6.6))
    ax = plt.gca()
    palette = sns.color_palette(["#1f4e5f", "#b46a28", "#6a8f3f", "#a3485b"], n_colors=len(work))
    for idx, row in enumerate(work.itertuples(index=False)):
        color = palette[idx]
        ax.hlines(idx, row.ci_lower, row.ci_upper, color=color, linewidth=3.4, alpha=0.95)
        ax.scatter(row.posterior_median, idx, s=160, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(
            row.ci_upper + (work["ci_upper"].max() - work["ci_lower"].min()) * 0.02,
            idx,
            f"P(best) = {row.probability_best:.2%}",
            va="center",
            fontsize=10.5,
            color="#303030",
        )
    ax.set_yticks(range(len(work)))
    ax.set_yticklabels([f"Group {group}" for group in work["group"]])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.45)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=False)
    _finish_figure(out_path)


def save_superiority_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    better_label: str,
) -> None:
    set_publication_style()
    matrix = df.pivot(index="row_group", columns="col_group", values="probability_row_better")
    labels = matrix.apply(lambda column: column.map(lambda value: f"{value:.0%}"))
    plt.figure(figsize=(7.4, 6.4))
    ax = sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap=sns.color_palette(["#8b2e3c", "#f6f2ea", "#2f6f4f"], as_cmap=True),
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": better_label, "shrink": 0.85},
    )
    ax.set_title(title)
    ax.set_xlabel("Compared Against")
    ax.set_ylabel("Reference Group")
    _finish_figure(out_path)


def save_spoilage_regplot(df: pd.DataFrame, out_path: Path, threshold: float = 6.8, pearson_r: float | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(9, 6.5))
    ax = sns.regplot(
        data=df,
        x="G",
        y="meat_surface_ph",
        ci=95,
        scatter_kws={"s": 80, "alpha": 0.85, "color": "#244c5a", "edgecolor": "white", "linewidths": 0.6},
        line_kws={"linewidth": 2.8, "color": "#9a1f40"},
    )
    y_min = float(df["meat_surface_ph"].min()) - 0.1
    y_max = float(df["meat_surface_ph"].max()) + 0.1
    ax.axhspan(y_min, threshold, color=SAFE_FILL, alpha=0.45, zorder=0)
    ax.axhspan(threshold, y_max, color=SPOILED_FILL, alpha=0.38, zorder=0)
    plt.xlabel("Fiber Green Channel Intensity")
    plt.ylabel("Meat Surface pH")
    plt.title("Real-World Spoilage Signal: Fiber Color vs Chicken pH")
    plt.axhline(threshold, linestyle="--", linewidth=1.6, color=SPOILED_COLOR, alpha=0.9, label=f"Spoilage threshold pH {threshold:.1f}")
    ax.text(0.98, 0.10, "Safe zone", transform=ax.transAxes, ha="right", va="bottom", color=SAFE_COLOR, fontsize=11.5, fontweight="bold")
    ax.text(0.98, 0.92, "Spoilage zone", transform=ax.transAxes, ha="right", va="top", color=SPOILED_COLOR, fontsize=11.5, fontweight="bold")
    if pearson_r is not None:
        ax.text(0.02, 0.98, f"Pearson r = {pearson_r:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    plt.legend(frameon=False)
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_calibration_curve(df: pd.DataFrame, out_path: Path, prediction_col: str = "predicted_pH", r2: float | None = None) -> None:
    set_publication_style()
    plt.figure(figsize=(9, 6.5))
    work = df.sort_values("G")
    ax = plt.gca()
    ax.scatter(work["G"], work["pH"], s=110, alpha=0.92, color="#d17a22", edgecolor="white", linewidths=0.8)
    if prediction_col in work.columns:
        ax.plot(work["G"], work[prediction_col], linewidth=2.8, color="#244c5a")
    plt.xlabel("Fiber Green Channel Intensity")
    plt.ylabel("pH")
    plt.title("Colorimetric Calibration Curve for Anthocyanin Fiber")
    if r2 is not None:
        ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_confusion_matrix_heatmap(cm_df: pd.DataFrame, title: str, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(6.5, 5.5))
    matrix = cm_df.set_index("actual").loc[["actual_safe", "actual_spoiled"], ["pred_safe", "pred_spoiled"]]
    cmap = LinearSegmentedColormap.from_list("safe_spoilage", ["#fffaf2", SAFE_FILL, SAFE_COLOR])
    ax = sns.heatmap(matrix, annot=True, fmt="g", cmap=cmap, cbar=False, linewidths=1.0, linecolor="white", annot_kws={"fontsize": 14, "fontweight": "bold"})
    plt.title(title)
    plt.xlabel("Predicted Class (safe shown first)")
    plt.ylabel("Actual Class (safe shown first)")
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
        scatter_kws={"s": 70, "alpha": 0.8, "color": "#244c5a", "edgecolor": "white", "linewidths": 0.6},
        line_kws={"linewidth": 2.6, "color": "#9a1f40"},
        ax=ax_left,
    )
    left_y_min = float(spoilage_df["meat_surface_ph"].min()) - 0.1
    left_y_max = float(spoilage_df["meat_surface_ph"].max()) + 0.1
    ax_left.axhspan(left_y_min, 6.8, color=SAFE_FILL, alpha=0.45, zorder=0)
    ax_left.axhspan(6.8, left_y_max, color=SPOILED_FILL, alpha=0.38, zorder=0)
    ax_left.axhline(6.8, linestyle="--", linewidth=1.4, color=SPOILED_COLOR)
    ax_left.set_title("Chicken Spoilage Signal")
    ax_left.set_xlabel("Fiber G-channel")
    ax_left.set_ylabel("Meat pH")
    ax_left.text(0.98, 0.10, "Safe zone", transform=ax_left.transAxes, ha="right", va="bottom", color=SAFE_COLOR, fontsize=10.8, fontweight="bold")
    ax_left.text(0.98, 0.92, "Spoilage zone", transform=ax_left.transAxes, ha="right", va="top", color=SPOILED_COLOR, fontsize=10.8, fontweight="bold")
    if pearson_r is not None:
        ax_left.text(0.03, 0.96, f"r = {pearson_r:.3f}", transform=ax_left.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})

    model_order = comparison_df.sort_values("accuracy", ascending=True)
    ax_right.barh(model_order["model_type"], model_order["accuracy"], color=[SAFE_COLOR, "#d17a22"][: len(model_order)])
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


def save_roc_curve(predictions_df: pd.DataFrame, out_path: Path, model_type: str = "logistic") -> None:
    work = predictions_df[(predictions_df["model_type"] == model_type) & predictions_df["y_proba"].notna()].copy()
    if work.empty:
        raise ValueError("ROC curve requires classifier probabilities.")
    fpr, tpr, _ = roc_curve(work["y_true"].astype(int), work["y_proba"].astype(float))
    roc_auc = auc(fpr, tpr)
    set_publication_style()
    plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()
    ax.plot(fpr, tpr, color="#9a1f40", linewidth=3, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#5d5d5d", linewidth=1.4, label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve for Spoilage Classifier")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_dual_axis_spoilage_plot(df: pd.DataFrame, out_path: Path) -> None:
    required = {"time_h", "meat_surface_ph", "G"}
    if not required.issubset(df.columns):
        raise ValueError("Dual-axis spoilage plot requires time_h, meat_surface_ph, and G columns.")
    work = df[list(required)].dropna().copy()
    if work.empty:
        raise ValueError("Dual-axis spoilage plot requires non-empty matched spoilage data.")
    summary = work.groupby("time_h", as_index=False).agg({"meat_surface_ph": "mean", "G": "mean"}).sort_values("time_h")
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    ax2 = ax1.twinx()
    ax1.plot(summary["time_h"], summary["meat_surface_ph"], color="#b53737", linewidth=2.8, marker="o", markersize=7)
    ax2.plot(summary["time_h"], summary["G"], color="#2f7d32", linewidth=2.8, marker="s", markersize=6)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Meat Surface pH", color="#b53737")
    ax2.set_ylabel("Fiber G-Channel Intensity", color="#2f7d32")
    ax1.tick_params(axis="y", colors="#b53737")
    ax2.tick_params(axis="y", colors="#2f7d32")
    ax1.set_title("Dual-Axis Spoilage Kinetics: Meat pH vs Fiber Color")
    sns.despine(ax=ax1, right=False)
    _finish_figure(out_path)


def save_correlation_heatmap(df: pd.DataFrame, out_path: Path, cols: list[str]) -> None:
    work = df[cols].dropna().copy()
    if work.empty:
        raise ValueError("Correlation heatmap requires non-empty numeric data.")
    corr = work.corr(numeric_only=True)
    set_publication_style()
    plt.figure(figsize=(7.6, 6.3))
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=1.0, linecolor="white", cbar_kws={"shrink": 0.85})
    ax.set_title("Pearson Correlation Heatmap")
    _finish_figure(out_path)


def save_raincloud_plot(
    df: pd.DataFrame,
    out_path: Path,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    disclosure_note: str | None = None,
) -> None:
    set_publication_style()
    plt.figure(figsize=(10.5, 6.8))
    order = list(pd.Series(df[group_col]).dropna().astype(str).unique())
    palette = GROUP_PALETTE[: len(order)]
    ax = plt.gca()
    sns.violinplot(
        data=df,
        y=group_col,
        x=value_col,
        order=order,
        hue=group_col,
        inner=None,
        cut=0,
        linewidth=1.3,
        palette=palette,
        orient="h",
        legend=False,
        ax=ax,
    )
    violin_bodies = [artist for artist in ax.collections if isinstance(artist, PolyCollection)][: len(order)]
    for violin in violin_bodies:
        for path in violin.get_paths():
            vertices = path.vertices
            center = np.mean(vertices[:, 1])
            vertices[vertices[:, 1] > center, 1] = center

    sns.boxplot(
        data=df,
        y=group_col,
        x=value_col,
        order=order,
        width=0.16,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "#fffaf2", "edgecolor": "#2b2521", "linewidth": 1.2, "zorder": 3},
        medianprops={"color": "#9a1f40", "linewidth": 2.1},
        whiskerprops={"color": "#2b2521", "linewidth": 1.1},
        capprops={"color": "#2b2521", "linewidth": 1.1},
        orient="h",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        y=group_col,
        x=value_col,
        order=order,
        color="#111111",
        alpha=0.85,
        size=5.5,
        jitter=0.17,
        orient="h",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    if disclosure_note:
        _add_axis_note(ax, disclosure_note)
    sns.despine(ax=ax, left=False)
    _finish_figure(out_path)


def save_weibull_probability_plot(plot_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(10.2, 7.0))
    ax = plt.gca()
    palette = ["#244c5a", "#d17a22", "#7a9e48", "#b64d6b"]
    for idx, group_name in enumerate(summary_df["group"]):
        group_points = plot_df[plot_df["group"] == group_name].sort_values("ln_stress")
        color = palette[idx % len(palette)]
        ax.scatter(group_points["ln_stress"], group_points["weibull_y"], s=70, color=color, edgecolor="white", linewidth=0.7, label=f"Group {group_name} data")
        ax.plot(group_points["ln_stress"], group_points["fitted_weibull_y"], color=color, linewidth=2.4)

    summary_lines = [
        f"{row.group}: m = {row.weibull_modulus:.2f}, η = {row.characteristic_strength_mpa:.2f} MPa"
        for row in summary_df.itertuples(index=False)
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.4"},
    )
    ax.set_title("Weibull Probability Plot for Fiber Failure Reliability")
    ax.set_xlabel("ln(Tensile Stress, MPa)")
    ax.set_ylabel("ln(-ln(1 - P))")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_4pl_calibration_plot(df: pd.DataFrame, out_path: Path, model_payload: dict, feature_col: str = "G", target_col: str = "pH") -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.3), gridspec_kw={"height_ratios": [3.3, 1.2]}, sharex=True)
    ax_main, ax_resid = axes
    work = df.sort_values(feature_col).copy()
    x_dense = np.linspace(work[feature_col].min() - 2, work[feature_col].max() + 2, 300)
    y_dense = model_payload["bottom"] + (model_payload["top"] - model_payload["bottom"]) / (1.0 + np.exp(-model_payload["slope"] * (x_dense - model_payload["midpoint"])))

    ax_main.scatter(work[feature_col], work[target_col], s=100, color="#d17a22", edgecolor="white", linewidth=0.8, zorder=3)
    ax_main.plot(x_dense, y_dense, color="#244c5a", linewidth=2.8)
    ax_main.fill_between(x_dense, y_dense - work["residual_4pl"].std(ddof=1), y_dense + work["residual_4pl"].std(ddof=1), color="#244c5a", alpha=0.12)
    ax_main.set_ylabel("pH")
    ax_main.set_title("4-Parameter Logistic Calibration Curve with Residual Diagnostics")
    ax_main.text(0.03, 0.95, f"4PL R² = {model_payload['r2']:.3f}", transform=ax_main.transAxes, ha="left", va="top", fontsize=11, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})

    ax_resid.axhline(0, color="#5d5d5d", linewidth=1.2, linestyle="--")
    ax_resid.scatter(work[feature_col], work["residual_4pl"], s=75, color="#9a1f40", edgecolor="white", linewidth=0.7)
    ax_resid.set_xlabel("Fiber Green Channel Intensity")
    ax_resid.set_ylabel("Residual")
    sns.despine(ax=ax_main)
    sns.despine(ax=ax_resid)
    _finish_figure(out_path)


def save_bland_altman_plot(df: pd.DataFrame, out_path: Path, stats_payload: dict) -> None:
    set_publication_style()
    plt.figure(figsize=(9.2, 6.6))
    ax = plt.gca()
    ax.scatter(df["mean_of_methods"], df["difference"], s=90, color="#244c5a", edgecolor="white", linewidth=0.8)
    ax.axhline(stats_payload["bias"], color="#9a1f40", linewidth=2.5, label=f"Bias = {stats_payload['bias']:.2f}")
    ax.axhline(stats_payload["loa_upper"], color="#d17a22", linewidth=1.8, linestyle="--", label=f"Upper LoA = {stats_payload['loa_upper']:.2f}")
    ax.axhline(stats_payload["loa_lower"], color="#d17a22", linewidth=1.8, linestyle="--", label=f"Lower LoA = {stats_payload['loa_lower']:.2f}")
    ax.set_title("Bland-Altman Agreement Between Fiber Prediction and Meat pH")
    ax.set_xlabel("Mean of Fiber-Predicted and Measured pH")
    ax.set_ylabel("Difference (Predicted - Measured)")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_svm_decision_surface(
    df: pd.DataFrame,
    model,
    out_path: Path,
    feature_cols: list[str],
    target_col: str = "target_spoiled",
) -> None:
    set_publication_style()
    plt.figure(figsize=(9.5, 6.8))
    ax = plt.gca()
    work = df.dropna(subset=feature_cols + [target_col]).copy()
    x_min, x_max = work[feature_cols[0]].min() - 2.0, work[feature_cols[0]].max() + 2.0
    y_min, y_max = work[feature_cols[1]].min() - 3.0, work[feature_cols[1]].max() + 3.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    grid = pd.DataFrame({feature_cols[0]: xx.ravel(), feature_cols[1]: yy.ravel()})
    zz = model.predict_proba(grid[feature_cols].to_numpy())[:, 1].reshape(xx.shape)

    contour_cmap = LinearSegmentedColormap.from_list("safe_to_spoiled", [SAFE_FILL, SAFE_COLOR, "#f1cfac", SPOILED_COLOR])
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=contour_cmap, alpha=0.55)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["#9a1f40"], linewidths=2.4)
    scatter = ax.scatter(
        work[feature_cols[0]],
        work[feature_cols[1]],
        c=work[target_col].astype(int),
        cmap=ListedColormap([SAFE_COLOR, SPOILED_COLOR]),
        s=110,
        edgecolor="white",
        linewidth=0.9,
    )
    if "time_h" in work.columns:
        for row in work.itertuples(index=False):
            ax.annotate(f"{int(getattr(row, 'time_h'))}h", (getattr(row, feature_cols[0]), getattr(row, feature_cols[1])), xytext=(6, 5), textcoords="offset points", fontsize=9.5)
    ax.set_title("SVM Decision Surface for Chicken Safety Classification")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Fiber Green Channel Intensity")
    legend1 = ax.legend(*scatter.legend_elements(), title="Class", labels=["Safe", "Spoiled"], frameon=False, loc="upper left")
    ax.add_artist(legend1)
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_latency_barplot(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(8.8, 6.3))
    ax = plt.gca()
    order = list(df["group"].astype(str))
    ax.bar(order, df["mean"], yerr=df["std"], color=GROUP_PALETTE[: len(df)], capsize=8, alpha=0.92)
    ax.set_title("Halochromic Latency by Formulation")
    ax.set_xlabel("Formulation Group")
    ax.set_ylabel("Response Time (s)")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_stability_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    plt.figure(figsize=(9.4, 6.4))
    ax = plt.gca()
    for treatment, color in [("control", "#b53737"), ("treated", "#2f7d32")]:
        subset = df[df["treatment"].astype(str).str.lower() == treatment].sort_values("time_h")
        ax.plot(subset["time_h"], subset["color_retention_pct"], marker="o", linewidth=2.8, markersize=7, color=color, label=treatment.title())
    ax.set_title("Color Retention Stability Over Time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Color Retention (%)")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_digestibility_bars(df: pd.DataFrame, out_path: Path) -> None:
    set_publication_style()
    summary = df.groupby("phase", as_index=False)["mass_loss_pct"].mean()
    plt.figure(figsize=(8.4, 6.2))
    ax = plt.gca()
    ax.bar(summary["phase"], summary["mass_loss_pct"], color=["#244c5a", "#d17a22"], alpha=0.93)
    ax.set_title("In-Vitro Digestibility by Simulated GI Phase")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mass Loss (%)")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_economics_breakdown(df: pd.DataFrame, out_path: Path, cost_per_meter: float) -> None:
    set_publication_style()
    plt.figure(figsize=(9.6, 6.5))
    ax = plt.gca()
    ingredients = df[~df["ingredient"].isin(["total_fiber_length_m", "final_cost_per_meter"])].copy()
    ax.barh(ingredients["ingredient"], ingredients["cost_usd"], color="#7a9e48", alpha=0.92)
    for idx, row in enumerate(ingredients.itertuples(index=False)):
        label = f"${row.cost_usd:.4f}" if row.cost_usd < 0.01 else f"${row.cost_usd:.2f}"
        ax.text(row.cost_usd + max(ingredients["cost_usd"].max() * 0.03, 0.0002), idx, label, va="center", fontsize=10.5)
    ax.text(0.98, 0.06, f"Estimated production cost = ${cost_per_meter:.4f}/m", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, bbox={"facecolor": "#fff6df", "edgecolor": "#d8cfc2", "boxstyle": "round,pad=0.35"})
    ax.set_title("Economic Viability Breakdown")
    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_formulation_radar(df: pd.DataFrame, out_path: Path) -> None:
    categories = ["strength_score", "speed_score", "reliability_score"]
    labels = ["Strength", "Speed", "Reliability"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.6, 8.0), subplot_kw={"projection": "polar"})
    palette = GROUP_PALETTE
    for idx, row in enumerate(df.itertuples(index=False)):
        values = [float(getattr(row, category)) for category in categories]
        values += values[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2.5, label=f"Group {row.group}")
        ax.fill(angles, values, color=color, alpha=0.13)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"])
    ax.set_ylim(0, 100)
    ax.set_title("Formulation Optimization Radar", pad=24)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.16, 1.12))
    _finish_figure(out_path)


def save_spoilage_density_cloud(
    simulated_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    out_path: Path,
    ph_col: str = "meat_surface_ph",
    signal_col: str = "G",
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.4, 7.1))
    sns.kdeplot(
        data=simulated_df,
        x=ph_col,
        y=signal_col,
        fill=True,
        levels=16,
        thresh=0.02,
        cmap=sns.blend_palette(["#f3e6c8", "#d9b26f", "#a86736", "#5b3421"], as_cmap=True),
        ax=ax,
    )
    ax.scatter(
        observed_df[ph_col],
        observed_df[signal_col],
        s=95,
        color="#244c5a",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        label=f"Observed data (n={len(observed_df)})",
    )
    ax.set_title(f"In-Silico Spoilage Density Cloud (Monte Carlo n={len(simulated_df)})")
    ax.set_xlabel("Measured Meat Surface pH")
    ax.set_ylabel("Fiber Green-Channel Intensity")
    ax.legend(frameon=False, loc="upper left")
    _add_axis_note(ax, DISCLOSURE_TEXT)
    sns.despine(ax=ax)
    _finish_figure(out_path)


def save_spoilage_response_surface(
    surface_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    out_path: Path,
    time_col: str = "time_h",
    ph_col: str = "meat_surface_ph",
    signal_col: str = "G",
    r2: float | None = None,
) -> None:
    set_publication_style()
    fig = plt.figure(figsize=(10.4, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    pivot = surface_df.pivot(index=ph_col, columns=time_col, values=signal_col).sort_index().sort_index(axis=1)
    time_mesh, ph_mesh = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    signal_mesh = pivot.to_numpy(dtype=float)

    surf = ax.plot_surface(time_mesh, ph_mesh, signal_mesh, cmap="YlGnBu", edgecolor="none", alpha=0.82)
    ax.scatter(
        observed_df[time_col].astype(float),
        observed_df[ph_col].astype(float),
        observed_df[signal_col].astype(float),
        s=80,
        color="#9a1f40",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )
    title = "3D Spoilage Response Surface"
    if r2 is not None:
        title = f"{title} (surface fit R² = {r2:.3f})"
    ax.set_title(title, pad=18)
    ax.set_xlabel("Time (hours)", labelpad=10)
    ax.set_ylabel("Measured Meat Surface pH", labelpad=10)
    ax.set_zlabel("Fiber Green-Channel Intensity", labelpad=10)
    ax.view_init(elev=28, azim=132)
    fig.colorbar(surf, shrink=0.68, aspect=16, pad=0.08, label="Predicted G-channel")
    _add_figure_note(fig, DISCLOSURE_TEXT)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)

