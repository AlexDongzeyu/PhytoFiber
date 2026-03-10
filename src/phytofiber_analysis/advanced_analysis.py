import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _four_param_logistic(x: np.ndarray, bottom: float, top: float, midpoint: float, slope: float) -> np.ndarray:
    return bottom + (top - bottom) / (1.0 + np.exp(-slope * (x - midpoint)))


def fit_weibull_reliability(
    df: pd.DataFrame,
    group_col: str = "group",
    value_col: str = "tensile_mpa",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[group_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float)

    summary_rows = []
    plot_rows = []
    for group_name, subset in work.groupby(group_col):
        sorted_values = np.sort(subset[value_col].to_numpy(dtype=float))
        n_obs = len(sorted_values)
        if n_obs < 3:
            continue

        ranks = np.arange(1, n_obs + 1)
        failure_probability = (ranks - 0.3) / (n_obs + 0.4)
        ln_stress = np.log(sorted_values)
        weibull_y = np.log(-np.log(1.0 - failure_probability))

        slope, intercept = np.polyfit(ln_stress, weibull_y, 1)
        fitted = slope * ln_stress + intercept
        ss_res = float(np.sum((weibull_y - fitted) ** 2))
        ss_tot = float(np.sum((weibull_y - weibull_y.mean()) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot else float("nan")
        characteristic_strength = float(np.exp(-intercept / slope)) if slope else float("nan")

        summary_rows.append(
            {
                "group": str(group_name),
                "n": int(n_obs),
                "weibull_modulus": float(slope),
                "characteristic_strength_mpa": characteristic_strength,
                "intercept": float(intercept),
                "r_squared": r_squared,
            }
        )

        for stress_value, probability_value, x_value, y_value, fitted_value in zip(
            sorted_values,
            failure_probability,
            ln_stress,
            weibull_y,
            fitted,
        ):
            plot_rows.append(
                {
                    "group": str(group_name),
                    "stress_mpa": float(stress_value),
                    "failure_probability": float(probability_value),
                    "ln_stress": float(x_value),
                    "weibull_y": float(y_value),
                    "fitted_weibull_y": float(fitted_value),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(plot_rows)


def fit_4pl_calibration(
    df: pd.DataFrame,
    feature_col: str = "G",
    target_col: str = "pH",
) -> tuple[dict, pd.DataFrame]:
    work = df[[feature_col, target_col]].dropna().copy()
    work[feature_col] = work[feature_col].astype(float)
    work[target_col] = work[target_col].astype(float)
    if len(work) < 4:
        raise ValueError("4PL calibration requires at least four observations.")

    x_data = work[feature_col].to_numpy(dtype=float)
    y_data = work[target_col].to_numpy(dtype=float)
    initial = [float(y_data.min()), float(y_data.max()), float(np.median(x_data)), 0.18]
    bounds = ([0.0, 0.0, float(x_data.min()) - 20.0, 0.001], [14.0, 14.0, float(x_data.max()) + 20.0, 5.0])

    params, _ = curve_fit(
        _four_param_logistic,
        x_data,
        y_data,
        p0=initial,
        bounds=bounds,
        maxfev=20000,
    )
    y_pred = _four_param_logistic(x_data, *params)
    residuals = y_data - y_pred
    ss_res = float(np.sum((y_data - y_pred) ** 2))
    ss_tot = float(np.sum((y_data - y_data.mean()) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot else float("nan")

    prediction_df = work.copy()
    prediction_df["predicted_pH_4pl"] = y_pred
    prediction_df["residual_4pl"] = residuals

    payload = {
        "feature_col": feature_col,
        "target_col": target_col,
        "model_type": "4pl",
        "bottom": float(params[0]),
        "top": float(params[1]),
        "midpoint": float(params[2]),
        "slope": float(params[3]),
        "r2": float(r_squared),
    }
    return payload, prediction_df


def apply_4pl_calibration(df: pd.DataFrame, model_payload: dict, feature_col: str = "G", out_col: str = "predicted_pH_4pl") -> pd.DataFrame:
    work = df.copy()
    x_data = work[feature_col].astype(float).to_numpy(dtype=float)
    params = (
        float(model_payload["bottom"]),
        float(model_payload["top"]),
        float(model_payload["midpoint"]),
        float(model_payload["slope"]),
    )
    work[out_col] = _four_param_logistic(x_data, *params)
    return work


def compute_bland_altman(
    df: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
) -> tuple[dict, pd.DataFrame]:
    work = df[[reference_col, candidate_col]].dropna().copy()
    work[reference_col] = work[reference_col].astype(float)
    work[candidate_col] = work[candidate_col].astype(float)
    if len(work) < 3:
        raise ValueError("Bland-Altman analysis requires at least three paired observations.")

    work["mean_of_methods"] = (work[reference_col] + work[candidate_col]) / 2.0
    work["difference"] = work[candidate_col] - work[reference_col]
    bias = float(work["difference"].mean())
    sd_diff = float(work["difference"].std(ddof=1))
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff
    payload = {
        "reference_col": reference_col,
        "candidate_col": candidate_col,
        "n_samples": int(len(work)),
        "bias": bias,
        "sd_difference": sd_diff,
        "loa_upper": float(loa_upper),
        "loa_lower": float(loa_lower),
    }
    return payload, work


def fit_svm_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_spoiled",
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame, Pipeline]:
    work = df.dropna(subset=feature_cols + [target_col]).copy()
    X = work[feature_cols].astype(float).to_numpy()
    y = work[target_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("SVM classification requires both safe and spoiled classes.")

    min_class_size = int(np.bincount(y).min())
    n_splits = max(2, min(3, min_class_size))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=random_state)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    pipeline.fit(X, y)

    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)
    metrics = {
        "model_type": "svm_rbf",
        "features": feature_cols,
        "cv_folds": int(n_splits),
        "n_samples": int(len(work)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

    predictions = work.copy().reset_index(drop=True)
    predictions["y_true"] = y
    predictions["y_pred_svm"] = y_pred
    predictions["y_proba_svm"] = y_proba
    return metrics, predictions, pipeline


def summarize_latency(df: pd.DataFrame, group_col: str = "group", value_col: str = "response_time_s") -> pd.DataFrame:
    work = df[[group_col, value_col]].dropna().copy()
    out = work.groupby(group_col)[value_col].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    out = out.rename(columns={group_col: "group", "count": "n"})
    out["sem"] = out["std"] / np.sqrt(out["n"])
    return out


def build_formulation_radar_scores(
    tensile_summary: pd.DataFrame,
    latency_summary: pd.DataFrame,
    weibull_summary: pd.DataFrame,
) -> pd.DataFrame:
    tensile = tensile_summary[["group", "mean"]].rename(columns={"mean": "strength_mpa"}).copy()
    latency = latency_summary[["group", "mean"]].rename(columns={"mean": "response_time_s"}).copy()
    latency["group"] = latency["group"].astype(str).str.split("_").str[0]
    weibull = weibull_summary[["group", "weibull_modulus"]].copy()

    merged = tensile.merge(latency, on="group", how="inner").merge(weibull, on="group", how="inner")
    if merged.empty:
        raise ValueError("Formulation radar scores require shared formulation groups across inputs.")

    merged["speed_inverse"] = 1.0 / merged["response_time_s"].astype(float)
    for raw_col, score_col in [
        ("strength_mpa", "strength_score"),
        ("speed_inverse", "speed_score"),
        ("weibull_modulus", "reliability_score"),
    ]:
        raw_values = merged[raw_col].astype(float)
        min_value = float(raw_values.min())
        max_value = float(raw_values.max())
        if np.isclose(min_value, max_value):
            merged[score_col] = 100.0
        else:
            merged[score_col] = 100.0 * (raw_values - min_value) / (max_value - min_value)
    return merged.sort_values("group").reset_index(drop=True)