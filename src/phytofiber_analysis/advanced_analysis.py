import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
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


def estimate_bayesian_group_posteriors(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    better_direction: str = "higher",
    draws: int = 20000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[group_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float)
    rng = np.random.default_rng(random_state)

    posterior_samples: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []
    groups = list(pd.Series(work[group_col]).dropna().astype(str).unique())
    for group_name in groups:
        subset = work.loc[work[group_col].astype(str) == group_name, value_col].to_numpy(dtype=float)
        n_obs = len(subset)
        sample_mean = float(np.mean(subset))
        sample_std = float(np.std(subset, ddof=1)) if n_obs > 1 else 0.0
        if n_obs > 1 and sample_std > 0:
            posterior = student_t.rvs(df=n_obs - 1, loc=sample_mean, scale=sample_std / np.sqrt(n_obs), size=draws, random_state=rng)
        else:
            posterior = np.full(draws, sample_mean, dtype=float)
        posterior_samples[group_name] = posterior

        lower, median, upper = np.quantile(posterior, [0.025, 0.5, 0.975])
        summary_rows.append(
            {
                "group": group_name,
                "n": int(n_obs),
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "posterior_mean": float(np.mean(posterior)),
                "posterior_median": float(median),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
            }
        )

    sample_matrix = np.vstack([posterior_samples[group_name] for group_name in groups])
    if better_direction == "higher":
        best_idx = np.argmax(sample_matrix, axis=0)
    elif better_direction == "lower":
        best_idx = np.argmin(sample_matrix, axis=0)
    else:
        raise ValueError("better_direction must be 'higher' or 'lower'.")

    probability_best = {group_name: float(np.mean(best_idx == idx)) for idx, group_name in enumerate(groups)}
    for row in summary_rows:
        row["probability_best"] = probability_best[row["group"]]

    pairwise_rows: list[dict] = []
    for row_group in groups:
        for col_group in groups:
            row_posterior = posterior_samples[row_group]
            col_posterior = posterior_samples[col_group]
            difference = row_posterior - col_posterior
            if row_group == col_group:
                probability_row_better = 0.5
            elif better_direction == "higher":
                probability_row_better = float(np.mean(difference > 0))
            else:
                probability_row_better = float(np.mean(difference < 0))
            diff_lower, diff_upper = np.quantile(difference, [0.025, 0.975])
            pairwise_rows.append(
                {
                    "row_group": row_group,
                    "col_group": col_group,
                    "probability_row_better": probability_row_better,
                    "mean_difference": float(np.mean(difference)),
                    "difference_ci_lower": float(diff_lower),
                    "difference_ci_upper": float(diff_upper),
                }
            )

    return pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True), pd.DataFrame(pairwise_rows)


def simulate_tensile_monte_carlo(
    df: pd.DataFrame,
    group_col: str = "group",
    value_col: str = "tensile_mpa",
    draws_per_group: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[group_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float)
    rng = np.random.default_rng(random_state)

    simulated_rows: list[dict] = []
    summary_rows: list[dict] = []
    for group_name, subset in work.groupby(group_col):
        group_values = subset[value_col].to_numpy(dtype=float)
        mean_value = float(np.mean(group_values))
        std_value = float(np.std(group_values, ddof=1)) if len(group_values) > 1 else 0.0
        effective_std = std_value if std_value > 0 else max(abs(mean_value) * 0.02, 0.01)
        simulated = rng.normal(loc=mean_value, scale=effective_std, size=draws_per_group)
        simulated = np.clip(simulated, a_min=0.01, a_max=None)

        summary_rows.append(
            {
                "group": str(group_name),
                "observed_n": int(len(group_values)),
                "simulated_n": int(draws_per_group),
                "observed_mean": mean_value,
                "observed_std": std_value,
                "simulated_mean": float(np.mean(simulated)),
                "simulated_std": float(np.std(simulated, ddof=1)),
            }
        )

        for idx, value in enumerate(simulated, start=1):
            simulated_rows.append(
                {
                    "group": str(group_name),
                    "simulation_id": idx,
                    value_col: float(value),
                    "source": "monte_carlo",
                }
            )

    return pd.DataFrame(simulated_rows), pd.DataFrame(summary_rows)


def simulate_spoilage_monte_carlo(
    df: pd.DataFrame,
    time_col: str = "time_h",
    ph_col: str = "meat_surface_ph",
    signal_col: str = "G",
    draws: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    work = df[[time_col, ph_col, signal_col]].dropna().copy()
    work[time_col] = work[time_col].astype(float)
    work[ph_col] = work[ph_col].astype(float)
    work[signal_col] = work[signal_col].astype(float)
    if len(work) < 3:
        raise ValueError("Monte Carlo spoilage simulation requires at least three observed rows.")

    observed = work[[time_col, ph_col, signal_col]].to_numpy(dtype=float)
    means = observed.mean(axis=0)
    covariance = np.cov(observed, rowvar=False)
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-6

    rng = np.random.default_rng(random_state)
    simulated = rng.multivariate_normal(mean=means, cov=covariance, size=draws)
    sim_df = pd.DataFrame(simulated, columns=[time_col, ph_col, signal_col])
    for column in [time_col, ph_col, signal_col]:
        lower = float(work[column].min())
        upper = float(work[column].max())
        sim_df[column] = sim_df[column].clip(lower=lower, upper=upper)

    sim_df["source"] = "monte_carlo"
    payload = {
        "draws": int(draws),
        "observed_n": int(len(work)),
        "means": {
            time_col: float(means[0]),
            ph_col: float(means[1]),
            signal_col: float(means[2]),
        },
        "covariance": covariance.round(6).tolist(),
    }
    return sim_df, payload


def fit_spoilage_response_surface(
    df: pd.DataFrame,
    time_col: str = "time_h",
    ph_col: str = "meat_surface_ph",
    signal_col: str = "G",
    grid_size: int = 60,
) -> tuple[dict, pd.DataFrame]:
    work = df[[time_col, ph_col, signal_col]].dropna().copy()
    work[time_col] = work[time_col].astype(float)
    work[ph_col] = work[ph_col].astype(float)
    work[signal_col] = work[signal_col].astype(float)
    if len(work) < 4:
        raise ValueError("Response surface fitting requires at least four observed rows.")

    X = work[[time_col, ph_col]].to_numpy(dtype=float)
    y = work[signal_col].to_numpy(dtype=float)
    features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    r_squared = float(model.score(X_poly, y))

    time_grid = np.linspace(work[time_col].min(), work[time_col].max(), grid_size)
    ph_grid = np.linspace(work[ph_col].min(), work[ph_col].max(), grid_size)
    mesh_time, mesh_ph = np.meshgrid(time_grid, ph_grid)
    grid_points = np.column_stack([mesh_time.ravel(), mesh_ph.ravel()])
    grid_signal = model.predict(features.transform(grid_points))
    surface_df = pd.DataFrame(
        {
            time_col: mesh_time.ravel(),
            ph_col: mesh_ph.ravel(),
            signal_col: grid_signal,
        }
    )
    payload = {
        "grid_size": int(grid_size),
        "r2": r_squared,
        "intercept": float(model.intercept_),
        "feature_names": features.get_feature_names_out([time_col, ph_col]).tolist(),
        "coefficients": model.coef_.astype(float).tolist(),
    }
    return payload, surface_df


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