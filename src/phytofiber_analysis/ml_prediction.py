import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy import stats


def prepare_spoilage_labels(
    df: pd.DataFrame,
    ph_col: str = "meat_surface_ph",
    threshold: float = 6.8,
) -> pd.DataFrame:
    out = df.copy()
    out["is_spoiled"] = (out[ph_col].astype(float) >= threshold).astype(int)
    out["target_spoiled"] = out["is_spoiled"]
    return out


def fit_polynomial_calibration(
    df: pd.DataFrame,
    feature_col: str = "G",
    target_col: str = "pH",
    degree: int = 2,
) -> tuple[dict, pd.DataFrame]:
    work = df[[feature_col, target_col]].dropna().copy()
    work[feature_col] = work[feature_col].astype(float)
    work[target_col] = work[target_col].astype(float)
    if len(work) < degree + 1:
        raise ValueError("Calibration curve needs enough samples for polynomial fitting.")

    X = work[[feature_col]].to_numpy()
    y = work[target_col].to_numpy()
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression()),
        ]
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    coefficients = model.named_steps["reg"].coef_.tolist()
    intercept = float(model.named_steps["reg"].intercept_)
    prediction_df = work.copy()
    prediction_df["predicted_pH"] = y_pred
    prediction_df["residual"] = prediction_df[target_col] - prediction_df["predicted_pH"]

    payload = {
        "feature_col": feature_col,
        "target_col": target_col,
        "degree": degree,
        "r2": float(score),
        "intercept": intercept,
        "coefficients": coefficients,
    }
    return payload, prediction_df


def run_pearson_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict:
    work = df[[x_col, y_col]].dropna().copy()
    work[x_col] = work[x_col].astype(float)
    work[y_col] = work[y_col].astype(float)
    if len(work) < 3:
        raise ValueError("Pearson correlation needs at least three matched observations.")

    r_value, p_value = stats.pearsonr(work[x_col], work[y_col])
    return {
        "x_col": x_col,
        "y_col": y_col,
        "n_samples": int(len(work)),
        "pearson_r": float(r_value),
        "p_value": float(p_value),
    }


def evaluate_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_spoiled",
    model_type: str = "logistic",
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    work = df.dropna(subset=feature_cols + [target_col]).copy()
    X = work[feature_cols].astype(float).values
    y = work[target_col].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes (safe/spoiled) to train classifier.")

    min_class_size = int(np.bincount(y).min())
    n_splits = max(2, min(5, min_class_size))

    if model_type == "logistic":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    else:
        raise ValueError("model_type must be 'logistic' or 'random_forest'.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    model.fit(X, y)

    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, y_pred)

    roc_auc = float("nan")
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        roc_auc = roc_auc_score(y, y_proba)

    metrics = {
        "model_type": model_type,
        "features": feature_cols,
        "random_state": random_state,
        "cv_folds": n_splits,
        "n_samples": int(len(work)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,
    }

    cm_df = pd.DataFrame(
        cm,
        index=["actual_safe", "actual_spoiled"],
        columns=["pred_safe", "pred_spoiled"],
    ).reset_index(names="actual")
    prediction_df = work.copy().reset_index(drop=True)
    prediction_df["model_type"] = model_type
    prediction_df["y_true"] = y
    prediction_df["y_pred"] = y_pred
    prediction_df["y_proba"] = y_proba if y_proba is not None else np.nan
    return metrics, cm_df, prediction_df

