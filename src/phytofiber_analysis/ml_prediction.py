import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_spoilage_labels(
    df: pd.DataFrame,
    ph_col: str = "meat_surface_ph",
    threshold: float = 6.5,
) -> pd.DataFrame:
    out = df.copy()
    out["target_spoiled"] = (out[ph_col].astype(float) >= threshold).astype(int)
    return out


def evaluate_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_spoiled",
    model_type: str = "logistic",
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    work = df.dropna(subset=feature_cols + [target_col]).copy()
    X = work[feature_cols].astype(float).values
    y = work[target_col].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes (safe/spoiled) to train classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

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

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_type": model_type,
        "test_size": test_size,
        "random_state": random_state,
        "n_samples": int(len(work)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    cm_df = pd.DataFrame(
        cm,
        index=["actual_safe", "actual_spoiled"],
        columns=["pred_safe", "pred_spoiled"],
    ).reset_index(names="actual")
    return metrics, cm_df

