from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"


def _normalize_prediction(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"safe", "s"}:
        return "Safe"
    if text in {"spoiled", "spoil", "spoilt", "bad", "sp"}:
        return "Spoiled"
    return "Unknown"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / "spoilage_validation.csv")

    required = ["Sample", "Container", "Time_h", "Blinded_Prediction", "Actual_Meat_pH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in spoilage_validation.csv: {missing}")

    df["Actual_Meat_pH"] = df["Actual_Meat_pH"].astype(float)
    df["Actual_Status"] = df["Actual_Meat_pH"].apply(lambda x: "Spoiled" if x >= 6.8 else "Safe")
    df["Predicted_Status"] = df["Blinded_Prediction"].map(_normalize_prediction)

    unknown = df[df["Predicted_Status"] == "Unknown"]
    if not unknown.empty:
        bad_samples = ", ".join(unknown["Sample"].astype(str).tolist())
        raise ValueError(f"Unknown blinded predictions found for samples: {bad_samples}")

    labels = ["Safe", "Spoiled"]
    actual = df["Actual_Status"]
    predicted = df["Predicted_Status"]

    cm = confusion_matrix(actual, predicted, labels=labels)
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)

    cm_df = pd.DataFrame(cm, index=[f"Actual_{x}" for x in labels], columns=[f"Predicted_{x}" for x in labels])
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})

    df.to_csv(PROCESSED_DIR / "spoilage_validation_labeled.csv", index=False)
    cm_df.to_csv(PROCESSED_DIR / "spoilage_validation_confusion_matrix.csv")
    report_df.to_csv(PROCESSED_DIR / "spoilage_validation_classification_report.csv", index=False)

    print("--- Blinded Prediction Performance ---")
    print(classification_report(actual, predicted, labels=labels, zero_division=0))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Safe", "Predicted Spoiled"],
        yticklabels=["Actual Safe", "Actual Spoiled"],
    )
    plt.title("Validation Study: Blinded Prediction Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "blinded_spoilage_cm.png", dpi=300)
    plt.close()
    print("Saved spoilage outputs to data/processed/ and figures/")


if __name__ == "__main__":
    main()
