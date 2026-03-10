from pathlib import Path
import json

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_checked(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def choose_existing_file(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected files exist: " + ", ".join(str(path) for path in paths)
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def maybe_rename_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    out = standardize_columns(df)
    applicable = {src: dst for src, dst in rename_map.items() if src in out.columns}
    if applicable:
        out = out.rename(columns=applicable)
    return out

