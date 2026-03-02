from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import (
    TENSILE_CSV,
    DATA_PROCESSED_DIR,
    ANOVA_RESULTS_CSV,
    TUKEY_RESULTS_CSV,
)
from phytofiber_analysis.io_utils import read_csv_checked, write_csv
from phytofiber_analysis.statistical_tests import run_assumption_checks, run_anova, run_tukey


def main() -> None:
    tensile = read_csv_checked(
        TENSILE_CSV, required_columns=["sample_id", "group", "force_n", "diameter_mm"]
    ).copy()
    tensile["cross_section_mm2"] = 3.141592653589793 * (tensile["diameter_mm"] / 2.0) ** 2
    tensile["tensile_mpa"] = tensile["force_n"] / tensile["cross_section_mm2"]

    assumption_df = run_assumption_checks(tensile, group_col="group", value_col="tensile_mpa")
    anova_df = run_anova(tensile, group_col="group", value_col="tensile_mpa")
    tukey_df = run_tukey(tensile, group_col="group", value_col="tensile_mpa")

    write_csv(tensile, DATA_PROCESSED_DIR / "tensile_with_mpa.csv")
    write_csv(assumption_df, DATA_PROCESSED_DIR / "assumption_checks.csv")
    write_csv(anova_df, ANOVA_RESULTS_CSV)
    write_csv(tukey_df, TUKEY_RESULTS_CSV)
    print("Saved statistics outputs to data/processed/")


if __name__ == "__main__":
    main()

