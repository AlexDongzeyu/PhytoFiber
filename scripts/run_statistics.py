from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from phytofiber_analysis.config import (
    ANOVA_RESULTS_CSV,
    ASSUMPTION_RESULTS_CSV,
    DATA_PROCESSED_DIR,
    EFFECT_SIZE_CSV,
    TENSILE_CSV,
    TENSILE_PROCESSED_CSV,
    TENSILE_RAW_CSV,
    TUKEY_RESULTS_CSV,
)
from phytofiber_analysis.io_utils import choose_existing_file, maybe_rename_columns, read_csv_checked, write_csv
from phytofiber_analysis.statistical_tests import compute_group_descriptives, compute_pairwise_effect_sizes, compute_tensile_stress, run_anova, run_assumption_checks, run_tukey


def main() -> None:
    tensile_path = choose_existing_file(TENSILE_CSV, TENSILE_RAW_CSV)
    tensile = pd.read_csv(tensile_path)
    tensile = maybe_rename_columns(
        tensile,
        {
            "Group": "group",
            "Force_N": "force_n",
            "Diameter_mm": "diameter_mm",
            "Sample_ID": "sample_id",
        },
    )
    missing = [col for col in ["sample_id", "group", "force_n", "diameter_mm"] if col not in tensile.columns]
    if missing:
        raise ValueError(f"{tensile_path.name} is missing required columns after normalization: {missing}")
    if tensile[["force_n", "diameter_mm"]].dropna(how="all").empty:
        raise ValueError("Tensile file exists, but numeric measurements have not been entered yet.")
    tensile = compute_tensile_stress(tensile)

    assumption_df = run_assumption_checks(tensile, group_col="group", value_col="stress_mpa")
    anova_df = run_anova(tensile, group_col="group", value_col="stress_mpa")
    descriptive_df = compute_group_descriptives(tensile, group_col="group", value_col="stress_mpa")
    effect_df = compute_pairwise_effect_sizes(tensile, group_col="group", value_col="stress_mpa")
    if bool(anova_df.loc[0, "significant"]):
        tukey_df = run_tukey(tensile, group_col="group", value_col="stress_mpa")
    else:
        tukey_df = pd.DataFrame(columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"])

    write_csv(tensile, TENSILE_PROCESSED_CSV)
    write_csv(assumption_df, ASSUMPTION_RESULTS_CSV)
    write_csv(anova_df, ANOVA_RESULTS_CSV)
    write_csv(tukey_df, TUKEY_RESULTS_CSV)
    write_csv(descriptive_df, DATA_PROCESSED_DIR / "tensile_descriptives.csv")
    write_csv(effect_df, EFFECT_SIZE_CSV)
    print("Saved statistics outputs to data/processed/")


if __name__ == "__main__":
    main()

