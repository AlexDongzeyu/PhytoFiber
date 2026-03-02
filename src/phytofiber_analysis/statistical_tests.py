import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_assumption_checks(
    df: pd.DataFrame, group_col: str, value_col: str
) -> pd.DataFrame:
    groups = [
        subset[value_col].dropna().values
        for _, subset in df.groupby(group_col)
        if len(subset[value_col].dropna()) >= 3
    ]
    shapiro_rows = []
    for group_name, subset in df.groupby(group_col):
        values = subset[value_col].dropna()
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            shapiro_rows.append(
                {
                    "test": "shapiro",
                    "group": group_name,
                    "statistic": stat,
                    "p_value": p,
                }
            )
    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*groups)
    else:
        lev_stat, lev_p = float("nan"), float("nan")
    shapiro_rows.append(
        {
            "test": "levene",
            "group": "all_groups",
            "statistic": lev_stat,
            "p_value": lev_p,
        }
    )
    return pd.DataFrame(shapiro_rows)


def run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    grouped = [vals[value_col].dropna().values for _, vals in df.groupby(group_col)]
    if len(grouped) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    f_stat, p_val = stats.f_oneway(*grouped)
    return pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "f_statistic": f_stat,
                "p_value": p_val,
                "alpha": 0.05,
                "significant": bool(p_val < 0.05),
            }
        ]
    )


def run_tukey(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    tukey = pairwise_tukeyhsd(
        endog=df[value_col].astype(float),
        groups=df[group_col].astype(str),
        alpha=0.05,
    )
    table = tukey.summary()
    tukey_df = pd.DataFrame(table.data[1:], columns=table.data[0])
    return tukey_df

