import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def compute_tensile_stress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["force_n"] = out["force_n"].astype(float)
    out["diameter_mm"] = out["diameter_mm"].astype(float)
    out["radius_mm"] = out["diameter_mm"] / 2.0
    out["cross_section_mm2"] = 3.141592653589793 * (out["radius_mm"] ** 2)
    out["stress_mpa"] = out["force_n"] / out["cross_section_mm2"]
    out["tensile_mpa"] = out["stress_mpa"]
    return out


def run_assumption_checks(
    df: pd.DataFrame, group_col: str, value_col: str
) -> pd.DataFrame:
    work = df[[group_col, value_col]].dropna().copy()
    all_values = work[value_col].astype(float)
    groups = []
    rows = []

    if len(all_values) >= 3:
        stat, p = stats.shapiro(all_values)
        rows.append(
            {
                "test": "shapiro_all",
                "group": "all_values",
                "statistic": stat,
                "p_value": p,
                "passes_alpha_0_05": bool(p > 0.05),
            }
        )

    for group_name, subset in df.groupby(group_col):
        values = subset[value_col].dropna().astype(float)
        if len(values) >= 3:
            stat, p = stats.shapiro(values)
            rows.append(
                {
                    "test": "shapiro_group",
                    "group": group_name,
                    "statistic": stat,
                    "p_value": p,
                    "passes_alpha_0_05": bool(p > 0.05),
                }
            )
            groups.append(values.values)

    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*groups)
    else:
        lev_stat, lev_p = float("nan"), float("nan")
    rows.append(
        {
            "test": "levene",
            "group": "all_groups",
            "statistic": lev_stat,
            "p_value": lev_p,
            "passes_alpha_0_05": bool(lev_p > 0.05) if lev_p == lev_p else False,
        }
    )
    return pd.DataFrame(rows)


def run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    grouped = [vals[value_col].dropna().astype(float).values for _, vals in df.groupby(group_col)]
    if len(grouped) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    f_stat, p_val = stats.f_oneway(*grouped)
    work = df[[group_col, value_col]].dropna().copy()
    grand_mean = work[value_col].astype(float).mean()
    ss_between = 0.0
    ss_total = ((work[value_col].astype(float) - grand_mean) ** 2).sum()
    for _, subset in work.groupby(group_col):
        values = subset[value_col].astype(float)
        ss_between += len(values) * ((values.mean() - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total else float("nan")
    return pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "f_statistic": f_stat,
                "p_value": p_val,
                "eta_squared": eta_squared,
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


def compute_group_descriptives(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col)[value_col]
    out = grouped.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    out = out.rename(columns={group_col: "group", "count": "n"})
    out["sem"] = out["std"] / out["n"].pow(0.5)
    out["ci95_half_width"] = 1.96 * out["sem"]
    return out


def compute_pairwise_effect_sizes(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    groups = {
        name: subset[value_col].dropna().astype(float).values
        for name, subset in df.groupby(group_col)
    }
    rows = []
    group_names = list(groups.keys())
    for idx, left_name in enumerate(group_names):
        for right_name in group_names[idx + 1 :]:
            left = groups[left_name]
            right = groups[right_name]
            if len(left) < 2 or len(right) < 2:
                continue
            pooled_sd = (((len(left) - 1) * left.std(ddof=1) ** 2) + ((len(right) - 1) * right.std(ddof=1) ** 2))
            pooled_sd /= max(len(left) + len(right) - 2, 1)
            pooled_sd = pooled_sd ** 0.5
            cohen_d = (left.mean() - right.mean()) / pooled_sd if pooled_sd else float("nan")
            rows.append(
                {
                    "group_a": left_name,
                    "group_b": right_name,
                    "cohen_d": cohen_d,
                    "mean_diff": float(left.mean() - right.mean()),
                }
            )
    return pd.DataFrame(rows)

