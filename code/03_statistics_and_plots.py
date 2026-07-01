from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

REPO_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_DIR / "outputs"

ANALYSIS_LABEL = "RMS"
FEATURES_FILE = BASE_DIR / "RMS_statistics_results.xlsx"

OUTPUT_DIR = BASE_DIR / f"statistics_{ANALYSIS_LABEL}"
OUTPUT_STATS_FILE = OUTPUT_DIR / f"emg_statistics_{ANALYSIS_LABEL}.xlsx"

FIGURES_DIR = OUTPUT_DIR / "figures_emg"
FIGURES_NO_NAMES_DIR = FIGURES_DIR / "no_names"
FIGURES_NAMED_DIR = FIGURES_DIR / "named_outliers"
FIGURES_ZOOM_DIR = FIGURES_DIR / "no_names_zoom"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
FIGURES_NO_NAMES_DIR.mkdir(exist_ok=True)
SAVE_NAMED_QC_FIGURES = False
if SAVE_NAMED_QC_FIGURES:
    FIGURES_NAMED_DIR.mkdir(exist_ok=True)
FIGURES_ZOOM_DIR.mkdir(exist_ok=True)



VARIABLES = [
    "rms_mean_10cycles_ratio",
]

PLOT_ORDER = [
    "rms_mean_10cycles_ratio_biceps",
    "rms_mean_10cycles_ratio_triceps",
]

RMS_PLOT_TITLES = {
    "rms_mean_10cycles_ratio_biceps": "RMS ratio over time - Biceps brachii",
    "rms_mean_10cycles_ratio_triceps": "RMS ratio over time - Triceps brachii",
}

TRIAL_LABELS = {
    1: "Pre",
    2: "Post",
    3: "Post 5",
    4: "Post 10",
    5: "Post 15",
    6: "Post 20",
    7: "Post 25",
    8: "Post 30",
}

POST_TRIALS = [2,3,4,5,6,7,8]
ALPHA = 0.05

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

def normalize_session(value):
    text = normalize_text(value)

    if "control" in text: return "control"
    if "vibr" in text: return "vibration"
    
    return text

def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    n = len(ranked_p)

    adjusted_ranked = ranked_p*n / np.arange(1, n+1)

    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)

    adjusted_valid = np.empty(n)
    adjusted_valid[order] = adjusted_ranked
    adjusted[valid_mask] = adjusted_valid

    return adjusted

def add_bh_correction(df, group_cols):
    df = df.copy()
    df["p_adj"] = np.nan

    if len(df) == 0:
        df["significant_raw"] = []
        df["significant_adj"] = []
        return df

    groupby_cols = (
        group_cols[0]
        if isinstance(group_cols, list) and len(group_cols) == 1
        else group_cols
    )

    for _, idx in df.groupby(groupby_cols, dropna=False).groups.items():
        p_values = df.loc[idx, "p_raw"].to_numpy(dtype=float)
        df.loc[idx, "p_adj"] = benjamini_hochberg(p_values)
    
    df["significant_raw"] = df["p_raw"] < ALPHA
    df["significant_adj"] = df["p_adj"] < ALPHA

    return df


def significance_label(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def prepare_long_table(df_clean, available_base_features):
    id_cols = [
        "subject_id",
        "name_folder",
        "session",
        "session_norm",
        "trial_number",
        "trial_label",
        "muscle",
        "file_name",
        "feature_status",
        "baseline_qc_flag",
        "baseline_method",
        "baseline_source_trial",
        "baseline_level",
        "baseline_threshold",
        "include_analysis",
        "qc_flag",
    ]

    id_cols = [col for col in id_cols if col in df_clean.columns]

    rows = []

    for feature in available_base_features:
        temp = df_clean[id_cols + [feature]].copy()
        temp = temp.rename(columns={feature: "ratio"})
        temp["feature"] = feature

        temp["variable"] = (
            temp["feature"].astype(str)
            + "_"
            + temp["muscle"].astype(str)
        )

        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)

    long_df["ratio"] = pd.to_numeric(long_df["ratio"], errors="coerce")

    if "include_analysis" not in long_df.columns:
        long_df["include_analysis"] = True

    if "qc_flag" not in long_df.columns:
        long_df["qc_flag"] = "ok"

    include_as_text = (
        long_df["include_analysis"]
        .fillna(True)
        .astype(str)
        .str.strip()
        .str.lower()
    )

    long_df["include_analysis_bool"] = include_as_text.isin(
        ["true", "1", "yes", "si", "sí"]
    )

    long_df["valid_for_log"] = (
        long_df["include_analysis_bool"]
        & long_df["ratio"].notna()
        & np.isfinite(long_df["ratio"])
        & (long_df["ratio"] > 0)
    )

    long_df["invalid_log_reason"] = ""

    long_df.loc[long_df["ratio"].isna(), "invalid_log_reason"] = "ratio_nan"

    long_df.loc[
        long_df["ratio"].notna()
        & np.isinf(long_df["ratio"]),
        "invalid_log_reason"
    ] = "ratio_inf"

    long_df.loc[
        long_df["ratio"].notna()
        & np.isfinite(long_df["ratio"])
        & (long_df["ratio"] <= 0),
        "invalid_log_reason"
    ] = "ratio_leq_zero"

    long_df.loc[
        ~long_df["include_analysis_bool"],
        "invalid_log_reason"
    ] = "include_analysis_false"

    long_df["log_ratio"] = np.nan
    long_df.loc[long_df["valid_for_log"], "log_ratio"] = np.log(
        long_df.loc[long_df["valid_for_log"], "ratio"]
    )

    long_df["trial_cat"] = "T" + long_df["trial_number"].astype(int).astype(str)
    long_df["cell"] = long_df["session_norm"] + "_" + long_df["trial_cat"]
    long_df["subject_id"] = long_df["subject_id"].astype(str)

    return long_df


def fit_mixedlm(formula, data, group_col="subject_id"):
    if len(data) < 6:
        return None, "not_enough_rows"

    if data[group_col].nunique() < 3:
        return None, "not_enough_subjects"

    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data[group_col],
    )
    last_error = None

    for method in ["lbfgs", "bfgs", "cg", "powell"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=2000,
                    disp=False,
                )
            if getattr(result, "converged", False):
                return result, f"converged_{method}"

            last_error = f"not_converged_{method}"

        except Exception as exc:
            last_error = f"{method}_error: {exc}"
    
    return None, last_error


def contrast_test(result, term_a, term_b):
    #contrasts trema-termb from a fitted lmm
    if result is None:
        return{
            "estimate_log": np.nan,
            "se_log": np.nan,
            "z_value": np.nan,
            "p_raw": np.nan,
        }
    
    fe_names = list(result.fe_params.index)

    if term_a not in fe_names or term_b not in fe_names:
        return{
            "estimate_log": np.nan,
            "se_log": np.nan,
            "z_value": np.nan,
            "p_raw": np.nan,
        }

    contrast = np.zeros(len(fe_names))
    contrast[fe_names.index(term_a)] = 1.0
    contrast[fe_names.index(term_b)] = -1.0

    estimate = float(np.dot(contrast, result.fe_params.values))

    cov = result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
    variance = float(np.dot(contrast, np.dot(cov, contrast)))

    if variance <= 0 or pd.isna(variance):
        se = np.nan
        z_value = np.nan
        p_raw = np.nan
    else:
        se = float(np.sqrt(variance))
        z_value = estimate / se
        p_raw = float(2 * stats.norm.sf(abs(z_value)))

    return {
        "estimate_log": estimate,
        "se_log": se,
        "z_value": z_value,
        "p_raw": p_raw,
    }


def linear_combination_test(result, weights):
    if result is None:
        return {
            "estimate_log": np.nan,
            "se_log": np.nan,
            "z_value": np.nan,
            "p_raw": np.nan,
        }

    fe_names = list(result.fe_params.index)
    contrast = np.zeros(len(fe_names))

    for term, weight in weights.items():
        if term not in fe_names:
            return {
                "estimate_log": np.nan,
                "se_log": np.nan,
                "z_value": np.nan,
                "p_raw": np.nan,
            }
        contrast[fe_names.index(term)] = weight

    estimate = float(np.dot(contrast, result.fe_params.values))

    cov = result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
    variance = float(np.dot(contrast, np.dot(cov, contrast)))

    if variance <= 0 or pd.isna(variance):
        se = np.nan
        z_value = np.nan
        p_raw = np.nan
    else:
        se = float(np.sqrt(variance))
        z_value = estimate / se
        p_raw = float(2 * stats.norm.sf(abs(z_value)))

    return {
        "estimate_log": estimate,
        "se_log": se,
        "z_value": z_value,
        "p_raw": p_raw,
    }


def safe_exp(value):
    if pd.isna(value):
        return np.nan

    if value > 700 or value < -700:
        return np.nan

    return float(np.exp(value))

def run_intragroup_lmm(long_df, available_variables):
    # Each coeff estimates log(post/pre) at one follow-up time. Testing coeff = o porque post/pre = 1
    rows = []
    fit_rows = []

    all_trials = POST_TRIALS
    reference_trial = "T2"

    for variable in available_variables:
        for session in ["control", "vibration"]:
            subset = long_df[
                (long_df["variable"] == variable)
                & (long_df["session_norm"] == session)
                & (long_df["trial_number"].isin(all_trials))
                & (long_df["valid_for_log"])
            ].copy()

            subset = subset.dropna(
                subset=["log_ratio", "subject_id", "trial_cat"]
            )

            if len(subset) > 0:
                subset["trial_cat"] = pd.Categorical(
                    subset["trial_cat"],
                    categories=[f"T{trial}" for trial in all_trials],
                    ordered=True,
                )

            formula = "log_ratio ~ C(trial_cat, Treatment(reference='T2'))"

            result, fit_status = fit_mixedlm(
                formula=formula,
                data=subset,
            )

            fit_rows.append(
                {
                    "comparison_type": "intragroup_lmm",
                    "variable": variable,
                    "session": session,
                    "n_rows": len(subset),
                    "n_subjects": subset["subject_id"].nunique()
                    if len(subset) > 0
                    else 0,
                    "fit_status": fit_status,
                    "model_formula": formula,
                }
            )

            for trial in POST_TRIALS:
                trial_cat = f"T{trial}"

                if trial_cat == reference_trial:
                    weights = {"Intercept": 1.0}
                else:
                    term = (
                        "C(trial_cat, Treatment(reference='T2'))"
                        f"[T.{trial_cat}]"
                    )
                    weights = {
                        "Intercept": 1.0,
                        term: 1.0,
                    }

                test = linear_combination_test(result, weights)

                estimate_log = test["estimate_log"]
                se_log = test["se_log"]

                rows.append(
                    {
                        "comparison_type": "intragroup_lmm_post_vs_pre",
                        "variable": variable,
                        "session": session,
                        "trial_number": trial,
                        "trial_label": TRIAL_LABELS.get(trial, str(trial)),
                        "n_rows_model": len(subset),
                        "n_subjects_model": subset["subject_id"].nunique()
                        if len(subset) > 0
                        else 0,
                        "fit_status": fit_status,
                        **test,
                        "estimated_ratio": safe_exp(estimate_log)
                        if pd.notna(estimate_log)
                        else np.nan,
                        "ci95_low_ratio": safe_exp(estimate_log - 1.96 * se_log)
                        if pd.notna(estimate_log) and pd.notna(se_log)
                        else np.nan,
                        "ci95_high_ratio": safe_exp(estimate_log + 1.96 * se_log)
                        if pd.notna(estimate_log) and pd.notna(se_log)
                        else np.nan,
                    }
                )

    intragroup = pd.DataFrame(rows)
    fit_summary = pd.DataFrame(fit_rows)

    if len(intragroup) > 0:
        intragroup = add_bh_correction(
            intragroup,
            group_cols=["variable", "session"],
        )

    return intragroup, fit_summary

def run_intergroup_lmm(long_df, available_variables):
    rows = []
    fit_rows = []

    for variable in available_variables:
        subset = long_df[
            (long_df["variable"] == variable)
            & (long_df["session_norm"].isin(["control", "vibration"]))
            & (long_df["trial_number"].isin(POST_TRIALS))
            & (long_df["valid_for_log"])
        ].copy()

        subset = subset.dropna(subset=["log_ratio", "subject_id", "cell"])

        formula = "log_ratio ~ 0 + C(cell)"

        result, fit_status = fit_mixedlm(
            formula=formula,
            data=subset,
        )

        fit_rows.append(
            {
                "comparison_type": "intergroup_lmm",
                "variable": variable,
                "n_rows": len(subset),
                "n_subjects": subset["subject_id"].nunique()
                if len(subset) > 0
                else 0,
                "fit_status": fit_status,
                "model_formula": formula,
            }
        )

        for trial in POST_TRIALS:
            control_term = f"C(cell)[control_T{trial}]"
            vibration_term = f"C(cell)[vibration_T{trial}]"

            test = contrast_test(result, vibration_term, control_term)

            estimate_log = test["estimate_log"]
            se_log = test["se_log"]

            rows.append(
                {
                    "comparison_type": "intergroup_lmm_vibration_vs_control",
                    "variable": variable,
                    "trial_number": trial,
                    "trial_label": TRIAL_LABELS.get(trial, str(trial)),
                    "n_rows_model": len(subset),
                    "n_subjects_model": subset["subject_id"].nunique()
                    if len(subset) > 0
                    else 0,
                    "fit_status": fit_status,
                    **test,
                    "estimated_ratio_of_ratios": safe_exp(estimate_log)
                    if pd.notna(estimate_log)
                    else np.nan,
                    "ci95_low_ratio_of_ratios": safe_exp(estimate_log - 1.96 * se_log)
                    if pd.notna(estimate_log) and pd.notna(se_log)
                    else np.nan,
                    "ci95_high_ratio_of_ratios": safe_exp(estimate_log + 1.96 * se_log)
                    if pd.notna(estimate_log) and pd.notna(se_log)
                    else np.nan,
                }
            )

    intergroup = pd.DataFrame(rows)
    fit_summary = pd.DataFrame(fit_rows)

    if len(intergroup) > 0:
        intergroup = add_bh_correction(
            intergroup,
            group_cols=["variable"],
        )

    return intergroup, fit_summary

def compute_descriptives(long_df):
    rows = []
    for (variable, session, trial), group in long_df.groupby(
        ["variable", "session_norm", "trial_number"],
        dropna=False
    ):
        values = group["ratio"].dropna()
        if len(values) == 0: continue

        rows.append(
            {
                    "variable": variable,
                    "session": session,
                    "trial_number": trial,
                    "trial_label": TRIAL_LABELS.get(trial, str(trial)),
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "sem": float(values.sem()) if len(values) > 1 else np.nan,
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def add_outlier_flags(long_df):
    df = long_df.copy()

    df["outlier_for_plot"] = False
    df["outlier_reason"] = ""

    def add_reason(mask, reason):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)

        for idx in df.index[mask]:
            previous_reason = str(df.at[idx, "outlier_reason"])

            if previous_reason == "":
                df.at[idx, "outlier_reason"] = reason
            else:
                df.at[idx, "outlier_reason"] = previous_reason + " | " + reason

            df.at[idx, "outlier_for_plot"] = True

   
    if "qc_flag" in df.columns:
        qc_mask = df["qc_flag"].fillna("ok") != "ok"
        add_reason(qc_mask, "qc_flag")

    invalid_log_mask = ~df["valid_for_log"]
    add_reason(invalid_log_mask, "invalid_log")


    df["label_for_plot"] = False

    df.loc[
        df["outlier_reason"].fillna("").str.contains(
            "qc_flag|invalid_log",
            regex=True,
        ),
        "label_for_plot"
    ] = True

    df["plot_label"] = df["name_folder"].astype(str)

    return df


def plot_variable_timecourse(variable, long_df, intergroup_lmm, output_dir, annotate_names=False, ylim=None, file_suffix=None):
    plot_data = long_df[
        (long_df["variable"] == variable)
        & (long_df["ratio"].notna())
    ].copy()

    if len(plot_data) == 0:
        return

    trials = sorted(TRIAL_LABELS.keys())
    sessions = ["control", "vibration"]

    fig, ax = plt.subplots(figsize=(11, 5))

    positions = []
    box_data = []
    box_colors = []

    offset = 0.18
    base_positions = np.arange(len(trials))

    session_position_map = {
        "control": base_positions - offset,
        "vibration": base_positions + offset,
    }

    for session in sessions:
        for i, trial in enumerate(trials):
            values = plot_data[
                (plot_data["session_norm"] == session)
                & (plot_data["trial_number"] == trial)
            ]["ratio"].dropna()

            positions.append(session_position_map[session][i])
            box_data.append(values.to_numpy())

            if session == "control":
                box_colors.append("C0")
            else:
                box_colors.append("C1")

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "white",
            "markeredgecolor": "gray",
            "alpha": 0.8,
        },
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)

    for session in sessions:
        color = "C0" if session == "control" else "C1"

        for i, trial in enumerate(trials):
            subset_points = plot_data[
                (plot_data["session_norm"] == session)
                & (plot_data["trial_number"] == trial)
            ].copy()

            if subset_points.empty:
                continue

            x_base = session_position_map[session][i]
            jitter = rng.uniform(-0.055, 0.055, size=len(subset_points))
            x_values = x_base + jitter
            y_values = subset_points["ratio"].to_numpy(dtype=float)

            ax.scatter(
                x_values,
                y_values,
                s=14,
                alpha=0.65,
                color=color,
                edgecolors="none",
                zorder=3,
            )

            if annotate_names:
                for x, y, (_, row) in zip(
                    x_values,
                    y_values,
                    subset_points.iterrows(),
                ):
                    if bool(row.get("label_for_plot", False)):
                        label = str(row.get("plot_label", ""))

                        ax.annotate(
                            label,
                            xy=(x, y),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=7,
                            alpha=0.9,
                        )

    for session in sessions:
        means = []
        x_positions = []

        for i, trial in enumerate(trials):
            values = plot_data[
                (plot_data["session_norm"] == session)
                & (plot_data["trial_number"] == trial)
            ]["ratio"].dropna()

            if len(values) == 0:
                means.append(np.nan)
            else:
                means.append(values.mean())

            x_positions.append(session_position_map[session][i])

        color = "C0" if session == "control" else "C1"
        label = "Control" if session == "control" else "Vibration"

        ax.plot(
            x_positions,
            means,
            marker="o",
            linestyle="--",
            color=color,
            label=label,
        )

    ax.axhline(1, linestyle="--", linewidth=1, color="gray")
    if ylim is not None:
        ax.set_ylim(ylim)

    title = RMS_PLOT_TITLES.get(variable, variable)
    ax.set_title(title)
    ax.set_xlabel("Follow-up point")
    ax.set_ylabel("RMS ratio relative to Pre")
    ax.set_xticks(base_positions)
    ax.set_xticklabels([TRIAL_LABELS[i] for i in trials], rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Session")

    sig_data = intergroup_lmm[
        (intergroup_lmm["variable"] == variable)
        & (intergroup_lmm["significant_adj"])
    ]

    if len(sig_data) > 0:
        if ylim is not None:
            y_star = ylim[1] * 0.93
        else:
            y_max = plot_data["ratio"].max()

            if pd.isna(y_max):
                y_max = 1.0

            y_star = y_max * 1.08

        for _, row in sig_data.iterrows():
            trial = int(row["trial_number"])

            if trial in trials:
                trial_index = trials.index(trial)
                label = significance_label(row["p_adj"])

                if label:
                    ax.text(
                        base_positions[trial_index],
                        y_star,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=14,
                    )

    fig.tight_layout()

    if file_suffix is None:
        suffix = "named" if annotate_names else "no_names"
    else:
        suffix = file_suffix

    output_path = output_dir / f"{variable}_boxplot_{suffix}_{ANALYSIS_LABEL}.png"

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found:\n{FEATURES_FILE}")
    
    df_clean = pd.read_excel(FEATURES_FILE, sheet_name="results_clean")
    df_clean.columns = [str(column).strip() for column in df_clean.columns]

    df_clean["session_norm"] = df_clean["session"].apply(normalize_session)
    df_clean["trial_label"] = df_clean["trial_number"].map(TRIAL_LABELS)

    df_clean = df_clean[
        df_clean["session_norm"].isin(["control", "vibration"])
    ].copy()

    available_base_features = [
        variable
        for variable in VARIABLES
        if variable in df_clean.columns
    ]

    missing_variables = [
        variable
        for variable in VARIABLES
        if variable not in df_clean.columns
    ]

    print(f"Rows loaded from clean file: {len(df_clean)}")
    print(f"Available base features: {available_base_features}")

    if missing_variables:
        print(f"Missing variables, skipped: {missing_variables}")

    long_df = prepare_long_table(df_clean, available_base_features)
    long_df = add_outlier_flags(long_df)

    available_variables = sorted(long_df["variable"].dropna().unique())

    print(f"Available final variables: {available_variables}")

    validity_summary = (
        long_df
        .assign(subject_valid=lambda df: df["subject_id"].where(df["valid_for_log"]))
        .groupby(["variable", "session_norm", "muscle"], dropna=False)
        .agg(
            n_total=("ratio", "size"),
            n_valid_for_log=("valid_for_log", "sum"),
            n_nan=("invalid_log_reason", lambda x: sum(x == "ratio_nan")),
            n_inf=("invalid_log_reason", lambda x: sum(x == "ratio_inf")),
            n_leq_zero=("invalid_log_reason", lambda x: sum(x == "ratio_leq_zero")),
            n_subjects_valid=("subject_valid", "nunique"),
        )
        .reset_index()
    )

    descriptives = compute_descriptives(long_df)


    intragroup_lmm_main, intragroup_fit_summary_main = run_intragroup_lmm(
        long_df,
        available_variables,
    )

    intergroup_lmm_main, intergroup_fit_summary_main = run_intergroup_lmm(
        long_df,
        available_variables,
    )

    intragroup_lmm_main["analysis_set"] = "main_all_valid"
    intergroup_lmm_main["analysis_set"] = "main_all_valid"
    intragroup_fit_summary_main["analysis_set"] = "main_all_valid"
    intergroup_fit_summary_main["analysis_set"] = "main_all_valid"

    intragroup_lmm = intragroup_lmm_main.copy()
    intergroup_lmm = intergroup_lmm_main.copy()

    fit_summary = pd.concat(
        [
            intragroup_fit_summary_main,
            intergroup_fit_summary_main,
        ],
        ignore_index=True,
        sort=False,
    )

    significant_tables=[]
    if len(intragroup_lmm) > 0:
        temp = intragroup_lmm[intragroup_lmm["significant_adj"]].copy()
        temp["source_table"] = "intragroup_lmm"
        significant_tables.append(temp)

    if len(intergroup_lmm) > 0:
        temp = intergroup_lmm[intergroup_lmm["significant_adj"]].copy()
        temp["source_table"] = "intergroup_lmm"
        significant_tables.append(temp)

    if significant_tables:
        significant_results = pd.concat(
            significant_tables,
            ignore_index=True,
            sort=False,
        )
    else:
        significant_results = pd.DataFrame()

    plot_order = [
        variable
        for variable in PLOT_ORDER
        if variable in available_variables
    ]

    for variable in plot_order:
        plot_variable_timecourse(
            variable=variable,
            long_df=long_df,
            intergroup_lmm=intergroup_lmm_main,
            output_dir=FIGURES_NO_NAMES_DIR,
            annotate_names=False,
        )

        if SAVE_NAMED_QC_FIGURES:
            plot_variable_timecourse(
                variable=variable,
                long_df=long_df,
                intergroup_lmm=intergroup_lmm_main,
                output_dir=FIGURES_NAMED_DIR,
                annotate_names=True,
            )

        plot_variable_timecourse(
            variable=variable,
            long_df=long_df,
            intergroup_lmm=intergroup_lmm_main,
            output_dir=FIGURES_ZOOM_DIR,
            annotate_names=False,
            ylim=(0, 3),
            file_suffix="no_names_zoom_y0_3",
        )
    qc_outliers = long_df[
        long_df["outlier_for_plot"]
    ].copy()

    qc_outlier_cols = [
        "subject_id",
        "name_folder",
        "session_norm",
        "trial_number",
        "trial_label",
        "muscle",
        "feature",
        "variable",
        "ratio",
        "log_ratio",
        "qc_flag",
        "outlier_reason",
        "invalid_log_reason",
        "baseline_method",
        "baseline_source_trial",
        "file_name",
    ]

    qc_outlier_cols = [
        col for col in qc_outlier_cols
        if col in qc_outliers.columns
    ]

    qc_outliers = qc_outliers[qc_outlier_cols].copy()

    extreme_high_ratios = long_df[
        long_df["ratio"].notna()
        & np.isfinite(long_df["ratio"])
        & (long_df["ratio"] > 3)
    ].copy()

    extreme_high_cols = [
        "subject_id",
        "name_folder",
        "session_norm",
        "trial_number",
        "trial_label",
        "muscle",
        "variable",
        "ratio",
        "log_ratio",
        "qc_flag",
        "outlier_reason",
        "invalid_log_reason",
        "baseline_method",
        "baseline_source_trial",
        "baseline_level",
        "baseline_threshold",
        "file_name",
    ]

    extreme_high_cols = [
        col for col in extreme_high_cols
        if col in extreme_high_ratios.columns
    ]

    extreme_high_ratios = (
        extreme_high_ratios[extreme_high_cols]
        .sort_values(["variable", "ratio"], ascending=[True, False])
    )

    analysis_info = pd.DataFrame(
        [
            {
                "item": "analysis_label",
                "value": ANALYSIS_LABEL,
            },
            {
                "item": "features_file",
                "value": str(FEATURES_FILE),
            },
            {
                "item": "n_rows_clean_loaded",
                "value": len(df_clean),
            },
            {
                "item": "n_rows_long",
                "value": len(long_df),
            },
            {
                "item": "available_variables",
                "value": ", ".join(available_variables),
            },
            {
                "item": "missing_variables",
                "value": ", ".join(map(str, missing_variables)),
            },
            {
                "item": "post_trials",
                "value": ", ".join(map(str, POST_TRIALS)),
            },
            {
                "item": "alpha",
                "value": ALPHA,
            },
            {
                "item": "multiple_comparison_correction",
                "value": "Benjamini-Hochberg FDR",
            },
            {
                "item": "main_statistical_model_intragroup",
                "value": "log(Post/Pre) ~ C(time_post, reference=Post) + (1|subject), fitted separately within each condition; planned contrasts tested each estimated time-specific log ratio against 0, equivalent to testing Post/Pre against 1.",
            },
            {
                "item": "main_statistical_model_intergroup",
                "value": "log(Post/Pre) ~ 0 +C(condition_time_cell) + (1|subject), where condition_time_cell represents condition*time; planned contrasts tested vibration-control at each post time.",
            },
            {
                "item": "main_analysis_set",
                "value": "main_all_valid: all rows with finite positive ratios and include_analysis=True",
            },
        ]
    )

    with pd.ExcelWriter(OUTPUT_STATS_FILE, engine="openpyxl") as writer:
        analysis_info.to_excel(writer, sheet_name="analysis_info", index=False)
        descriptives.to_excel(writer, sheet_name="descriptives", index=False)
        intragroup_lmm.to_excel(writer, sheet_name="intragroup_lmm", index=False)
        intergroup_lmm.to_excel(writer, sheet_name="intergroup_lmm", index=False)
        significant_results.to_excel(writer, sheet_name="significant_results", index=False)

    print("LMM statistical analysis finished.")
    print(f"Results saved in:\n{OUTPUT_STATS_FILE}")
    print(f"Figures without names saved in:\n{FIGURES_NO_NAMES_DIR}")
    if SAVE_NAMED_QC_FIGURES:
        print(f"Named QC figures saved in:\n{FIGURES_NAMED_DIR}")
    print(f"Zoom figures without names saved in:\n{FIGURES_ZOOM_DIR}")

    if len(significant_results) == 0:
        print("\nNo significant adjusted results found.")
    else:
        print("\nSignificant adjusted results found:")
        cols_to_show = [
            col
            for col in [
                "source_table",
                "comparison_type",
                "variable",
                "session",
                "trial_number",
                "trial_label",
                "p_raw",
                "p_adj",
                "estimated_ratio",
                "estimated_ratio_of_ratios",
                "fit_status",
            ]
            if col in significant_results.columns
        ]
        print(significant_results[cols_to_show])


if __name__ == "__main__":
    main()
