#!/usr/bin/env python3
"""
Analyze Blair et al. (2013) endorsement experiment data.

This script:
1. Loads the Pakistan endorsement survey data from the endorse R package
2. Rescales 1-5 responses to [0,1] (as Blair et al. did)
3. For each policy × endorser pair, computes:
   - Control mean (α)
   - Treatment mean
   - ATE (β = treatment - control)
   - Variance test for binary switching assumption
4. Applies the five behavioral models from generate_tables_and_figures.py
5. Outputs summary statistics and model-implied support estimates

Usage:
    python scripts/analyze_blair_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
TABS_DIR = os.path.join(ROOT_DIR, "tabs")

POLICIES = ["Polio", "FCR", "Durand", "Curriculum"]
ENDORSERS = {
    "b": "Kashmiri militants",
    "c": "Afghan Taliban",
    "d": "Al-Qaeda",
    "e": "Sectarian groups",
}

D_SYMMETRIC = 0.05
SIGMA_NORMAL = 0.05


def rescale_1_5_to_0_1(x):
    """Rescale 1-5 scale to [0,1] interval."""
    return (x - 1) / 4


def binary_switching(alpha, beta):
    return alpha + beta


def binary_no_baseline(beta):
    return (beta + 1) / 2


def symmetric_effects(beta, d):
    return beta / (2 * d) + 0.5


def extreme_opponent(beta):
    return 1 + beta


def normal_heterogeneity(beta, sigma):
    return 1 - stats.norm.cdf(-beta / sigma)


def predicted_variance_binary_switching(p):
    """
    Under binary switching, responses are 0 or 1 with probability (1-p) and p.
    Variance = p(1-p).
    """
    return p * (1 - p)


def load_data():
    path = os.path.join(DATA_DIR, "pakistan_endorsement.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run: Rscript scripts/download_blair_data.R"
        )
    return pd.read_csv(path)


def analyze_policy_endorser(df, policy, endorser_code):
    """Analyze a single policy-endorser pair."""
    control_col = f"{policy}.a"
    treatment_col = f"{policy}.{endorser_code}"

    control = df[control_col].dropna()
    treatment = df[treatment_col].dropna()

    control_rescaled = rescale_1_5_to_0_1(control)
    treatment_rescaled = rescale_1_5_to_0_1(treatment)

    n_control = len(control_rescaled)
    n_treatment = len(treatment_rescaled)
    alpha = control_rescaled.mean()
    treatment_mean = treatment_rescaled.mean()
    beta = treatment_mean - alpha

    se_control = control_rescaled.std() / np.sqrt(n_control)
    se_treatment = treatment_rescaled.std() / np.sqrt(n_treatment)
    se_ate = np.sqrt(se_control**2 + se_treatment**2)

    var_control = control_rescaled.var()
    var_treatment = treatment_rescaled.var()

    p_implied = binary_switching(alpha, beta)
    var_predicted = predicted_variance_binary_switching(p_implied)

    return {
        "policy": policy,
        "endorser": ENDORSERS[endorser_code],
        "endorser_code": endorser_code,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "alpha": alpha,
        "treatment_mean": treatment_mean,
        "beta": beta,
        "se_ate": se_ate,
        "var_control": var_control,
        "var_treatment": var_treatment,
        "var_predicted_binary": var_predicted,
        "p_binary_switching": p_implied,
        "p_binary_no_baseline": binary_no_baseline(beta),
        "p_symmetric": symmetric_effects(beta, D_SYMMETRIC),
        "p_normal": normal_heterogeneity(beta, SIGMA_NORMAL),
        "p_extreme_opponent": extreme_opponent(beta),
    }


def variance_ratio_test(var_observed, var_predicted, n):
    """
    Test if observed variance differs from predicted under binary switching.
    Uses chi-square test: (n-1)*s^2/sigma^2 ~ chi^2(n-1)
    """
    if var_predicted <= 0:
        return np.nan, np.nan
    chi2_stat = (n - 1) * var_observed / var_predicted
    df = n - 1
    p_value = 2 * min(
        stats.chi2.cdf(chi2_stat, df),
        1 - stats.chi2.cdf(chi2_stat, df)
    )
    return chi2_stat, p_value


def generate_variance_table(results_df):
    """Generate LaTeX table comparing observed vs predicted variance."""
    tex = (
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Policy & Endorser & Var(T) Obs & Var(T) Pred & Ratio \\\\\n"
        "\\midrule\n"
    )
    for _, row in results_df.iterrows():
        var_obs = row["var_treatment"]
        var_pred = row["var_predicted_binary"]
        ratio = var_obs / var_pred
        tex += (
            f"{row['policy']} & {row['endorser']} & "
            f"{var_obs:.4f} & {var_pred:.4f} & {ratio:.2f} \\\\\n"
        )

    avg_var_obs = results_df["var_treatment"].mean()
    avg_var_pred = results_df["var_predicted_binary"].mean()
    avg_ratio = avg_var_obs / avg_var_pred

    tex += "\\midrule\n"
    tex += f"\\textbf{{Average}} & & {avg_var_obs:.4f} & {avg_var_pred:.4f} & {avg_ratio:.2f} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    return tex


def generate_empirical_support_table(summary_by_endorser):
    """Generate LaTeX table of empirical support estimates by endorser."""
    tex = (
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Endorser & ATE & Binary & Symmetric & Normal & Extreme \\\\\n"
        "\\midrule\n"
    )
    for endorser, row in summary_by_endorser.iterrows():
        tex += (
            f"{endorser} & ${row['beta']:+.4f}$ & "
            f"{row['p_binary_switching']:.1%} & "
            f"{row['p_symmetric']:.1%} & "
            f"{row['p_normal']:.1%} & "
            f"{row['p_extreme_opponent']:.1%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    return tex.replace("%", "\\%")


def main():
    print("=" * 70)
    print("ANALYSIS OF BLAIR ET AL. (2013) ENDORSEMENT EXPERIMENT DATA")
    print("=" * 70)

    df = load_data()
    print(f"\nLoaded {len(df)} observations")

    results = []
    for policy in POLICIES:
        for endorser_code in ENDORSERS.keys():
            result = analyze_policy_endorser(df, policy, endorser_code)
            results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SECTION 1: VERIFY REPORTED STATISTICS")
    print("=" * 70)

    overall_alpha = results_df["alpha"].mean()
    overall_beta = results_df["beta"].mean()

    print(f"\nOverall control mean (α): {overall_alpha:.4f}")
    print(f"Overall ATE (β):          {overall_beta:.4f}")
    print(f"\nBlair et al. reported:    α ≈ 0.80, β ≈ -0.011")

    print("\n--- By Policy (pooled across endorsers) ---")
    for policy in POLICIES:
        policy_df = results_df[results_df["policy"] == policy]
        n_ctrl = policy_df["n_control"].iloc[0]
        alpha = policy_df["alpha"].iloc[0]
        beta_mean = policy_df["beta"].mean()
        print(f"  {policy:12s}: α = {alpha:.3f}, β = {beta_mean:+.4f} (n_control = {n_ctrl})")

    print("\n--- By Endorser (pooled across policies) ---")
    for code, name in ENDORSERS.items():
        endorser_df = results_df[results_df["endorser_code"] == code]
        beta_mean = endorser_df["beta"].mean()
        print(f"  {name:20s}: β = {beta_mean:+.4f}")

    print("\n" + "=" * 70)
    print("SECTION 2: VARIANCE TEST FOR BINARY SWITCHING ASSUMPTION")
    print("=" * 70)

    print("\nUnder binary switching, if treatment mean p = α + β,")
    print("then treatment variance should equal p(1-p).")
    print("\nIf responses remain on [0,1] scale (not binary 0/1),")
    print("observed variance will typically be LOWER than p(1-p).")

    print("\n" + "-" * 70)
    print(f"{'Policy':<12} {'Endorser':<20} {'Var(T) Obs':>10} {'Var(T) Pred':>11} {'Ratio':>8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        var_obs = row["var_treatment"]
        var_pred = row["var_predicted_binary"]
        ratio = var_obs / var_pred if var_pred > 0 else np.nan
        print(
            f"{row['policy']:<12} {row['endorser']:<20} "
            f"{var_obs:10.4f} {var_pred:11.4f} {ratio:8.2f}"
        )

    avg_var_obs = results_df["var_treatment"].mean()
    avg_var_pred = results_df["var_predicted_binary"].mean()
    avg_ratio = avg_var_obs / avg_var_pred

    print("-" * 70)
    print(f"{'AVERAGE':<33} {avg_var_obs:10.4f} {avg_var_pred:11.4f} {avg_ratio:8.2f}")
    print()

    print(f"Average observed variance:  {avg_var_obs:.4f}")
    print(f"Average predicted variance: {avg_var_pred:.4f}")
    print(f"Ratio (observed/predicted): {avg_ratio:.2f}")

    if avg_ratio < 1:
        print("\n=> Observed variance is LOWER than binary switching predicts.")
        print("   This suggests responses are NOT simply 0 or 1.")
    else:
        print("\n=> Observed variance is HIGHER than binary switching predicts.")

    print("\n" + "=" * 70)
    print("SECTION 3: MODEL-IMPLIED SUPPORT ESTIMATES")
    print("=" * 70)

    print("\nUsing overall averages: α = {:.4f}, β = {:.4f}".format(overall_alpha, overall_beta))

    models = [
        ("Binary switching (with baseline)", binary_switching(overall_alpha, overall_beta)),
        ("Binary switching (no baseline)", binary_no_baseline(overall_beta)),
        (f"Symmetric effects (d={D_SYMMETRIC})", symmetric_effects(overall_beta, D_SYMMETRIC)),
        (f"Normal heterogeneity (σ={SIGMA_NORMAL})", normal_heterogeneity(overall_beta, SIGMA_NORMAL)),
        ("Extreme opponent reaction", extreme_opponent(overall_beta)),
    ]

    print(f"\n{'Model':<40} {'Implied Support':>15}")
    print("-" * 56)
    for name, support in models:
        print(f"{name:<40} {support:>14.1%}")

    print("\n" + "=" * 70)
    print("SECTION 4: DETAILED RESULTS BY POLICY × ENDORSER")
    print("=" * 70)

    print(f"\n{'Policy':<10} {'Endorser':<18} {'α':>6} {'β':>8} {'SE(β)':>7} {'p_binary':>8}")
    print("-" * 62)

    for _, row in results_df.iterrows():
        print(
            f"{row['policy']:<10} {row['endorser']:<18} "
            f"{row['alpha']:6.3f} {row['beta']:+8.4f} {row['se_ate']:7.4f} "
            f"{row['p_binary_switching']:8.1%}"
        )

    print("\n" + "=" * 70)
    print("SECTION 5: SUMMARY BY ENDORSER GROUP")
    print("=" * 70)

    summary_by_endorser = (
        results_df.groupby("endorser")
        .agg({
            "beta": "mean",
            "p_binary_switching": "mean",
            "p_symmetric": "mean",
            "p_normal": "mean",
            "p_extreme_opponent": "mean",
        })
        .sort_values("beta")
    )

    print(f"\n{'Endorser':<22} {'ATE':>8} {'Binary':>8} {'Symmetric':>9} {'Normal':>8} {'Extreme':>8}")
    print("-" * 68)

    for endorser, row in summary_by_endorser.iterrows():
        print(
            f"{endorser:<22} {row['beta']:+8.4f} "
            f"{row['p_binary_switching']:8.1%} "
            f"{row['p_symmetric']:9.1%} "
            f"{row['p_normal']:8.1%} "
            f"{row['p_extreme_opponent']:8.1%}"
        )

    output_path = os.path.join(TABS_DIR, "empirical_analysis.csv")
    os.makedirs(TABS_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nWrote detailed results to: {output_path}")

    variance_tex = generate_variance_table(results_df)
    variance_path = os.path.join(TABS_DIR, "variance_test.tex")
    with open(variance_path, "w") as f:
        f.write(variance_tex)
    print(f"Wrote variance test table to: {variance_path}")

    empirical_tex = generate_empirical_support_table(summary_by_endorser)
    empirical_path = os.path.join(TABS_DIR, "empirical_support.tex")
    with open(empirical_path, "w") as f:
        f.write(empirical_tex)
    print(f"Wrote empirical support table to: {empirical_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
