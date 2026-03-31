#!/usr/bin/env python3
"""
Generate all tables and figures for:
  "Inferring Support from Endorsement Experiments"

Outputs:
  tabs/blair_reinterpretation.tex
  tabs/method_comparison.tex
  tabs/simulation_results.tex
  figs/method_comparison.pdf
  figs/sensitivity.pdf

Usage:
  python scripts/generate_tables_and_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Paths - output to repo root, not script directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TABS_DIR = os.path.join(ROOT_DIR, "tabs")
FIGS_DIR = os.path.join(ROOT_DIR, "figs")

# ---------------------------------------------------------------------------
# Empirical parameters from Blair et al. (2013) data (via endorse R package)
# ---------------------------------------------------------------------------
BASELINE = 0.736         # overall control mean across all policy x endorser pairs
ATE_OVERALL = -0.013     # overall ATE across all policy x endorser pairs

# ATEs by endorser group (averaged across policies)
GROUPS = {
    "Afghan Taliban":      -0.0175,
    "Al-Qaeda":            -0.0138,
    "Kashmiri Militants":  -0.0111,
    "Sectarian Groups":    -0.0082,
}

# Symmetric-effects magnitude used in the paper
D_SYMMETRIC = 0.05

# Normal heterogeneity sigma used in the paper
SIGMA_NORMAL = 0.05

# Ensure output dirs exist
os.makedirs(TABS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def binary_switching(alpha, beta):
    """Treatment average = alpha + beta = p under binary switching."""
    return alpha + beta


def binary_no_baseline(beta):
    """p_max = (beta + 1) / 2 when baseline is unavailable."""
    return (beta + 1) / 2


def symmetric_effects(beta, d):
    """p = beta / (2d) + 0.5."""
    return beta / (2 * d) + 0.5


def extreme_opponent(beta):
    """Opponents react maximally (delta_opp = -1, delta_supp = 0): p = 1 + beta."""
    return 1 + beta


def normal_heterogeneity(beta, sigma):
    """Fraction with positive individual effects when delta_i ~ N(beta, sigma^2)."""
    return 1 - stats.norm.cdf(-beta / sigma)


def hungarian_max_positive(n, ate, baseline=0.80, seed=42):
    """
    Upper bound on positive individual effects via Hungarian matching.

    In a between-subjects design, individual effects Y_i(1) - Y_i(0) are
    not observed. Given the marginal distributions of control and treatment,
    the Hungarian algorithm finds the pairing that maximises the number of
    positive "effects" (treatment_j > control_i).
    """
    rng = np.random.default_rng(seed)

    # Control: Beta with mean = baseline
    conc = 20
    a_param = baseline * conc
    b_param = (1 - baseline) * conc
    control = rng.beta(a_param, b_param, n)

    # Treatment: same base distribution, shifted to hit target ATE
    treatment = rng.beta(a_param, b_param, n)
    shift = ate - (treatment.mean() - control.mean())
    treatment = np.clip(treatment + shift, 0, 1)
    # iterate shift to nail ATE after clipping
    for _ in range(50):
        current_ate = treatment.mean() - control.mean()
        treatment = np.clip(treatment + (ate - current_ate), 0, 1)

    # Cost matrix: -1 for positive-effect pairs (we minimise total cost)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if treatment[j] - control[i] > 0:
                cost[i, j] = -1

    row_idx, col_idx = linear_sum_assignment(cost)
    effects = treatment[col_idx] - control[row_idx]
    pos = int(np.sum(effects > 0))
    return pos, n, treatment.mean() - control.mean()


def write_tex(filename, tex):
    """Write tex string to file in TABS_DIR, escaping bare % for LaTeX."""
    path = os.path.join(TABS_DIR, filename)
    with open(path, "w") as f:
        f.write(tex.replace("%", "\\%"))
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Table 1: Blair reinterpretation by group
# ---------------------------------------------------------------------------

def make_blair_table():
    tex = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Group & ATE & Blair et al. & Implied Support \\\\\n"
        "\\midrule\n"
    )
    for group, ate in GROUPS.items():
        support = binary_switching(BASELINE, ate)
        tex += f"{group} & ${ate:.3f}$ & ``Low regard'' & {support:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("blair_reinterpretation.tex", tex)


# ---------------------------------------------------------------------------
# Table 2: Method comparison
# ---------------------------------------------------------------------------

def make_method_comparison_table():
    beta = ATE_OVERALL
    rows = [
        ("Binary switching (with baseline)",
         "All supporters $\\to$ 1, opponents $\\to$ 0",
         binary_switching(BASELINE, beta)),
        ("Binary switching (no baseline)",
         "Symmetric switching, no constant",
         binary_no_baseline(beta)),
        (f"Symmetric effects ($d = {D_SYMMETRIC}$)",
         f"$\\delta^+ = +{D_SYMMETRIC}$, $\\delta^- = -{D_SYMMETRIC}$",
         symmetric_effects(beta, D_SYMMETRIC)),
        (f"Normal heterogeneity ($\\sigma = {SIGMA_NORMAL}$)",
         f"$\\delta_i \\sim \\mathcal{{N}}({beta}, {SIGMA_NORMAL}^2)$",
         normal_heterogeneity(beta, SIGMA_NORMAL)),
        ("Extreme opponent reaction",
         "$\\delta^- = -1$, $\\delta^+ = 0$",
         extreme_opponent(beta)),
    ]

    tex = (
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Model & Key Assumption & Implied Support \\\\\n"
        "\\midrule\n"
    )
    for model, assumption, support in rows:
        tex += f"{model} & {assumption} & {support:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("method_comparison.tex", tex)


# ---------------------------------------------------------------------------
# Table 3: Simulation results (Hungarian optimisation)
# ---------------------------------------------------------------------------

def make_simulation_table():
    """
    Compute two quantities for each sample size:
      1. P(T > C) under random pairing (analytical + simulated)
      2. Hungarian upper bound on positive individual effects
    """
    sample_sizes = [100, 500, 1000]

    tex = (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "$n$ & Observed ATE & Random Pairing $P(T > C)$ & Hungarian Max \\\\\n"
        "\\midrule\n"
    )
    for n in sample_sizes:
        pos_hung, total, realised_ate = hungarian_max_positive(n, ATE_OVERALL)
        # P(T > C) under random pairing via simulation
        rng = np.random.default_rng(123)
        conc = 20
        control = rng.beta(BASELINE * conc, (1 - BASELINE) * conc, 100_000)
        treatment = rng.beta(BASELINE * conc, (1 - BASELINE) * conc, 100_000)
        shift = ATE_OVERALL - (treatment.mean() - control.mean())
        treatment = np.clip(treatment + shift, 0, 1)
        p_t_gt_c = np.mean(
            rng.choice(treatment, 100_000) > rng.choice(control, 100_000)
        )
        tex += (
            f"{n} & ${realised_ate:.3f}$ & "
            f"{p_t_gt_c:.0%} & {pos_hung / total:.0%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("simulation_results.tex", tex)


# ---------------------------------------------------------------------------
# Figure 1: Method comparison bar chart
# ---------------------------------------------------------------------------

def make_method_comparison_figure():
    beta = ATE_OVERALL
    methods = [
        "Binary switching\n(with baseline)",
        "Binary switching\n(no baseline)",
        f"Symmetric\neffects (d={D_SYMMETRIC})",
        f"Normal heterog.\n(\u03c3={SIGMA_NORMAL})",
        "Extreme\nopponent reaction",
    ]
    estimates = [
        binary_switching(BASELINE, beta),
        binary_no_baseline(beta),
        symmetric_effects(beta, D_SYMMETRIC),
        normal_heterogeneity(beta, SIGMA_NORMAL),
        extreme_opponent(beta),
    ]
    colors = ["#f4a460", "#a8d5e5", "#fffacd", "#c4b7e0", "#90ee90"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(methods, estimates, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5,
               label="50% majority threshold")
    ax.set_ylabel("Implied proportion supporting group", fontsize=12)
    ax.set_title(
        "Implied support under different behavioural models\n"
        f"Blair et al. (2013): ATE = {beta}, baseline = {BASELINE}",
        fontsize=13,
    )
    ax.set_ylim(0, 1.08)

    for bar, est in zip(bars, estimates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{est:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, "method_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Sensitivity analysis
# ---------------------------------------------------------------------------

def make_sensitivity_figure():
    ate_range = np.linspace(-0.10, 0.10, 200)

    baseline_line = BASELINE + ate_range
    binary_line = (ate_range + 1) / 2
    extreme_line = np.clip(1 + ate_range, 0, 1)
    symmetric_line = np.clip(ate_range / (2 * D_SYMMETRIC) + 0.5, 0, 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ate_range, baseline_line, "-", lw=2.2, color="#d45500",
            label="Binary switching (with baseline)")
    ax.plot(ate_range, binary_line, "--", lw=1.8, color="#3377bb",
            label="Binary switching (no baseline)")
    ax.plot(ate_range, extreme_line, ":", lw=1.8, color="#228833",
            label="Extreme opponent reaction")
    ax.plot(ate_range, symmetric_line, "-.", lw=1.8, color="#aa8822",
            label=f"Symmetric effects (d={D_SYMMETRIC})")

    ax.axvline(ATE_OVERALL, color="grey", linestyle="--", lw=1.2, alpha=0.7)
    ax.text(ATE_OVERALL + 0.003, 0.05,
            f"Observed\nATE = {ATE_OVERALL}", fontsize=9, color="grey")
    ax.axhline(0.5, color="red", linestyle="--", lw=1, alpha=0.5)

    ax.set_xlabel("Average treatment effect", fontsize=12)
    ax.set_ylabel("Implied proportion supporting group", fontsize=12)
    ax.set_title("Sensitivity of support estimates to the ATE", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.10, 0.10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, "sensitivity.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating tables and figures ...\n")

    print("Tables:")
    make_blair_table()
    make_method_comparison_table()
    make_simulation_table()

    print("\nFigures:")
    make_method_comparison_figure()
    make_sensitivity_figure()

    print("\nDone.")
