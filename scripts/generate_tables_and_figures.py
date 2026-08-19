#!/usr/bin/env python3
"""Generate every table and figure the manuscript uses, from the microdata.

Single writer for tabs/ and figs/: each artifact ms/endorsement.tex inputs is
produced here and nowhere else. Data: the `pakistan` dataset shipped with the
endorse R package (Blair et al. 2013); regenerate it with
scripts/download_blair_data.R.

Endorser codes follow Blair et al. (2013) Table 1, column order confirmed
against their replication files: a = control, b = Kashmiri tanzeems,
c = Afghan Taliban, d = al-Qa'ida, e = sectarian tanzeems.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "pakistan_endorsement.csv")
TABS_DIR = os.path.join(ROOT_DIR, "tabs")
FIGS_DIR = os.path.join(ROOT_DIR, "figs")

POLICIES = ["Polio", "FCR", "Durand", "Curriculum"]
ENDORSERS = {
    "b": "Kashmiri tanzeems",
    "c": "Afghan Taliban",
    "d": "Al-Qaeda",
    "e": "Sectarian tanzeems",
}

CATS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
D_SYMMETRIC = 0.05
D_SYMMETRIC_SMALL = 0.02
SIGMA_NORMAL = 0.05


def rescale(series):
    """Map the 1-5 response scale onto [0, 1]."""
    return (series.dropna() - 1) / 4


def binary_switching(alpha, beta):
    return alpha + beta


def symmetric_effects(beta, d):
    return beta / (2 * d) + 0.5


def extreme_opponent(beta):
    return 1 + beta


def normal_heterogeneity(beta, sigma):
    return 1 - stats.norm.cdf(-beta / sigma)


def discrete_scale(beta, control_dist, steps):
    """Solve beta = p * up + (1 - p) * down for p, where up/down are the
    average shifts after truncation at the scale endpoints.

    Assumes supporters and opponents share the control response distribution
    and that every respondent shifts `steps` categories.
    """
    shift = steps * 0.25
    up = float(np.dot(control_dist, np.minimum(CATS + shift, 1) - CATS))
    down = float(np.dot(control_dist, np.maximum(CATS - shift, 0) - CATS))
    p = (beta - down) / (up - down)
    return {
        "p": float(np.clip(p, 0, 1)),
        "up": up,
        "down": down,
        "ceiling_loss": shift - up,
    }


def category_distribution(values):
    return np.array([(np.abs(values - c) < 1e-9).mean() for c in CATS])


def pairing_probabilities(treat_dist, control_dist):
    """P(T > C), P(T = C), P(T < C) under independent random pairing."""
    gt = sum(
        treat_dist[i] * control_dist[j]
        for i in range(len(CATS))
        for j in range(len(CATS))
        if CATS[i] > CATS[j]
    )
    lt = sum(
        treat_dist[i] * control_dist[j]
        for i in range(len(CATS))
        for j in range(len(CATS))
        if CATS[i] < CATS[j]
    )
    return gt, 1 - gt - lt, lt


def coupling_bound(treat_dist, control_dist, maximize):
    """Extreme value of P(T > C) over all couplings of the two marginals.

    The experiment identifies only the marginal distributions; the share of
    respondents whose response the endorsement strictly raises depends on the
    joint distribution. This linear program finds its smallest and largest
    values consistent with the data.
    """
    n = len(CATS)
    sign = -1.0 if maximize else 1.0
    cost = np.array(
        [sign if CATS[i] > CATS[j] else 0.0 for i in range(n) for j in range(n)]
    )
    a_eq, b_eq = [], []
    for i in range(n):
        row = np.zeros(n * n)
        row[i * n : (i + 1) * n] = 1
        a_eq.append(row)
        b_eq.append(treat_dist[i])
    for j in range(n):
        row = np.zeros(n * n)
        row[j::n] = 1
        a_eq.append(row)
        b_eq.append(control_dist[j])
    res = linprog(cost, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=(0, None))
    return sign * res.fun if maximize else res.fun


def write_tex(filename, tex):
    path = os.path.join(TABS_DIR, filename)
    with open(path, "w") as f:
        f.write(tex.replace("%", "\\%"))
    print(f"  wrote {path}")


def load():
    df = pd.read_csv(DATA_PATH)
    control = np.concatenate([rescale(df[f"{p}.a"]).values for p in POLICIES])
    groups = {}
    for code, name in ENDORSERS.items():
        t = np.concatenate([rescale(df[f"{p}.{code}"]).values for p in POLICIES])
        groups[name] = t
    return df, control, groups


def make_variance_table(df):
    """Per policy-endorser cell: observed treatment variance against the
    p(1-p) that binary switching predicts."""
    rows = []
    for policy in POLICIES:
        alpha = rescale(df[f"{policy}.a"]).mean()
        for code, name in ENDORSERS.items():
            t = rescale(df[f"{policy}.{code}"])
            p = alpha + (t.mean() - alpha)
            rows.append((policy, name, t.var(), p * (1 - p)))

    tex = (
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "Policy & Endorser & Var(T) Obs & Var(T) Pred & Ratio \\\\\n\\midrule\n"
    )
    for policy, name, obs, pred in rows:
        tex += f"{policy} & {name} & {obs:.4f} & {pred:.4f} & {obs / pred:.2f} \\\\\n"
    avg_obs = np.mean([r[2] for r in rows])
    avg_pred = np.mean([r[3] for r in rows])
    tex += "\\midrule\n"
    tex += (
        f"\\textbf{{Average}} & & {avg_obs:.4f} & {avg_pred:.4f} "
        f"& {avg_obs / avg_pred:.2f} \\\\\n"
    )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("variance_test.tex", tex)
    return avg_obs / avg_pred


def make_method_comparison(alpha, beta, control_dist):
    d1 = discrete_scale(beta, control_dist, 1)
    d2 = discrete_scale(beta, control_dist, 2)
    rows = [
        (
            "Extreme opponent reaction",
            "$\\delta^- = -1$, $\\delta^+ = 0$",
            extreme_opponent(beta),
        ),
        (
            "Binary switching",
            "All supporters $\\to$ 1, opponents $\\to$ 0",
            binary_switching(alpha, beta),
        ),
        (
            "5-point scale, 2-step shift",
            "Shift $\\pm 2$ categories, truncated",
            d2["p"],
        ),
        (
            "5-point scale, 1-step shift",
            "Shift $\\pm 1$ category, truncated",
            d1["p"],
        ),
        (
            f"Normal heterogeneity ($\\sigma = {SIGMA_NORMAL}$)",
            f"$\\delta_i \\sim \\mathcal{{N}}(\\beta, {SIGMA_NORMAL}^2)$",
            normal_heterogeneity(beta, SIGMA_NORMAL),
        ),
        (
            f"Symmetric effects ($d = {D_SYMMETRIC}$)",
            f"$\\delta^+ = +{D_SYMMETRIC}$, $\\delta^- = -{D_SYMMETRIC}$",
            symmetric_effects(beta, D_SYMMETRIC),
        ),
        (
            f"Symmetric effects ($d = {D_SYMMETRIC_SMALL}$)",
            f"$\\delta^+ = +{D_SYMMETRIC_SMALL}$, "
            f"$\\delta^- = -{D_SYMMETRIC_SMALL}$",
            symmetric_effects(beta, D_SYMMETRIC_SMALL),
        ),
    ]
    tex = (
        "\\begin{tabular}{lll}\n\\toprule\n"
        "Model & Key Assumption & Implied Support \\\\\n\\midrule\n"
    )
    for model, assumption, support in rows:
        tex += f"{model} & {assumption} & {support:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("method_comparison.tex", tex)
    return rows


def make_discrete_table(beta, control_dist):
    tex = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Shift & Eff.\\ $\\delta^+$ & Eff.\\ $\\delta^-$ "
        "& Ceiling Loss & Implied $p$ \\\\\n\\midrule\n"
    )
    for steps in (1, 2, 3):
        r = discrete_scale(beta, control_dist, steps)
        tex += (
            f"{steps} category & {r['up']:.3f} & {r['down']:.3f} "
            f"& {r['ceiling_loss']:.3f} & {r['p']:.1%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("discrete_scale.tex", tex)


def make_empirical_support(alpha, groups):
    tex = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Endorser & ATE & Binary & Symmetric & Normal & Extreme \\\\\n"
        "\\midrule\n"
    )
    for name, t in groups.items():
        beta = t.mean() - alpha
        tex += (
            f"{name} & ${beta:+.4f}$ & "
            f"{binary_switching(alpha, beta):.1%} & "
            f"{symmetric_effects(beta, D_SYMMETRIC):.1%} & "
            f"{normal_heterogeneity(beta, SIGMA_NORMAL):.1%} & "
            f"{extreme_opponent(beta):.1%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("empirical_support.tex", tex)


def make_pairing_bounds(control, groups):
    control_dist = category_distribution(control)
    tex = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Endorser & $P(T<C)$ & $P(T=C)$ & $P(T>C)$ "
        "& Min.\\ Positive & Max.\\ Positive \\\\\n\\midrule\n"
    )
    pooled = np.concatenate(list(groups.values()))
    for name, t in list(groups.items()) + [("All groups", pooled)]:
        td = category_distribution(t)
        gt, eq, lt = pairing_probabilities(td, control_dist)
        lo = coupling_bound(td, control_dist, maximize=False)
        hi = coupling_bound(td, control_dist, maximize=True)
        prefix = "\\midrule\n" if name == "All groups" else ""
        tex += (
            f"{prefix}{name} & {lt:.0%} & {eq:.0%} & {gt:.0%} "
            f"& {lo:.0%} & {hi:.0%} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("pairing_bounds.tex", tex)


ENDORSE_SPECS = {
    "primary": "Baseline",
    "covariates": "Covariates",
    "hierarchical": "Hierarchical",
}


def load_endorse(spec):
    """Posterior summaries written by scripts/fit_endorse.R."""
    return pd.read_csv(os.path.join(ROOT_DIR, "data", f"endorse_{spec}.csv"))


def make_endorse_tables():
    primary = load_endorse("primary")
    tex = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Endorser & $P(s > 0)$ & 95% CI & Mean Reaction & Reaction SD \\\\\n"
        "\\midrule\n"
    )
    for _, row in primary.iterrows():
        prefix = "\\midrule\n" if row.group == "Average" else ""
        tex += (
            f"{prefix}{row.group} & {row.p_mean:.0%} & "
            f"[{row.p_lo:.0%}, {row.p_hi:.0%}] & "
            f"${row.mu_mean:+.2f}$ & {row.omega_mean:.2f} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("endorse_results.tex", tex)

    frames = {label: load_endorse(spec) for spec, label in ENDORSE_SPECS.items()}
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Endorser & Baseline & Covariates & Hierarchical \\\\\n\\midrule\n"
    )
    for i in range(len(primary)):
        name = primary.group.iloc[i]
        prefix = "\\midrule\n" if name == "Average" else ""
        cells = " & ".join(
            f"{df.p_mean.iloc[i]:.0%} "
            f"[{df.p_lo.iloc[i]:.0%}, {df.p_hi.iloc[i]:.0%}]"
            for df in frames.values()
        )
        tex += f"{prefix}{name} & {cells} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    write_tex("endorse_robustness.tex", tex)


def make_bar_figure(alpha, beta, rows):
    labels = {
        "Extreme opponent reaction": "Extreme\nopponent",
        "Binary switching": "Binary\nswitching",
        "5-point scale, 2-step shift": "5-pt scale\n2-step",
        "5-point scale, 1-step shift": "5-pt scale\n1-step",
        f"Normal heterogeneity ($\\sigma = {SIGMA_NORMAL}$)": "Normal\nheterog.",
        f"Symmetric effects ($d = {D_SYMMETRIC}$)": f"Symmetric\n(d={D_SYMMETRIC})",
        f"Symmetric effects ($d = {D_SYMMETRIC_SMALL}$)": (
            f"Symmetric\n(d={D_SYMMETRIC_SMALL})"
        ),
    }
    names = [labels[r[0]] for r in rows]
    estimates = [r[2] for r in rows]
    colors = [
        "#90ee90",
        "#f4a460",
        "#e8b87a",
        "#dcc69a",
        "#c4b7e0",
        "#fffacd",
        "#f5deb3",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, estimates, color=colors, edgecolor="black", lw=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=1.5, label="50% majority threshold")
    ax.set_ylabel("Implied proportion supporting group", fontsize=12)
    ax.set_title(
        "Implied support under different behavioral models\n"
        f"Blair et al. (2013): ATE = {beta:.3f}, baseline = {alpha:.3f}",
        fontsize=12,
    )
    ax.set_ylim(0, 1.08)
    for bar, est in zip(bars, estimates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{est:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "method_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figs/method_comparison.pdf")


def make_sensitivity_figure(alpha, beta):
    ate = np.linspace(-0.10, 0.10, 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        ate,
        alpha + ate,
        "-",
        lw=2.2,
        color="#d45500",
        label=f"Binary switching (baseline = {alpha:.2f})",
    )
    ax.plot(
        ate,
        np.clip(1 + ate, 0, 1),
        ":",
        lw=1.8,
        color="#228833",
        label="Extreme opponent reaction",
    )
    ax.plot(
        ate,
        np.clip(symmetric_effects(ate, D_SYMMETRIC), 0, 1),
        "-.",
        lw=1.8,
        color="#aa8822",
        label=f"Symmetric effects (d = {D_SYMMETRIC})",
    )
    ax.plot(
        ate,
        np.clip(symmetric_effects(ate, D_SYMMETRIC_SMALL), 0, 1),
        "--",
        lw=1.8,
        color="#3377bb",
        label=f"Symmetric effects (d = {D_SYMMETRIC_SMALL})",
    )
    ax.axvline(beta, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.text(
        beta + 0.003,
        0.05,
        f"Observed\nATE = {beta:.3f}",
        fontsize=9,
        color="grey",
    )
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Average treatment effect", fontsize=12)
    ax.set_ylabel("Implied proportion supporting group", fontsize=12)
    ax.set_title("Sensitivity of support estimates to the ATE", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.10, 0.10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figs/sensitivity.pdf")


def clustered_ate_se(df):
    """SE of the pooled ATE with respondent-clustered residuals."""
    frames = []
    for p in POLICIES:
        for code in list(ENDORSERS) + ["a"]:
            col = df[f"{p}.{code}"].dropna()
            frames.append(
                pd.DataFrame(
                    {
                        "id": col.index,
                        "y": (col - 1) / 4,
                        "treated": int(code != "a"),
                    }
                )
            )
    long = pd.concat(frames, ignore_index=True)
    for v in (0, 1):
        long.loc[long.treated == v, "resid"] = (
            long.loc[long.treated == v, "y"] - long.loc[long.treated == v, "y"].mean()
        )
    se2 = 0.0
    for v in (0, 1):
        sub = long[long.treated == v]
        cluster_sums = sub.groupby("id")["resid"].sum()
        se2 += cluster_sums.pow(2).sum() / len(sub) ** 2
    return np.sqrt(se2)


def main():
    os.makedirs(TABS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    df, control, groups = load()
    alpha = control.mean()
    treatment = np.concatenate(list(groups.values()))
    beta = treatment.mean() - alpha
    control_dist = category_distribution(control)

    print("Tables:")
    avg_ratio = make_variance_table(df)
    rows = make_method_comparison(alpha, beta, control_dist)
    make_discrete_table(beta, control_dist)
    make_empirical_support(alpha, groups)
    make_pairing_bounds(control, groups)
    make_endorse_tables()

    print("\nFigures:")
    make_bar_figure(alpha, beta, rows)
    make_sensitivity_figure(alpha, beta)

    print("\nNumbers for the manuscript:")
    print(f"  respondents                : {len(df)}")
    print(f"  control mean (alpha)       : {alpha:.4f}")
    print(f"  pooled ATE (beta)          : {beta:.4f}")
    print(f"  clustered SE of pooled ATE : {clustered_ate_se(df):.4f}")
    print(f"  treatment mean             : {treatment.mean():.4f}")
    print(f"  treatment SD               : {treatment.std(ddof=1):.4f}")
    print(
        f"  predicted SD under binary  : "
        f"{np.sqrt((alpha + beta) * (1 - alpha - beta)):.4f}"
    )
    print(f"  average variance ratio     : {avg_ratio:.2f}")
    print(
        f"  control distribution       : "
        f"{dict(zip(CATS, np.round(control_dist, 3)))}"
    )
    for name, t in groups.items():
        print(
            f"  {name:20s}: ATE = {t.mean() - alpha:+.4f}, " f"binary = {t.mean():.1%}"
        )


if __name__ == "__main__":
    main()
