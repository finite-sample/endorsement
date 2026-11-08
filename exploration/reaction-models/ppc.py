"""Posterior predictive checks for a saved fit.

For each of D posterior draws, simulate a fresh survey on the real design (new ideal
points and reactions, same respondents' arms, items and endorsers), then compare with the
observed data on (a) the response distribution by endorser and (b) the moment diagnostic
from moments.R: mean item variance and mean cross-item covariance, treated minus control.
usage: python3 ppc.py <data dir> <fit dir> [draws=200]
"""

import sys
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

data_dir, fit_dir = Path(sys.argv[1]), Path(sys.argv[2])
D = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rng = np.random.default_rng(7)

long = pd.read_csv(data_dir / "long.csv")
people = long.groupby("respondent")["inc"].first().sort_index()
pidx = long["respondent"].map({r: i for i, r in enumerate(people.index)}).to_numpy()
j = long["item"].to_numpy() - 1
k = long["group"].to_numpy()
inc = people.to_numpy()
treated = k > 0

fit = cmdstanpy.from_csv([str(f) for f in sorted(fit_dir.glob("endorse_shared*_[0-9].csv"))])
draws = fit.draws_pd()
shared = any(c.startswith("omega_eta") for c in draws.columns)
n_inc = len({c.split(",")[1] for c in draws.columns if c.startswith("lambda[")})
pick = rng.choice(len(draws), D, replace=False)


def arm_moments(y_long):
    """Mean item variance and mean cross-item covariance by arm, on the 0-1 response scale."""
    wide = pd.DataFrame({"p": pidx, "j": j, "y": (y_long - 1) / 4}).pivot(index="p", columns="j", values="y")
    arm = pd.Series(treated).groupby(pidx).all()
    out = {}
    for name, sel in [("T", arm), ("C", ~pd.Series(treated).groupby(pidx).any())]:
        out[name] = wide[sel.to_numpy()].cov().to_numpy()
    diff = out["T"] - out["C"]
    return np.concatenate([[np.mean(np.diag(diff)), np.mean(diff[np.triu_indices(4, 1)])], diff[np.triu_indices(4, 1)]])


def simulate(row):
    c_p = inc if n_inc > 1 else np.ones_like(inc)
    mu = np.array([row[f"mu_x[{c}]"] for c in range(1, 4)])
    sd = np.array([row[f"sd_x[{c}]"] for c in range(1, 4)])
    x = mu[c_p - 1] + sd[c_p - 1] * rng.normal(size=len(people))
    beta = np.array([row[f"beta[{i}]"] for i in range(1, 5)])
    tau = np.array([[row[f"tau[{i},{t}]"] for t in range(1, 5)] for i in range(1, 5)])
    lam = np.array([[0.0] * n_inc] + [[row[f"lambda[{g},{c}]"] for c in range(1, n_inc + 1)] for g in range(1, 5)])
    free_scale = "s_k[1]" in row.index
    w_u = np.zeros(5) if free_scale else np.array([0.0] + [row[f"omega_u[{g}]"] for g in range(1, 5)])
    noise_sd = np.array([1.0] + [row[f"s_k[{g}]"] for g in range(1, 5)])[k] if free_scale else 1.0
    c_row = c_p[pidx] - 1
    react = lam[k, c_row] + w_u[k] * rng.normal(size=len(long))
    if shared:
        w_eta = (
            np.ones(n_inc)
            if "gamma[1]" in row.index
            else np.array([row[f"omega_eta[{c}]"] for c in range(1, n_inc + 1)])
        )
        eta = w_eta[c_p - 1] * rng.normal(size=len(people))
        react = react + treated * eta[pidx]
    if "gamma[1]" in row.index:
        react = react * np.array([row[f"gamma[{i}]"] for i in range(1, 5)])[j]
    if "rho[1]" in row.index:
        latent = np.where(
            treated, row["rho[1]"] * beta[j] * x[pidx] + react, beta[j] * x[pidx]
        ) + noise_sd * rng.normal(size=len(long))
    else:
        latent = beta[j] * (x[pidx] + react) + noise_sd * rng.normal(size=len(long))
    return 1 + (latent[:, None] > tau[j]).sum(axis=1)


obs_dist = pd.crosstab(k, long["y"].to_numpy(), normalize="index")
rep_dist = np.zeros((D,) + obs_dist.shape)
rep_mom = np.zeros((D, 8))
for d, i in enumerate(pick):
    y = simulate(draws.iloc[i])
    rep_dist[d] = pd.crosstab(k, y, normalize="index").reindex(columns=[1, 2, 3, 4, 5], fill_value=0).to_numpy()
    rep_mom[d] = arm_moments(y)

print(f"=== {fit_dir} ({'shared' if shared else 'BIS'}, {n_inc} income group(s)), {D} replicated surveys")
print("\nresponse distribution: observed vs replicated mean [95% interval], rows = endorser (0 = control)")
for g in range(5):
    cells = []
    for y in range(5):
        lo, hi = np.percentile(rep_dist[:, g, y], [2.5, 97.5])
        flag = "*" if not lo <= obs_dist.iloc[g, y] <= hi else " "
        cells.append(f"{obs_dist.iloc[g, y]:.3f} vs {rep_dist[:, g, y].mean():.3f}[{lo:.3f},{hi:.3f}]{flag}")
    print(f"  {g}: " + " | ".join(cells))
obs_mom = arm_moments(long["y"].to_numpy())
PAIRS = ["Polio x FCR", "Polio x Durand", "Polio x Curric", "FCR x Durand", "FCR x Curric", "Durand x Curric"]
for name, o, r in [
    ("variance diff", obs_mom[0], rep_mom[:, 0]),
    ("mean covariance diff", obs_mom[1], rep_mom[:, 1]),
] + [(f"cov diff {PAIRS[i]}", obs_mom[2 + i], rep_mom[:, 2 + i]) for i in range(6)]:
    lo, hi = np.percentile(r, [2.5, 97.5])
    flag = "*" if not lo <= o <= hi else " "
    print(
        f"{name:28s} (T - C): observed {o:+.4f}; replicated {r.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{flag}",
        f"P(rep >= obs) = {np.mean(r >= o):.3f}",
    )
