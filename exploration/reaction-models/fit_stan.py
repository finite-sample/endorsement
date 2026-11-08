"""Fit the endorsement model variants with CmdStan and save draws.

usage: python3 fit_stan.py <data dir with long.csv> <variant> <warmup> <samples> [out dir] [adapt_delta]
variants: bis (independent spread only), shared (adds person-level affect), hetero (shared affect +
          free treated-item scale, i.e. heteroskedastic ordered probit), additive (reaction shifts the
          response unscaled by beta_j, with a free treated-arm loading rho on the ideal point),
          *_inc (lambda, spreads and ideal-point moments vary by income group)
"""

import sys
from pathlib import Path

import cmdstanpy
import pandas as pd

here = Path(__file__).resolve().parent
data_dir = Path(sys.argv[1])
variant, warmup, samples = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = Path(sys.argv[5]) if len(sys.argv) > 5 else data_dir / f"fit_{variant}"
shared, by_inc, free_scale, additive, free_loading = {
    "bis": (0, 0, 0, 0, 0),
    "shared": (1, 0, 0, 0, 0),
    "shared_inc": (1, 1, 0, 0, 0),
    "bis_inc": (0, 1, 0, 0, 0),
    "hetero": (1, 0, 1, 0, 0),
    "hetero_inc": (1, 1, 1, 0, 0),
    "additive": (1, 0, 1, 1, 0),
    "additive_inc": (1, 1, 1, 1, 0),
    "twofactor": (1, 0, 1, 1, 1),
}[variant]
adapt_delta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.8

long = pd.read_csv(data_dir / "long.csv")
people = long.groupby("respondent")["inc"].first().sort_index()
index = {r: i + 1 for i, r in enumerate(people.index)}
long["p"] = long["respondent"].map(index)
long["cell_inc"] = long["inc"] if by_inc else 1
long = long.sort_values(["item", "group", "cell_inc"]).reset_index(drop=True)
cells = long.groupby(["item", "group", "cell_inc"], sort=False).apply(lambda d: (d.index.min() + 1, d.index.max() + 1))
data = {
    "N": len(long),
    "P": len(people),
    "J": 4,
    "K": 4,
    "C": 3,
    "G": len(cells),
    "person": long["p"].tolist(),
    "y": long["y"].tolist(),
    "start": [a for a, _ in cells],
    "end": [b for _, b in cells],
    "cell_item": [i for i, _, _ in cells.index],
    "cell_group": [g for _, g, _ in cells.index],
    "cell_inc": [c for _, _, c in cells.index],
    "inc": people.tolist(),
    "ref": 2,
    "shared": shared,
    "by_inc": by_inc,
    "free_scale": free_scale,
    "additive": additive,
    "free_loading": free_loading,
}
# ~/.cmdstan/cmdstan-2.36.0 is an x86_64 build that cannot link on this arm64 machine;
# 2.40.0 was built natively, and cmdstanpy picks the newest install by default.
model = cmdstanpy.CmdStanModel(stan_file=str(here / "endorse_shared_v.stan"))
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=warmup,
    iter_sampling=samples,
    seed=20261105,
    output_dir=str(out),
    show_progress=False,
    adapt_delta=adapt_delta,
)
keep = (
    ["beta", "lambda", "omega_u" if not free_scale else "s_k", "mu_x", "sd_x"]
    + (["omega_eta"] if shared and not free_loading else [])
    + (["rho"] if additive else [])
    + (["gamma"] if free_loading else [])
)
summary = fit.summary()
summary = summary[summary.index.str.match("|".join(keep))]
print(summary[["Mean", "5%", "95%", "R_hat", "ESS_bulk"]].round(3).to_string())
fit.draws_pd(vars=keep).to_csv(out / "draws.csv", index=False)
print(fit.diagnose())
