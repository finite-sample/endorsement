"""Fit the latent-class (opponent / indifferent / supporter) endorsement model.

usage: python3 fit_mixture.py <data dir with long.csv> <warmup> <samples> [out dir] [adapt_delta]
       [symmetric=1] [inits.json|''] [flex_x=0] [top_class=0]
"""

import sys
from pathlib import Path

import cmdstanpy
import pandas as pd

here = Path(__file__).resolve().parent
data_dir = Path(sys.argv[1])
warmup, samples = int(sys.argv[2]), int(sys.argv[3])
out = Path(sys.argv[4]) if len(sys.argv) > 4 else data_dir / "fit_mixture"
adapt_delta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
symmetric = int(sys.argv[6]) if len(sys.argv) > 6 else 1
inits = (sys.argv[7] or None) if len(sys.argv) > 7 else None  # JSON file; '' for random inits
flex_x = int(sys.argv[8]) if len(sys.argv) > 8 else 0
top_class = int(sys.argv[9]) if len(sys.argv) > 9 else 0

long = pd.read_csv(data_dir / "long.csv")
arm = long.groupby("respondent")["group"].agg(lambda g: (g > 0).mean())
long = long[long["respondent"].map(arm).isin([0.0, 1.0])]  # drop the 35 mixed-arm respondents
long["treated"] = (long["group"] > 0).astype(int)
long = long.sort_values(["treated", "respondent", "item"]).reset_index(drop=True)
people = long["respondent"].drop_duplicates().tolist()
index = {r: i + 1 for i, r in enumerate(people)}
long["p"] = long["respondent"].map(index)
trt = long[long["treated"] == 1]
ctl = long[long["treated"] == 0]
cbounds = ctl.groupby("p").apply(lambda d: (d.index.min() + 1, d.index.max() + 1)).sort_index()
all_top = (long.groupby("p")["y"].min() == 5).astype(int).sort_index()
bounds = trt.groupby("p").apply(lambda d: (d.index.min() + 1, d.index.max() + 1)).sort_index()
data = {
    "N": len(long),
    "P": len(people),
    "J": 4,
    "N_c": int((long["treated"] == 0).sum()),
    "P_t": len(bounds),
    "person": long["p"].tolist(),
    "item": long["item"].tolist(),
    "y": long["y"].tolist(),
    "t_start": [a for a, _ in bounds],
    "t_end": [b for _, b in bounds],
    "symmetric": symmetric,
    "flex_x": flex_x,
    "top_class": top_class,
    "P_c": len(cbounds),
    "c_start": [a for a, _ in cbounds],
    "c_end": [b for _, b in cbounds],
    "all_top": all_top.tolist(),
}
model = cmdstanpy.CmdStanModel(stan_file=str(here / "endorse_mixture.stan"))
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
    inits=inits,
)
keep = [
    "beta",
    "pi",
    "delta",
    "rho",
    "s_T",
    "share",
    "mean_shift",
    "w_bump",
    "m_bump",
    "s_bump",
    "top_share",
    "favorable_",
]
summary = fit.summary()
summary = summary[summary.index.str.match("|".join(keep))]
print(summary[["Mean", "5%", "95%", "R_hat", "ESS_bulk"]].round(3).to_string())
fit.draws_pd(
    vars=["beta", "pi", "delta_neg", "delta_pos", "rho", "s_T", "mean_shift"]
    + (["w_bump", "m_bump", "s_bump"] if flex_x else [])
    + ["top_share", "favorable_lower", "favorable_upper"]
).to_csv(out / "draws.csv", index=False)
print(fit.diagnose())
