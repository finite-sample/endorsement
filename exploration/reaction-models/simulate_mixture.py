"""Simulate from the latent-class model on the real design.
usage: python3 simulate_mixture.py <data dir> <out dir> <pi_neg> <pi_ind> <pi_pos> <delta_neg> <delta_pos>
       [seed] [top_share]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

here, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
pi = np.array([float(a) for a in sys.argv[3:6]])
delta_neg, delta_pos = float(sys.argv[6]), float(sys.argv[7])
rng = np.random.default_rng(int(sys.argv[8]) if len(sys.argv) > 8 else 1)
top_share = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0

long = pd.read_csv(here / "long.csv")
people = long.groupby("respondent")["inc"].first().sort_index()
truth = {
    "top_share": top_share,
    "pi": pi.tolist(),
    "delta_neg": delta_neg,
    "delta_pos": delta_pos,
    "rho": 0.9,
    "s_T": 0.95,
    "beta": [0.24, 1.7, 1.4, 0.43],
}
beta = np.array(truth["beta"])
control_marginal = np.array([0.055, 0.074, 0.165, 0.287, 0.419])
tau = [norm.ppf(np.cumsum(control_marginal)[:-1]) * np.sqrt(1 + b**2) for b in beta]

pidx = long["respondent"].map({r: i for i, r in enumerate(people.index)}).to_numpy()
j = long["item"].to_numpy() - 1
treated = (long["group"] > 0).to_numpy()
x = rng.normal(size=len(people))
cls = rng.choice(3, size=len(people), p=pi)
shift = np.array([-delta_neg, 0.0, delta_pos])[cls]
latent = np.where(
    treated,
    truth["rho"] * beta[j] * x[pidx] + shift[pidx] + truth["s_T"] * rng.normal(size=len(long)),
    beta[j] * x[pidx] + rng.normal(size=len(long)),
)
y = np.ones(len(long), dtype=int)
for jj in range(4):
    rows = j == jj
    y[rows] = 1 + np.searchsorted(tau[jj], latent[rows])
top = rng.random(len(people)) < top_share
y[top[pidx]] = 5  # always-top respondents answer the top category everywhere, in either arm
sim = long.copy()
sim["y"] = y
sim.to_csv(out / "long.csv", index=False)
json.dump(truth, open(out / "truth.json", "w"), indent=1)
print(pd.crosstab(sim["group"] > 0, sim["y"], normalize="index").round(3).to_string())
