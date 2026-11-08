"""Simulate responses from the shared-affect model on the real survey design.

Parameter recovery is the check that the model can tell omega_eta (person-level
affect, shared across endorsers) from omega_u (endorser-specific spread or noise).
Writes a long.csv with the same columns as export.R plus truth.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

here, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)

long = pd.read_csv(here / "long.csv")
people = long.groupby("respondent")["inc"].first().sort_index()
truth = {
    "beta": [0.5, 1.5, 1.2, 0.6],
    "lambda": [-0.10, -0.15, -0.10, -0.05],
    "omega_eta": 0.4,
    "omega_u": [0.2, 0.3, 0.2, 0.3],
}
control_marginal = np.array([0.055, 0.074, 0.165, 0.287, 0.419])
beta = np.array(truth["beta"])
# Cutpoints that reproduce the observed control marginal when x ~ N(0, 1).
from scipy.stats import norm  # noqa: E402

tau = [norm.ppf(np.cumsum(control_marginal)[:-1]) * np.sqrt(1 + b**2) for b in beta]
truth["tau"] = [t.tolist() for t in tau]

x = pd.Series(rng.normal(size=len(people)), index=people.index)
eta = pd.Series(rng.normal(scale=truth["omega_eta"], size=len(people)), index=people.index)
j = long["item"].to_numpy() - 1
k = long["group"].to_numpy()
treated = k > 0
lam = np.where(treated, np.array([0.0] + truth["lambda"])[k], 0.0)
u = np.where(treated, rng.normal(size=len(long)) * np.array([0.0] + truth["omega_u"])[k], 0.0)
e = np.where(treated, eta.loc[long["respondent"]].to_numpy(), 0.0)
latent = beta[j] * (x.loc[long["respondent"]].to_numpy() + lam + e + u) + rng.normal(size=len(long))
y = np.ones(len(long), dtype=int)
for jj in range(4):
    rows = j == jj
    y[rows] = 1 + np.searchsorted(tau[jj], latent[rows])
sim = long.copy()
sim["y"] = y
sim.to_csv(out / "long.csv", index=False)
json.dump(truth, open(out / "truth.json", "w"), indent=1)
print(pd.crosstab(sim["group"], sim["y"], normalize="index").round(3).to_string())
