"""Turn saved draws into extensive- and intensive-margin summaries.

Reaction to endorser k for a person in income group c: s ~ N(lambda_kc, w^2).
  extensive margin: P(s > 0) = Phi(lambda / w)
  intensive margin: E[s | s > 0] = lambda + w * phi(lambda / w) / Phi(lambda / w), and E[s | s < 0] likewise
The spread w is omega_u (bis), omega_eta (shared, conservative) or
sqrt(omega_eta^2 + omega_u^2) (shared, liberal). Scale: middle-income ideal points have SD 1.
usage: python3 summarize.py <fit dir> [<fit dir> ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ENDORSERS = ["Kashmiri tanzeems", "Afghan Taliban", "Al-Qaeda", "Sectarian tanzeems"]
INCOME = {1: "low", 2: "middle", 3: "high"}


def margins(lam, w):
    z = lam / w
    # log-space ratios keep the conditional means finite when P(s>0) is near 0 or 1
    return (
        norm.cdf(z),
        lam + w * np.exp(norm.logpdf(z) - norm.logcdf(z)),
        lam - w * np.exp(norm.logpdf(z) - norm.logsf(z)),
    )


def ci(v):
    return f"{np.mean(v):6.3f} [{np.percentile(v, 2.5):6.3f},{np.percentile(v, 97.5):6.3f}]"


for fit_dir in sys.argv[1:]:
    d = pd.read_csv(Path(fit_dir) / "draws.csv")
    two_factor = "gamma[1]" in d.columns
    shared = two_factor or any(c.startswith("omega_eta") for c in d.columns)
    n_inc = len({c.split("[")[1].split(",")[1] for c in d.columns if c.startswith("lambda[")})
    print(f"\n=== {fit_dir}: {'shared affect' if shared else 'BIS'}, {'by income' if n_inc > 1 else 'pooled'}")
    rows = []
    for c in range(1, n_inc + 1):
        label = INCOME[c] if n_inc > 1 else "all"
        w_eta = np.ones(len(d)) if two_factor else (d[f"omega_eta[{c}]"].to_numpy() if shared else None)
        for k, name in enumerate(ENDORSERS, 1):
            lam = d[f"lambda[{k},{c}]"].to_numpy()
            free_scale = f"s_k[{k}]" in d.columns
            w_u = None if free_scale else d[f"omega_u[{k}]"].to_numpy()
            if two_factor:
                specs = {"two-factor, eta SD fixed at 1": w_eta}
            elif not shared:
                specs = {"independent spread (BIS)": w_u}
            elif free_scale:
                specs = {f"shared only; treated-item scale s_k = {ci(d[f's_k[{k}]'])}": w_eta}
            else:
                specs = {
                    "conservative: shared only": w_eta,
                    "liberal: shared + independent": np.sqrt(w_eta**2 + w_u**2),
                }
            for spec, w in specs.items():
                p, up, down = margins(lam, w)
                rows.append([label, name, spec, ci(lam), ci(w), ci(p), ci(up), ci(down)])
        if two_factor:
            for i, item in enumerate(["Polio", "FCR", "Durand", "Curriculum"], 1):
                rows.append([label, f"(loading {item})", "gamma_j", "", ci(d[f"gamma[{i}]"]), "", "", ""])
            rows.append([label, "(rho)", "treated loading on x", "", ci(d["rho[1]"]), "", "", ""])
        elif shared:
            rows.append([label, "(all)", "omega_eta", "", ci(w_eta), "", "", ""])
    out = pd.DataFrame(
        rows, columns=["income", "endorser", "spread used", "lambda", "spread", "P(s>0)", "E[s|s>0]", "E[s|s<0]"]
    )
    print(out.to_string(index=False))
    if n_inc > 1:
        print("ideal points by income (middle fixed at N(0,1)):")
        for c in range(1, n_inc + 1):
            print(f"  {INCOME[c]:7s} mean {ci(d[f'mu_x[{c}]'])}  sd {ci(d[f'sd_x[{c}]'])}")
