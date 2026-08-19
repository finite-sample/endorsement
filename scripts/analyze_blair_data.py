#!/usr/bin/env python3
"""Console diagnostics for the Blair et al. (2013) endorsement data.

Prints cell-level estimates, standard errors, and the variance test so the
manuscript numbers can be eyeballed against Blair et al.'s published Table 1.
Writes nothing: every table and figure the manuscript uses comes from
generate_tables_and_figures.py.
"""

import numpy as np
import pandas as pd
from generate_tables_and_figures import (
    DATA_PATH,
    ENDORSERS,
    POLICIES,
    binary_switching,
    rescale,
)

# Blair et al. (2013), Table 1, Panel A (weighted difference in means).
BLAIR_TABLE1 = {
    "Kashmiri tanzeems": -0.011,
    "Afghan Taliban": -0.015,
    "Al-Qaeda": -0.010,
    "Sectarian tanzeems": -0.008,
}
BLAIR_ALPHA = 0.796


def cell_estimates(df):
    rows = []
    for policy in POLICIES:
        control = rescale(df[f"{policy}.a"])
        for code, name in ENDORSERS.items():
            treatment = rescale(df[f"{policy}.{code}"])
            alpha = control.mean()
            beta = treatment.mean() - alpha
            se = np.sqrt(
                control.var() / len(control) + treatment.var() / len(treatment)
            )
            rows.append(
                {
                    "policy": policy,
                    "endorser": name,
                    "n_control": len(control),
                    "n_treatment": len(treatment),
                    "alpha": alpha,
                    "beta": beta,
                    "se": se,
                    "var_treatment": treatment.var(),
                    "p_binary": binary_switching(alpha, beta),
                }
            )
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)
    cells = cell_estimates(df)
    print(f"Loaded {len(df)} respondents; {len(cells)} policy x endorser cells\n")

    print("Unweighted estimates against Blair et al. Table 1 (weighted):")
    print(
        f"  control mean: {cells['alpha'].mean():.3f} here "
        f"vs {BLAIR_ALPHA:.3f} published"
    )
    for name, published in BLAIR_TABLE1.items():
        beta = cells.loc[cells.endorser == name, "beta"].mean()
        print(f"  {name:20s}: {beta:+.4f} here vs {published:+.3f} published")

    print("\nVariance test (observed treatment variance / p(1-p)):")
    ratio = cells.var_treatment / (cells.p_binary * (1 - cells.p_binary))
    for _, row in cells.iterrows():
        r = row.var_treatment / (row.p_binary * (1 - row.p_binary))
        print(
            f"  {row.policy:<11s} {row.endorser:<19s} "
            f"beta {row.beta:+.4f} (se {row.se:.4f})  ratio {r:.2f}"
        )
    print(f"  average ratio: {ratio.mean():.2f}")


if __name__ == "__main__":
    main()
