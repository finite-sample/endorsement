"""Percent match on ANES 2012 FTF subsets built to mimic the Pakistan item profile.

Pakistan (q-items with >=50% non-missing, >1 distinct value, DK/refused codes
counted): 92 items with 2:46, 4:7, 5:7, 7:29, 10:2, 12:1 categories.
ANES pool: the 514 interviewer-administered items. Categories are counted the
same way (distinct non-missing codes, -8/-9 kept). Exact counts are used where
ANES has enough items; 7 -> 6-8 and 10/12 -> 9-15 otherwise.
"""

import json
import sys

import numpy as np
import pandas as pd

SRC, PARQ, OUT, PK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = ["percent_match.py", SRC, PARQ, OUT]
head = open("percent_match.py").read().split("results = {}")[0]
g = {}
exec(compile(head, "percent_match_head", "exec"), g)
X, items, casi, psu, n = g["X"], g["items"], g["casi"], g["psu"], g["n"]
pool = [v for v, c in zip(items, casi) if not c]
k = X[pool].nunique()


def chance_agree(s):
    p = s.dropna().value_counts(normalize=True)
    return float((p**2).sum())


agree = pd.Series({v: chance_agree(X[v]) for v in pool})

PK_PROFILE = [((2, 2), 46), ((4, 4), 7), ((5, 5), 7), ((6, 8), 29), ((9, 15), 3)]
avail = {f"{lo}-{hi}": int(((k >= lo) & (k <= hi)).sum()) for (lo, hi), _ in PK_PROFILE}
print("ANES pool items per bin:", avail)

# Pakistan chance agreement, for comparison of item homogeneity.
import pyreadstat  # noqa: E402

pk, _ = pyreadstat.read_dta(PK, apply_value_formats=False)
pkq = [
    c for c in pk.columns if c.lower().startswith("q") and pk[c].notna().mean() >= 0.5 and pk[c].dropna().nunique() > 1
]
pk_agree = pd.Series({c: chance_agree(pk[c]) for c in pkq})
pk_k = pd.Series({c: pk[c].dropna().nunique() for c in pkq})
print("Pakistan items:", len(pkq), "mean chance agreement:", round(pk_agree.mean(), 4))
print("Pakistan chance agreement by k:", pk_agree.groupby(pk_k).mean().round(3).to_dict())
print(
    "ANES pool chance agreement by k-bin:",
    {f"{lo}-{hi}": round(float(agree[(k >= lo) & (k <= hi)].mean()), 3) for (lo, hi), _ in PK_PROFILE},
)


def one_hot(cols):
    blocks = []
    for v in cols:
        codes, _ = pd.factorize(X[v])
        oh = np.zeros((n, codes.max() + 1), dtype=np.float32)
        ok = codes >= 0
        oh[np.flatnonzero(ok), codes[ok]] = 1
        blocks.append(oh)
    return np.hstack(blocks), X[cols].notna().to_numpy(dtype=np.float32)


same_psu = psu[:, None] == psu[None, :]
iu = np.triu_indices(n, 1)


def evaluate(cols):
    OH, A = one_hot(cols)
    match = OH @ OH.T
    joint = A @ A.T
    S = np.where(joint >= 0.5 * len(cols), match / np.maximum(joint, 1), np.nan)
    np.fill_diagonal(S, np.nan)
    out = {}
    for scope, M in [("whole", None), ("within_psu", same_psu)]:
        T = S if M is None else np.where(M, S, np.nan)
        valid = ~np.all(np.isnan(T), axis=1)
        mx = np.nanmax(np.where(np.isnan(T), -1, T), axis=1)[valid]
        dup = np.isclose(T[iu], 1.0)
        out[scope] = {
            "n_resp": int(valid.sum()),
            "median_max": float(np.median(mx)),
            "share_ge_85": float(np.mean(mx >= 0.85)),
            "share_ge_90": float(np.mean(mx >= 0.90)),
            "share_exact_twin": float(np.mean(np.isclose(mx, 1.0))),
            "exact_dup_pairs": int(dup.sum()),
        }
    return out


rng = np.random.default_rng(92)
rows = []
for r in range(200):
    cols = []
    for (lo, hi), m in PK_PROFILE:
        cand = k.index[(k >= lo) & (k <= hi)].to_numpy()
        cols += list(rng.choice(cand, m, replace=m > cand.size))
    res = evaluate(cols)
    row = {"subset": r, "mean_chance_agree": float(agree[cols].mean())}
    for scope, d in res.items():
        row.update({f"{scope}_{kk}": vv for kk, vv in d.items()})
    rows.append(row)
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/benchmark_subsets.csv", index=False)
summ = t.drop(columns="subset").agg(["median", lambda s: s.quantile(0.95), "max"]).T
summ.columns = ["median", "p95", "max"]
print(summ.round(4).to_string())
with open(f"{OUT}/benchmark_summary.json", "w") as fh:
    json.dump(
        {"avail": avail, "pk_mean_chance_agree": pk_agree.mean(), "summary": summ.to_dict(orient="index")}, fh, indent=1
    )
