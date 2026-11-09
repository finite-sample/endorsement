"""Near-duplicate check on Pew Global Attitudes Pakistan interviews with interviewer IDs."""

import re
import sys

import numpy as np
import pandas as pd
import pyreadstat

path, interviewer = sys.argv[1], sys.argv[2]
df, meta = pyreadstat.read_sav(path, encoding="latin1", apply_value_formats=False)
country = df["COUNTRY"].map(meta.variable_value_labels["COUNTRY"]).str.strip()
pk = df[country == "Pakistan"].reset_index(drop=True)
labels = meta.column_names_to_labels
area = [
    c
    for c in pk.columns
    if re.search(r"psu|cluster|sampling point|region|province|district", (labels.get(c) or "") + " " + c, re.I)
]
print(
    "n Pakistan:",
    len(pk),
    " candidate area vars:",
    [(c, (labels.get(c) or "")[:40], pk[c].nunique()) for c in area][:8],
)
skip = {interviewer, "COUNTRY", "WEIGHT", "weight", "ID", "id", "PSU"}
items = [
    c
    for c in pk.columns
    if re.match(r"^Q\d", c)
    and c not in skip
    and pk[c].notna().mean() >= 0.5
    and 1 < pk[c].nunique() <= 15
    and c not in area
]
print("items:", len(items))
X = np.hstack([np.stack([(pk[c] == k).to_numpy() for k in sorted(pk[c].dropna().unique())], 1) for c in items]).astype(
    np.float32
)
A = np.stack([pk[c].notna().to_numpy() for c in items], 1).astype(np.float32)
m, b = X @ X.T, A @ A.T
s = np.where(b >= 0.5 * len(items), m / np.maximum(b, 1), np.nan)
np.fill_diagonal(s, np.nan)
best = np.nanmax(s, 1)
who = pk[interviewer].to_numpy()
arg = np.nanargmax(np.where(np.isnan(s), -1, s), 1)
iu = np.triu_indices(len(pk), 1)
exact = np.nan_to_num(s[iu]) >= 0.999
share = {cut: 100 * (best >= cut).mean() for cut in (0.85, 0.90, 0.999)}
print("median best", round(float(np.median(best)), 3), "share at 85/90/100%:", share, "exact pairs:", int(exact.sum()))
same = 100 * (who[arg] == who).mean()
loads = pd.Series(who).value_counts()
print("best match same interviewer %:", round(same), "interviewers:", loads.size, "median load:", loads.median())
for k in sorted(set(who)):
    own = who == k
    high = (best[own] >= 0.85).mean()
    if high > 0.3:
        print(int(k), int(own.sum()), round(float(np.median(best[own])), 2), round(100 * high))
