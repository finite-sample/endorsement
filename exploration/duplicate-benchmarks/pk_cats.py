"""Category-count distribution of Pakistan q-items (>=50% non-missing, >1 distinct value)."""

import sys

import pandas as pd
import pyreadstat

df, meta = pyreadstat.read_dta(sys.argv[1], apply_value_formats=False)
print(df.shape)
q = [c for c in df.columns if c.lower().startswith("q")]
rows = []
for c in q:
    s = df[c]
    if s.notna().mean() < 0.5:
        continue
    k = s.dropna().nunique()
    if k <= 1:
        continue
    rows.append((c, k, pd.api.types.is_numeric_dtype(s)))
t = pd.DataFrame(rows, columns=["var", "k", "numeric"])
print("items:", len(t), "non-numeric:", int((~t.numeric).sum()))
print(t.k.value_counts().sort_index().to_dict())
b = pd.cut(t.k, [1, 2, 3, 4, 5, 6, 1000], labels=["2", "3", "4", "5", "6", "7+"]).value_counts().sort_index()
print(b.to_dict())
print(t.sort_values("k").to_string(index=False))
