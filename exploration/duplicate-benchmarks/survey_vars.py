import collections
import re
import sys

import pyreadstat

df, meta = pyreadstat.read_dta(sys.argv[1], apply_value_formats=False)
df = df[df["mode"] == 1]
df.to_parquet(sys.argv[2])
print(df.shape)
neg = collections.Counter()
for v, labs in meta.variable_value_labels.items():
    for k, lab in labs.items():
        if k < 0:
            neg[(k, re.sub(r"^-?\d+\.\s*", "", lab)[:60])] += 1
for (k, lab), c in sorted(neg.items(), key=lambda t: (t[0][0], -t[1])):
    if c >= 5:
        print(k, c, lab)
prefixes = collections.Counter(n.split("_")[0] for n in meta.column_names)
print(prefixes.most_common(400))
