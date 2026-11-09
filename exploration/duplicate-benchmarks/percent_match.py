"""Whole-questionnaire near-duplicate screen (Kuriakose & Robbins 2016) for ANES 2012 FTF.

Max percent match for respondent i = max over j != i of
(# items with identical values) / (# items both answered), among pairs that
jointly answered >= 50% of the items used.
"""

import json
import re
import sys

import numpy as np
import pandas as pd
import pyreadstat

SRC, PARQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
IWR = "iwrdesc_pre_iwrid"
rng = np.random.default_rng(2016)

_, meta = pyreadstat.read_dta(SRC, metadataonly=True)
d = pd.read_parquet(PARQ).reset_index(drop=True)
labels = dict(zip(meta.column_names, meta.column_labels))
vlabs = meta.variable_value_labels

EXCL_PREFIX = (
    "randordq",
    "randord",
    "randopts",
    "randsplice",
    "rand_",
    "paprofile",
    "profile",
    "hhlist",
    "admin",
    "iwrdesc",
    "iwrobspre",
    "iwrobspost",
    "dwell",
    "recontact",
    "sample",
    "weight",
    "psu",
    "strata",
    "version",
    "caseid",
    "ftf_",
    "wave_",
    "mode",
    "casistart",
    "invited",
    "gend_gendobs",
    "rstype",
    "stype",
)
EXCL_LABEL = re.compile(
    r"SUMMARY|SUMARY|RESTRICTED|specify|IWR OBS|INTERVIEWER|CHECKPOINT|CKPT|mention|"
    r"\(no text\)|date|time|duration|version|randomiz|preload|DK/RF sel",
    re.I,
)
MISSING_NEG = {-1, -3, -4, -5, -6, -7}
MISS_M2 = re.compile(r"missing|available|text|restricted", re.I)

rule = {}
items = []
X = {}
for v in d.columns:
    lab = labels.get(v) or ""
    if v.startswith(EXCL_PREFIX):
        rule[v] = "admin/design/iwr/random-order prefix"
        continue
    if v.endswith("_x") or v.endswith("_oth") or v.endswith("_dsc") or v.endswith("_dkrf"):
        rule[v] = "derived/other-specify suffix"
        continue
    if EXCL_LABEL.search(lab):
        rule[v] = "label: summary/restricted/specify/obs/mention/date"
        continue
    s = d[v]
    if not pd.api.types.is_numeric_dtype(s):
        rule[v] = "string (open-ended)"
        continue
    s = s.astype(float).copy()
    s[s.isin(MISSING_NEG)] = np.nan
    m2 = vlabs.get(v, {}).get(-2)
    if m2 is None or MISS_M2.search(m2):
        s[s == -2] = np.nan
    if s.isna().mean() > 0.5:
        rule[v] = ">50% missing in FTF"
        continue
    if s.nunique() > 110:
        rule[v] = ">110 distinct values"
        continue
    if s.nunique() < 2:
        rule[v] = "constant"
        continue
    rule[v] = "USED"
    items.append(v)
    X[v] = s
X = pd.DataFrame(X)
casi = np.array([bool(re.search(r"CASI", labels.get(v) or "")) for v in items])
print("rule counts:", pd.Series(rule).value_counts().to_dict())
print("items used:", len(items), "of which CASI:", int(casi.sum()))
pd.DataFrame({"var": list(rule), "rule": list(rule.values()), "label": [labels.get(v) for v in rule]}).to_csv(
    f"{OUT}/item_selection.csv", index=False
)

n = len(d)
iwr = d[IWR].to_numpy()
psu = d["psu_ftf"].to_numpy()
strat = d["strata_ftf"].to_numpy()


def similarity(cols):
    """Return S (match share, NaN where pair ineligible), joint counts."""
    sub = X[cols]
    blocks = []
    for v in cols:
        codes, _ = pd.factorize(sub[v])  # NaN -> -1
        k = codes.max() + 1
        oh = np.zeros((n, k), dtype=np.float32)
        ok = codes >= 0
        oh[np.flatnonzero(ok), codes[ok]] = 1
        blocks.append(oh)
    OH = np.hstack(blocks)
    A = sub.notna().to_numpy(dtype=np.float32)
    match = np.zeros((n, n), dtype=np.float32)
    joint = np.zeros((n, n), dtype=np.float32)
    for lo in range(0, n, 512):
        match[lo : lo + 512] = OH[lo : lo + 512] @ OH.T
        joint[lo : lo + 512] = A[lo : lo + 512] @ A.T
    S = np.where(joint >= 0.5 * len(cols), match / np.maximum(joint, 1), np.nan)
    np.fill_diagonal(S, np.nan)
    return S, OH.shape[1]


def group_stats(S, g, best, valid, maxpm):
    """Per-interviewer stats for label vector g."""
    ids, gi = np.unique(g, return_inverse=True)
    k = ids.size
    G = np.zeros((n, k), dtype=np.float32)
    G[np.arange(n), gi] = 1
    S0, V = S
    within_sum = np.einsum("ik,ik->k", G, S0 @ G) / 2
    within_cnt = np.einsum("ik,ik->k", G, V @ G) / 2
    has = ~np.isnan(maxpm)
    cnt = np.bincount(gi, weights=has, minlength=k)
    mean_max = np.bincount(gi, weights=np.where(has, maxpm, 0), minlength=k) / np.maximum(cnt, 1)
    s85 = np.bincount(gi, weights=has & (maxpm >= 0.85), minlength=k) / np.maximum(cnt, 1)
    same = np.where(valid, g == g[np.where(valid, best, 0)], False)
    sbest = np.bincount(gi, weights=same, minlength=k) / np.maximum(cnt, 1)
    wmean = within_sum / np.maximum(within_cnt, 1)
    return ids, mean_max, s85, sbest, wmean


def permute_within(g, block):
    out = g.copy()
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        out[idx] = g[rng.permutation(idx)]
    return out


results = {}
for name, mask in [("all", np.ones(len(items), bool)), ("interviewer_administered", ~casi)]:
    cols = [v for v, m in zip(items, mask) if m]
    S, ncat = similarity(cols)
    valid = ~np.all(np.isnan(S), axis=1)
    maxpm = np.where(valid, np.nanmax(np.where(np.isnan(S), -1, S), axis=1), np.nan)
    best = np.where(valid, np.nanargmax(np.where(np.isnan(S), -1, S), axis=1), -1)
    ties = np.where(valid, np.sum(np.isclose(S, maxpm[:, None]), axis=1), 0)
    same_iwr = valid & (iwr == iwr[np.where(valid, best, 0)])
    same_psu = valid & (psu == psu[np.where(valid, best, 0)])
    per_r = pd.DataFrame(
        {
            "caseid": d["caseid"].astype(int),
            "iwr_pre": iwr,
            "psu": psu,
            "stratum": strat,
            "n_answered": X[cols].notna().sum(axis=1).to_numpy(),
            "max_pm": maxpm,
            "best_match_caseid": np.where(valid, d["caseid"].to_numpy()[np.where(valid, best, 0)], np.nan),
            "n_ties": ties,
            "best_same_iwr": same_iwr,
            "best_same_psu": same_psu,
        }
    )
    per_r.to_csv(f"{OUT}/respondents_{name}.csv", index=False)

    # Pairwise similarity by relationship (upper triangle, eligible pairs).
    iu = np.triu_indices(n, 1)
    s = S[iu]
    ok = ~np.isnan(s)
    si = iwr[iu[0]] == iwr[iu[1]]
    sp = psu[iu[0]] == psu[iu[1]]
    ss = strat[iu[0]] == strat[iu[1]]
    pair = {
        "same_iwr": float(np.mean(s[ok & si])),
        "same_psu_diff_iwr": float(np.mean(s[ok & sp & ~si])),
        "same_stratum_diff_psu": float(np.mean(s[ok & ss & ~sp])),
        "diff_stratum": float(np.mean(s[ok & ~ss])),
        "n_pairs": {
            "same_iwr": int((ok & si).sum()),
            "same_psu_diff_iwr": int((ok & sp & ~si).sum()),
            "same_stratum_diff_psu": int((ok & ss & ~sp).sum()),
            "diff_stratum": int((ok & ~ss).sum()),
        },
        "overall": float(np.mean(s[ok])),
    }

    SV = (np.nan_to_num(S).astype(np.float32), (~np.isnan(S)).astype(np.float32))
    ids, mm, s85, sb, wm = group_stats(SV, iwr, best, valid, maxpm)
    nper = pd.Series(iwr).value_counts().reindex(ids).to_numpy()
    tab = pd.DataFrame(
        {
            "iwr": ids,
            "n": nper,
            "mean_max_pm": mm,
            "share_ge_85": s85,
            "share_best_same_iwr": sb,
            "within_iwr_mean_sim": wm,
        }
    )
    P = 1000
    for blk_name, blk in [("stratum", strat), ("psu", psu)]:
        ge = {c: np.zeros(ids.size) for c in ["mean_max_pm", "share_best_same_iwr", "within_iwr_mean_sim"]}
        for _ in range(P):
            gp = permute_within(iwr, blk)
            _, mm_p, _, sb_p, wm_p = group_stats(SV, gp, best, valid, maxpm)
            ge["mean_max_pm"] += mm_p >= mm - 1e-12
            ge["share_best_same_iwr"] += sb_p >= sb - 1e-12
            ge["within_iwr_mean_sim"] += wm_p >= wm - 1e-12
        for c, v in ge.items():
            tab[f"p_{c}_{blk_name}"] = (1 + v) / (P + 1)
    tab = tab.sort_values("within_iwr_mean_sim", ascending=False)
    tab.to_csv(f"{OUT}/interviewers_{name}.csv", index=False)

    q = np.nanquantile(maxpm, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    big = tab[tab.n >= 5]
    results[name] = {
        "items": len(cols),
        "categories": int(ncat),
        "respondents_with_eligible_pair": int(valid.sum()),
        "max_pm_quantiles(0,1,5,25,50,75,95,99,100)": [round(float(x), 4) for x in q],
        "share_ge_85": float(np.nanmean(maxpm >= 0.85)),
        "share_ge_90": float(np.nanmean(maxpm >= 0.90)),
        "n_ge_85": int(np.nansum(maxpm >= 0.85)),
        "best_match_same_iwr": float(same_iwr[valid].mean()),
        "best_match_same_psu": float(same_psu[valid].mean()),
        "pairwise_mean_sim": pair,
        "n_iwr": int(ids.size),
        "n_iwr_ge5": int(len(big)),
        "flags_p001": {c: int((big[c] < 0.001).sum()) for c in big.columns if c.startswith("p_")},
        "flags_p01": {c: int((big[c] < 0.01).sum()) for c in big.columns if c.startswith("p_")},
    }
    print(f"\n######## {name}")
    print(json.dumps(results[name], indent=1))
    pd.set_option("display.width", 250)
    print("top 10 interviewers (n>=5) by within-interviewer mean similarity:")
    print(big.head(10).round(4).to_string(index=False))
    print("top 10 interviewers (n>=5) by share best match same interviewer:")
    print(big.sort_values("share_best_same_iwr", ascending=False).head(10).round(4).to_string(index=False))
    print("top 10 respondents by max_pm:")
    print(per_r.sort_values("max_pm", ascending=False).head(10).to_string(index=False))

with open(f"{OUT}/results.json", "w") as fh:
    json.dump(results, fh, indent=1)
