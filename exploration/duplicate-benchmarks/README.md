# Near-Duplicate Benchmarks

The manuscript compares the Pakistan survey's within-village near-duplicates (Section 5) with two surveys that record interviewer identifiers: the 2012 American National Election Study face-to-face interviews and Pew's Spring 2009 Global Attitudes survey of Pakistan. Neither survey's data are redistributed here; download `anes_timeseries_2012.dta` from electionstudies.org and `Pew Gap 2009 Data.sav` from pewresearch.org.

## What the scripts do

1. `survey_vars.py ANES.dta out.parquet` extracts the face-to-face cases (`mode == 1`, n = 2,054).
2. `percent_match.py` selects closed-ended items asked of at least half the face-to-face respondents, keeping don't know (-8) and refusal (-9) as answers and treating inapplicable and other negative codes as missing. It leaves 601 items, 514 of them interviewer-administered.
3. `pk_cats.py PAKISTAN.dta` reads the Pakistan item profile: 92 questions, of which 46 have 2 answer codes, 7 have 4, 7 have 5, 29 have 7, 2 have 10, and 1 has 12.
4. `benchmark_subsets.py` draws 200 sets of 92 interviewer-administered ANES items with the same mix of answer-code counts (binary items with replacement, since ANES has only 25) and computes each respondent's best percent match, over the whole sample and within primary sampling units. A pair counts only when both answered at least half the items.

## Results

Saved in `results/benchmark_summary.json` and `results/benchmark_subsets.csv`. Median over the 200 sets, with the 95th percentile in parentheses:

| | ANES, within PSU | Pakistan, within sampling point |
|---|---|---|
| Median best match | 0.70 (0.73) | 0.79 |
| Share with a match on at least 85% of items | 0.1% (1.3%) | 31% |
| Share with a match on at least 90% of items | 0.0% (0.1%) | 17% |
| Share with an exact twin | 0 (0) | 1.8% |

The ANES sets agree by chance slightly more often than the Pakistan items (mean per-item chance agreement 0.50 against 0.45), so the comparison does not favor Pakistan. ANES primary sampling units average about 16 face-to-face respondents, against 12 per Pakistan sampling point.

## Pew 2009 Pakistan

`pew_pakistan.py PEW.sav Q106` keeps the 1,254 Pakistani respondents and the 142 closed-ended questions (`Q`-prefixed, at most 15 answer codes) asked of at least half of them, and compares each respondent with every other Pakistani respondent. The file has no sampling-point identifier, so the comparison pool is the national sample. Median best match 0.69; 3.2% of respondents match another on at least 85% of questions, 0.3% on at least 90%; no exact twins. Running it on the 2010 file (`Q141`) gives median 0.71, 6.8% at 85% or more, and no exact twins.
