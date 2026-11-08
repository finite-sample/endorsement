# Modelling individual reactions to endorsements: what we tried and why it failed

This folder records an attempt (November 2026) to go beyond the average endorsement effect and estimate
how individual reactions are distributed: how many respondents react favorably, how many unfavorably, and
how strongly. None of the models here produced a number we are willing to report. Two things did survive,
and both now live in the main R pipeline and the manuscript (section "Within-Respondent Patterns"):

- the model-free correlation and person-mean tables (`within_respondent_moments()`,
  `person_mean_distribution()` in `R/analysis.R`), and
- the simulated one-sided versus two-sided worlds (`reaction_worlds()`), which replaced
  `one_sided_check.py` from this folder.

Everything below explains why the model-based route was abandoned, so that nobody repeats it without a
new idea or new data.

## The question

The Pakistan endorsement experiment (Blair et al. 2013) asks four policy questions. Control respondents
see no endorser; endorsed respondents see a different militant group on each question. The average effect
of an endorsement is small (about -0.013 on a 0-1 scale). Two readings fit that number:

- **indifference**: almost nobody reacts; or
- **cancellation**: many react strongly, in both directions, and the reactions offset.

The published model (Bullock, Imai and Shapiro 2011, the `endorse` R package) reports that about 43% of
respondents react favorably. That figure comes from treating each reaction as an independent normal draw
around a group mean and reading off the share above zero.

## How the four questions carry information

Randomization tells us what the joint distribution of a person's four answers would look like without
endorsements: the control arm's. Two kinds of change are distinguishable:

- noise that is independent across questions raises each answer's variance but not the covariance between
  answers;
- a reaction the respondent carries to all four endorsements raises the covariances as well.

Every model here tries to turn that contrast into an estimate of the reaction distribution.

## What we tried

All Stan models use an ordinal probit: an answer is the category whose thresholds bracket a latent value
`y* = beta_j x_i + (reaction) + noise`, where `x_i` is the respondent's policy position and `beta_j` the
question's loading on it.

| Model (file, flag) | What it assumes about reactions | What happened | Why it fails |
|---|---|---|---|
| BIS (`fit_stan.py bis`) | Independent per question, scaled by the question's loading: `beta_j * s_ijk`, `s ~ N(lambda_k, omega_k^2)` | Reproduces the `endorse` estimates (lambda about -0.03, omega 0.14 to 0.48) | Posterior predictive check misses all six treated-minus-control covariance changes. The model rules out reactions that respondents carry across questions, and those are in the data. |
| Shared affect (`shared`) | Adds a person-level reaction `eta_i`, scaled by `beta_j` | Covariance changes improve but the pattern is wrong | Scaling by `beta_j` makes the most strongly linked pair (FCR and Durand) gain the most covariance. In the data that pair's correlation falls (0.66 to 0.54) while every other pair's rises. |
| Free treated scale (`hetero`, `hetero_inc`) | As above, plus a free scale on the treated latent | Fits better; income contrasts not supported | Same mechanism problem; the income differences in the point estimates disappear under the bootstrap. |
| Additive shift (`additive`) | `rho * beta_j * x_i + lambda + eta_i`: a reaction that is the same size on every question, plus a weakened link to policy position (`rho < 1`) | Best fit of the continuous models | Still a normal reaction distribution, so "share favorable" comes from the normal shape rather than the data. |
| Two factors (`twofactor`) | Free loadings for the reaction | Loadings poorly identified | Too many free parameters for what four questions can support. |
| Mixed model (`lme_check.R`, `lme_boot.R`) | Linear, one person effect, variances free by arm | Person-level variance rises by 0.0054 [0.0029, 0.0075]; residual variance falls | Agrees with the Stan models in direction. Ignores the ordinal scale and the ceiling, so it is a cross-check, not an estimate. |
| Latent classes (`endorse_mixture.stan`, `fit_mixture.py`) | Each respondent is an opponent, indifferent, or a supporter; opponents shift down by delta, supporters up by the same delta (symmetric) | 35% [25, 43] favorable in the dominant mode, but chains split between two explanations | See "Why the latent-class model failed" below. |
| + bump component (`flex_x=1`) | Lets the position distribution have a lump at the top of the scale | 21% [9, 34] favorable | The answer moves by 14 points depending on how the ceiling is modelled. |
| + free magnitudes (`symmetric=0`) | Supporters and opponents shift by different amounts | Every chain ran to a bound (delta_pos = 2, delta_neg about 0.27) | Not identified: a few large positive shifts and many small ones fit equally well. |
| + always-top class (`top_class=1`) | Respondents answering 5 on all four questions get their own class in both arms (16.4% control, 16.2% endorsed) | Chains disagree; one question's loading runs to 4 or 5 | See below. |

## Why the latent-class model failed

The latent-class model is the natural way to ask "how many favorable, how many unfavorable". It failed in
three ways, and each fix exposed the next.

1. **Two competing explanations (multimodality).** About 16% of respondents answer at the top of the
   scale on every question. The model can explain that cluster either as people with very high policy
   positions or as "supporters" with a large upward shift. Chains started in different places settle on
   different explanations. Running longer does not help, because the chains are each in a stable but
   different solution. Hand-picked starting values (`inits_goodmode.json`) only choose a solution; they
   don't justify it.

2. **Removing the ceiling cluster created a new degenerate solution.** Giving the always-top respondents
   their own class removed the first problem. With them gone, the remaining control-arm correlations are
   weak, and the model lets one question's loading grow to 4 or 5, so that one question carries the whole
   factor. Factor analysts call this a Heywood case. Different chains chose different questions, and the
   sampler's energy diagnostic (E-BFMI 0.01 against a threshold of 0.3) shows the chains were not
   exploring. Favorable share among respondents who can move: 15-18% in three chains, 30-45% in one.

3. **The model reports a supporter share even when the data contain none.** We simulated data with known
   shares (0%, 20%, 35% favorable) and refit. Those simulated worlds were built with too little added
   spread, so they were a poor match to the real data. But they still showed something: with little
   signal, the model returned 14-20% favorable whatever the truth, which is the prior speaking. An earlier
   calibration without the top class gave 13% for a world with no supporters and 31% for a world with 35%.

Behind all three is one fact. Within each arm, what the data pin down is roughly the mean and the spread
of the reaction distribution. A world in which 4% of respondents react strongly against the endorser
reproduces the observed mean and spread about as well as a world with 30% on each side. The class shares
are set by the assumptions (normal positions, symmetric magnitudes, how the ceiling is treated) and the
prior, not by the data.

## What the data do support, and what that rests on

The test that discriminates is direction-specific. Starting from the real control respondents and adding
reactions tuned to the observed average effect, opposition alone leaves the share of respondents whose
answers average the midpoint unchanged (people leaving it downward are replaced from above). Only worlds
with a favorable group empty it. The observed drop is 3.8 points [2.3, 5.4]. This is Table 5 in the
manuscript.

That argument assumes reactions are unrelated to where a respondent's answers would otherwise sit. Without
that assumption, the sharp lower bound on the share whose mean answer rose is 0.2%. A reaction aimed only at
respondents just above the midpoint could refill it. Another mechanism that would also empty the midpoint:
the endorsement pushes respondents toward their own policy views, without anyone reacting to the group.

## Files

| File | Role |
|---|---|
| `export.R` | Writes the long-format data the Python scripts read. Run from the repository root. |
| `endorse_shared_v.stan`, `fit_stan.py` | The continuous models: BIS, shared, hetero, additive, twofactor, and the `_inc` versions by income group |
| `summarize.py` | Favorable share (extensive margin) and mean reaction among the favorable (intensive margin) from saved draws |
| `ppc.py` | Posterior predictive checks: simulates replicated surveys and compares variance and covariance changes |
| `simulate.py` | Parameter recovery for the shared-affect model |
| `endorse_mixture.stan`, `fit_mixture.py` | The latent-class model and its variants |
| `simulate_mixture.py` | Simulated worlds with known class shares, for calibration |
| `inits_goodmode.json` | The hand-picked starting values from the multimodality episode |
| `lme_check.R`, `lme_boot.R` | Linear mixed-model cross-check and its respondent bootstrap (`nlme`) |
| `income/` | The original question: does the extensive or intensive margin differ by income. Point estimates suggested fewer favorable reactions among the poor, but the bootstrap does not support it. |
| `results/` | Console logs of each fit (summaries and diagnostics) and the bootstrap output. |

## Reproducing

From the repository root, with CmdStan 2.40 (a native arm64 build on Apple silicon; an x86_64 build will
not link) and `cmdstanpy`:

```sh
Rscript exploration/reaction-models/export.R exploration/reaction-models/runs/long.csv
python3 exploration/reaction-models/fit_stan.py exploration/reaction-models/runs bis 500 500 \
  exploration/reaction-models/runs/fit_bis
python3 exploration/reaction-models/fit_mixture.py exploration/reaction-models/runs 500 500 \
  exploration/reaction-models/runs/fit_top 0.9 1 '' 0 1
```

Each fit takes 10-60 minutes and writes hundreds of megabytes of draws. `runs/` is ignored by git.
