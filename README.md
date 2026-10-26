# Do Pakistanis Hold Militant Groups in Low Regard?

[![ci](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml)

Reanalysis of Blair, Fair, Malhotra, and Shapiro (2013), "Poverty and Support for Militant Politics: Evidence from Pakistan." The manuscript, the data, and every number in between are in this repository. Read the [paper on gsood.com](https://gsood.com/research/papers/endorsement.pdf); the repository copy is [ms/endorsement.pdf](ms/endorsement.pdf).

## Findings

A negative average endorsement effect does not identify how many people view the endorser favorably. This reanalysis distinguishes the average effect on stated policy support, the probability of a positive latent reaction under a measurement model, and bounds on increases in observed responses.

<!-- BEGIN GENERATED RESULTS -->

In the unweighted analysis sample of 5,212 respondents, the fitted measurement model implies a **43% probability of a positive latent reaction** to an endorsement, averaged equally across four militant groups (95% credible interval 32%–53%).

| Endorser | Positive latent reaction | 95% credible interval |
|---|---:|---:|
| Kashmiri tanzeems | 45% | [25%, 58%] |
| Afghan Taliban | 32% | [2%, 49%] |
| Al-Qaeda | 47% | [40%, 54%] |
| Sectarian tanzeems | 49% | [38%, 57%] |
| Average | 43% | [32%, 53%] |

The pooled outcome mean is 0.736 without an endorsement and 0.723 with one, an average difference of -0.013 on the rescaled 0–1 response scale (respondent-clustered SE 0.0052). Bounds calculated from the pooled response marginals run from 1% to 58% for the fraction of person–policy responses strictly increased by endorsement. These are plug-in bounds, not confidence intervals or bounds on the latent reaction probability.

<!-- END GENERATED RESULTS -->

The latent probabilities depend on the ordinal measurement model and its normal reaction distribution. They describe the fitted unweighted sample from the 2009 survey, not a nationally representative estimate of supporter prevalence or the share favorable toward at least one group. Positive latent reactions can leave observed answer categories unchanged.

Illustrative moment restrictions produce widely different fractions, but those calculations are not a set of equally well-fitting models. Binary switching is contradicted by interior responses on the five-point scale. The extreme-opponent calculation describes an unaffected fraction, while the normal-effect calculation is a continuous approximation. Figure captions distinguish these quantities, and sensitivity curves omit infeasible parameter ranges.

The package sample also differs from the original article's sample and assignment exclusions. The manuscript reports an exclusion sensitivity check for the pooled ATE. Population adjustment, Bayesian refits under those exclusions, prior sensitivity, and posterior predictive checks remain necessary before drawing stronger conclusions about national attitudes.

## Reproducing the Paper

Use R 4.6.0, the package versions in `renv.lock`, XeLaTeX with `latexmk` and BibTeX, and Poppler (`pdftotext`). Run commands from the repository root.

```sh
make restore
make paper
make check
```

`make paper` regenerates tables, the sensitivity figure, manuscript numbers, and the results block above, then compiles `ms/endorsement.pdf`. `make check` runs R lint and tests, checks that generated text is current, builds the PDF, and checks references. `make format` applies styler. `make ci-docker` runs the same checks in a standard Rocker image.

The committed posterior summaries let you rebuild the paper without rerunning MCMC. To refit all three specifications, run `make fit`; to refit one, run `Rscript scripts/run.R fit primary` (or `covariates` or `hierarchical`). Each uses three dispersed chains with 200,000 iterations, 20,000 burn-in iterations, and thinning by 40. Optional iteration settings follow the specification as three integers. Full fits can take hours.

Fits write ignored draws to `data/endorse_fits_<spec>.rds` and committed summaries to `data/endorse_<spec>.csv`. `make summarize` rebuilds summaries from saved draws without fitting; then run `make paper` to refresh the paper. Diagnostics include Gelman–Rubin statistics and effective sample sizes.

## Data

`data/pakistan_endorsement.csv` is exported from the [endorse](https://CRAN.R-project.org/package=endorse) package's `pakistan` dataset. `make prepare` regenerates it. Both descriptive analysis and model fitting read this CSV through the same validation and factor-coding routine. Education remains categorical, as in the package data.

Endorser codes, confirmed against Blair et al.'s Table 1: a = control, b = Kashmiri tanzeems, c = Afghan Taliban, d = al-Qa'ida, e = sectarian tanzeems. `data/dataverse/` preserves the original replication materials, including third-party R and Stata code. These are reference inputs, not a second analysis pipeline.

## Repository Layout

```text
R/
  data.R       Input validation, factor coding, and response reshaping
  analysis.R   Effects, uncertainty, moment restrictions, and response bounds
  model.R      Three-chain measurement-model fitting and posterior summaries
  paper.R      Tables, figure, numerical macros, and README results
scripts/run.R  Single entry point: prepare, fit, summarize, analysis
tests/         testthat checks for data coding, estimands, bounds, and model summaries
data/          Committed inputs and posterior summaries; local draws are ignored
ms/            LaTeX source, bibliography, and compiled paper
tabs/, figs/   Generated exhibits and numerical macros
DESCRIPTION    Direct R dependencies
renv.lock      Pinned R package versions
Makefile       Build order and checks
```

Calculations live in `R/analysis.R`; `R/paper.R` formats their results. There is one writer for each output. The method-comparison table replaces the redundant bar chart; the remaining figure shows how the formulas change with the ATE.

## Citation

```
Sood, Gaurav. (2026). Do Pakistanis Hold Militant Groups in Low Regard?
https://github.com/finite-sample/endorsement
```

## License

Code and manuscript are MIT-licensed (see [LICENSE](LICENSE)). The files under `data/dataverse/` are Blair et al.'s replication materials, redistributed for reference under their original terms; `data/pakistan_endorsement.csv` is exported from the GPL-licensed `endorse` R package.
