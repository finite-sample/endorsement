# Do Pakistanis Hold Militant Groups in "Low Regard"?

[![ci](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml)

Reanalysis of Blair, Fair, Malhotra, and Shapiro (2013), "Poverty and Support for Militant Politics: Evidence from Pakistan." The manuscript, the data, and every number in between are in this repository. Read the [paper on gsood.com](https://gsood.com/research/papers/endorsement.pdf); the repository copy is [ms/endorsement.pdf](ms/endorsement.pdf).

## Findings

Blair et al. read small negative endorsement effects as evidence that Pakistanis hold militant groups in low regard. Because each respondent answers four policy questions, the data can tell indifference from large reactions that offset. They show offsetting reactions.

<!-- BEGIN GENERATED RESULTS -->

In the unweighted analysis sample of 5,212 respondents, endorsements lower stated policy support by -0.013 on the 0–1 scale (respondent-clustered SE 0.0052). That small average is the net of reactions in both directions, not the absence of reactions.

- **Answers move together differently.** Under endorsement, the correlation between answers rises for five of six policy pairs (by 0.07 to 0.18) and falls for FCR and Durand (0.66 to 0.54).
- **The midpoint empties in both directions.** Respondents whose four answers average the midpoint fall from 10.8% to 7.0%, with gains one step below and one step above.
- **Opposition alone cannot do this.** Endorsed arms simulated from control respondents, tuned to the observed average effect, leave the midpoint share unchanged when reactions run only against the endorser.

How many respondents react favorably is not identified. The standard measurement model gives 43% (95% credible interval 32%–53%), but it treats a respondent's reactions as independent across questions, which the correlation changes contradict.

<!-- END GENERATED RESULTS -->

The midpoint comparison assumes that how a respondent reacts is unrelated to where their answers would otherwise sit. The analysis is unweighted and uses the `endorse` package sample, which differs from the original article's sample and assignment exclusions; it describes that sample, not national prevalence.

`exploration/reaction-models/` records attempts to estimate the share reacting favorably with Stan and mixed models, the assumptions each made, and why none produced a reportable number.

## Reproducing the Paper

Use R 4.6.0, the package versions in `renv.lock`, XeLaTeX with `latexmk` and BibTeX, and Poppler (`pdftotext`). Run commands from the repository root.

```sh
make restore
make paper
make check
```

`make paper` regenerates tables, manuscript numbers, and the results block above, then compiles `ms/endorsement.pdf`. `make check` runs R lint and tests, checks that generated text is current, builds the PDF, and checks references. `make format` applies styler. `make ci-docker` runs the same checks in a standard Rocker image.

The committed posterior summaries let you rebuild the paper without rerunning MCMC. To refit all three specifications, run `make fit`; to refit one, run `Rscript scripts/run.R fit primary` (or `covariates` or `hierarchical`). Each uses three dispersed chains with 200,000 iterations, 20,000 burn-in iterations, and thinning by 40. Optional iteration settings follow the specification as three integers. Full fits can take hours.

Fits write ignored draws to `data/endorse_fits_<spec>.rds` and committed summaries to `data/endorse_<spec>.csv`. `make summarize` rebuilds summaries from saved draws without fitting; then run `make paper` to refresh the paper. Diagnostics include Gelman–Rubin statistics and effective sample sizes.

## Data

`data/pakistan_endorsement.csv` is exported from the [endorse](https://CRAN.R-project.org/package=endorse) package's `pakistan` dataset. `make prepare` regenerates it. Both descriptive analysis and model fitting read this CSV through the same validation and factor-coding routine. Education remains categorical, as in the package data.

Endorser codes, confirmed against Blair et al.'s Table 1: a = control, b = Kashmiri tanzeems, c = Afghan Taliban, d = al-Qa'ida, e = sectarian tanzeems. `data/dataverse/` preserves the original replication materials, including third-party R and Stata code. These are reference inputs, not a second analysis pipeline.

## Repository Layout

```text
R/
  data.R       Input validation, factor coding, and response reshaping
  analysis.R   Effects, within-respondent patterns, simulated worlds, restrictions, and bounds
  model.R      Three-chain measurement-model fitting and posterior summaries
  paper.R      Tables, numerical macros, and README results
scripts/run.R  Single entry point: prepare, fit, summarize, analysis
tests/         testthat checks for data coding, estimands, bounds, and model summaries
data/          Committed inputs and posterior summaries; local draws are ignored
ms/            LaTeX source, bibliography, and compiled paper
tabs/          Generated tables and numerical macros
exploration/   Model attempts that did not produce a reportable estimate
DESCRIPTION    Direct R dependencies
renv.lock      Pinned R package versions
Makefile       Build order and checks
```

Calculations live in `R/analysis.R`; `R/paper.R` formats their results. There is one writer for each output.

## Citation

```
Sood, Gaurav. (2026). Do Pakistanis Hold Militant Groups in "Low Regard"?
https://github.com/finite-sample/endorsement
```

## License

Code and manuscript are MIT-licensed (see [LICENSE](LICENSE)). The files under `data/dataverse/` are Blair et al.'s replication materials, redistributed for reference under their original terms; `data/pakistan_endorsement.csv` is exported from the GPL-licensed `endorse` R package.
