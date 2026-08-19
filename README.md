# Do Pakistanis Hold Militant Groups in Low Regard?

[![ci](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/endorsement/actions/workflows/ci.yml)

Reanalysis of Blair, Fair, Malhotra, and Shapiro (2013), "Poverty and Support for Militant Politics: Evidence from Pakistan." The manuscript, the data, and every number in between are in this repository. Read the paper at [ms/endorsement.pdf](ms/endorsement.pdf).

## The Finding

Blair et al. concluded that "Pakistanis in general are weakly negative toward Islamist militant organizations" and that the negative, small, mostly significant treatment effects suggest "Pakistanis hold militant groups in low regard." Refitting their data with the measurement model the endorsement-experiment literature itself developed (Bullock, Imai, and Shapiro 2011), roughly two in five Pakistanis react positively to a militant group's endorsement:

| Group | Share reacting positively | 95% credible interval |
|-------|--------------------------|-----------------------|
| Kashmiri tanzeems | 45% | [25%, 58%] |
| Afghan Taliban | 32% | [2%, 49%] |
| Al-Qaeda | 47% | [40%, 54%] |
| Sectarian tanzeems | 49% | [38%, 57%] |
| Average | 43% | [32%, 53%] |

The mean reaction is negative but tiny, at most 0.08 ideal-point standard deviations, while the dispersion of reactions is several times larger. With heterogeneity that wide around a mean that small, "weakly negative on average" and "two in five react positively" are the same fact. The sign of the ATE measures the direction of the average, not the level of support.

The paper also shows the general point: the ATE decomposes as beta = p * delta_plus + (1 - p) * delta_minus, three unknowns in one equation, so explicit behavioral models map the same ATE (-1.3 points) and baseline (74%) to supporter shares from 18% to 99%, and assumption-free coupling bounds place the share of positive reactions anywhere between 1% and 58%. Blair et al.'s own subgroup estimates corroborate the heterogeneity reading: their middle-class treatment effect is between -0.6 and 0.1 percentage points, indistinguishable from zero.

## Reproducing the Paper

Requirements: Python 3.10+ with the packages in `requirements.txt`, R with the `endorse` package (installed automatically by the download script), and a LaTeX distribution with `pdflatex` and `bibtex`.

```bash
make tables     # regenerate tabs/ and figs/ from data (seconds, deterministic)
make pdf        # compile ms/endorsement.pdf
make check      # regenerated tables must match the committed ones; pdf must compile clean
make endorse    # refit the measurement model: 3 chains x 3 specifications, 1-2 hours
make lint       # black, isort, flake8
```

The expensive MCMC step writes posterior summaries to `data/endorse_*.csv`, which are committed, so `make tables` and `make check` never refit the model. `make endorse` reruns the fits from scratch (three overdispersed chains per specification, following Bullock et al.'s protocol) and prints Gelman-Rubin statistics and effective sample sizes.

## Data

`data/pakistan_endorsement.csv` is the `pakistan` dataset shipped with the [endorse](https://CRAN.R-project.org/package=endorse) R package (n = 5,212; regenerate it with `Rscript scripts/download_blair_data.R`). Endorser codes, confirmed against Blair et al.'s Table 1: a = control, b = Kashmiri tanzeems, c = Afghan Taliban, d = al-Qa'ida, e = sectarian tanzeems. `data/dataverse/` holds Blair et al.'s original replication files for reference, including their paper and Stata code.

## Repository Layout

```
├── ms/endorsement.tex                      # Manuscript
├── scripts/fit_endorse.R                   # Measurement-model fits (writes data/endorse_*.csv)
├── scripts/generate_tables_and_figures.py  # Single writer of tabs/ and figs/
├── scripts/analyze_blair_data.py           # Console diagnostics, writes nothing
├── scripts/download_blair_data.R           # Regenerates data/pakistan_endorsement.csv
├── data/                                   # Microdata, posterior summaries, dataverse files
├── tabs/, figs/                            # Generated exhibits
└── Makefile                                # tables, pdf, check, endorse, lint
```

## Citation

```
Sood, Gaurav. (2026). Do Pakistanis Hold Militant Groups in Low Regard?
https://github.com/finite-sample/endorsement
```

## License

Code and manuscript are MIT-licensed (see [LICENSE](LICENSE)). The files under `data/dataverse/` are Blair et al.'s replication materials, redistributed for reference under their original terms; `data/pakistan_endorsement.csv` is exported from the GPL-licensed `endorse` R package.
