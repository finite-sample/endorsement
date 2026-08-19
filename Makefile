PY = python3

all: tables pdf

tables:
	$(PY) scripts/generate_tables_and_figures.py

# Expensive MCMC step: refits the measurement model (three chains per
# specification) and rewrites data/endorse_*.csv, the inputs to `tables`.
endorse:
	Rscript scripts/fit_endorse.R primary
	Rscript scripts/fit_endorse.R covariates
	Rscript scripts/fit_endorse.R hierarchical
	$(PY) scripts/generate_tables_and_figures.py

pdf:
	cd ms && pdflatex -interaction=nonstopmode endorsement.tex >/dev/null && \
	bibtex endorsement >/dev/null && \
	pdflatex -interaction=nonstopmode endorsement.tex >/dev/null && \
	pdflatex -interaction=nonstopmode endorsement.tex >/dev/null

# Regenerating must not change any committed table, and the manuscript must
# compile without unresolved references. figs/ are deliberately excluded from
# the diff gate: matplotlib embeds timestamps, so the PDFs are byte-unstable
# across runs even when the generating code is unchanged.
check: tables
	git diff --exit-code tabs
	$(MAKE) pdf
	@! grep -q "LaTeX Warning.*undefined" ms/endorsement.log || \
	  (echo "undefined references in ms/endorsement.log" && exit 1)
	@echo "check passed"

lint:
	black --check scripts
	isort --check-only scripts
	flake8 scripts

.PHONY: all tables endorse pdf check lint
