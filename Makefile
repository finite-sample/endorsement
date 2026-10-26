RSCRIPT = Rscript

all: paper

restore:
	$(RSCRIPT) -e 'renv::restore(prompt = FALSE)'

prepare:
	$(RSCRIPT) scripts/run.R prepare

fit:
	$(RSCRIPT) scripts/run.R fit

summarize:
	$(RSCRIPT) scripts/run.R summarize

analysis:
	$(RSCRIPT) scripts/run.R analysis

manuscript:
	cd ms && latexmk -xelatex -interaction=nonstopmode -halt-on-error endorsement.tex

paper: analysis
	$(MAKE) manuscript

format:
	$(RSCRIPT) -e 'styler::style_dir("R"); styler::style_dir("scripts"); styler::style_dir("tests")'

lint:
	$(RSCRIPT) -e 'issues <- unlist(lapply(c("R", "scripts", "tests"), lintr::lint_dir), recursive = FALSE); print(issues); if (length(issues)) quit(status = 1)'

test:
	$(RSCRIPT) tests/testthat.R

check: lint test
	@set -eu; check_tmp=$$(mktemp -d); \
	trap 'rm -rf "$$check_tmp"' EXIT; \
	cp -R tabs "$$check_tmp/tabs"; cp README.md "$$check_tmp/README.md"; \
	$(MAKE) analysis; \
	diff -ru "$$check_tmp/tabs" tabs; \
	diff -u "$$check_tmp/README.md" README.md
	$(MAKE) manuscript
	$(MAKE) check-pdf

check-pdf:
	@test -s ms/endorsement.pdf && test -s ms/endorsement.log
	@! grep -Eq 'Warning.*undefined|Overfull' ms/endorsement.log
	@set -eu; check_tmp=$$(mktemp); trap 'rm -f "$$check_tmp"' EXIT; \
	pdftotext ms/endorsement.pdf "$$check_tmp"; ! grep -Fq '??' "$$check_tmp"

ci-docker:
	docker run --rm -v "$(CURDIR):/project" -w /project \
	  -e RENV_PATHS_LIBRARY=/tmp/renv-library rocker/verse:4.6.0 \
	  bash -c 'apt-get update && apt-get install -y poppler-utils && make restore check'

.PHONY: all restore prepare fit summarize analysis manuscript paper format lint test check check-pdf ci-docker
