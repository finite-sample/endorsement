#!/usr/bin/env Rscript
arguments <- commandArgs(trailingOnly = TRUE)
for (module in c("data", "analysis", "model", "format", "paper")) source(file.path("R", paste0(module, ".R")))
if (!length(arguments) || !arguments[1] %in% c("prepare", "fit", "summarize", "analysis")) {
  stop("Usage: Rscript scripts/run.R prepare | analysis | fit <spec> [iterations burn thin] | summarize [spec]")
}
action <- arguments[1]
if (action == "prepare") {
  prepare_data()
} else if (action == "analysis") {
  write_paper(analyze_responses(read_responses(), read_posteriors()))
} else {
  specs <- if (length(arguments) >= 2L) arguments[2] else names(specifications)
  stopifnot(all(specs %in% names(specifications)))
  settings <- c(200000L, 20000L, 40L)
  if (length(arguments) > 2L) {
    stopifnot(action == "fit", length(arguments) == 5L)
    settings <- as.integer(arguments[3:5])
  }
  for (spec in specs) {
    path <- file.path("data", paste0("endorse_fits_", spec, ".rds"))
    if (action == "fit") {
      fits <- fit_model(read_responses(), spec, settings[1], settings[2], settings[3])
      saveRDS(fits, path)
    } else {
      fits <- readRDS(path)
    }
    posterior <- summarize_fits(fits, spec)
    write.csv(posterior, file.path("data", paste0("endorse_", spec, ".csv")), row.names = FALSE)
    print(posterior, row.names = FALSE)
  }
}
