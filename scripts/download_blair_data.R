#!/usr/bin/env Rscript

if (!require("endorse", quietly = TRUE)) {
  install.packages("endorse", repos = "https://cloud.r-project.org")
}

library(endorse)
data(pakistan)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

root_dir <- dirname(script_dir)
data_dir <- file.path(root_dir, "data")

if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

output_path <- file.path(data_dir, "pakistan_endorsement.csv")
write.csv(pakistan, output_path, row.names = FALSE)
cat("Wrote", nrow(pakistan), "observations to", output_path, "\n")
