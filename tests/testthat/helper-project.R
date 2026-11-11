root <- normalizePath(file.path("..", ".."))
for (module in c("data", "analysis", "blair", "model", "format", "paper")) {
  source(file.path(root, "R", paste0(module, ".R")))
}
respondents <- read_responses(file.path(root, "data", "pakistan_endorsement.csv"))
