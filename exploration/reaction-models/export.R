for (m in c("data", "analysis")) source(file.path("R", paste0(m, ".R")))
d <- read_responses()
r <- reshape_responses(d)
r$y <- as.integer(round(r$response * 4)) + 1L
r$item <- match(r$policy, policies)
r$group <- match(r$endorser, c("a", names(endorsers))) - 1L
r$inc <- d$inc[r$respondent]
write.csv(r[, c("respondent", "item", "group", "y", "inc")], commandArgs(TRUE)[1], row.names = FALSE)
