for (m in c("data", "analysis")) source(file.path("R", paste0(m, ".R")))
d <- read_responses()
r <- reshape_responses(d)
r$inc <- d$inc[r$respondent]
rows <- list()
for (inc in 1:3) {
  s <- r[r$inc == inc, ]
  control <- s$response[s$endorser == "a"]
  cd <- category_distribution(control)
  for (code in c(names(endorsers), "all")) {
    tr <- if (code == "all") s$response[s$endorser != "a"] else s$response[s$endorser == code]
    sub <- if (code == "all") s else s[s$endorser %in% c("a", code), ]
    se <- clustered_ate_se(sub)
    b <- pairing_bounds(category_distribution(tr), cd)
    rows[[length(rows) + 1]] <- data.frame(
      inc, endorser = if (code == "all") "All" else endorsers[[code]],
      n_resp = length(unique(s$respondent)), ate = mean(tr) - mean(control), se,
      lower = b[["lower"]], upper = b[["upper"]]
    )
  }
}
out <- do.call(rbind, rows)
out[, 4:7] <- round(out[, 4:7], 3)
print(out, row.names = FALSE)
