# Cluster (respondent) bootstrap of the nlme decomposition in lme_check.R:
# percentile intervals for the variance excesses and the share favorable, pooled and by income.
# usage: Rscript exploration/reaction-models/lme_boot.R <B> <out csv>
suppressPackageStartupMessages(library(nlme))
for (m in c("data", "analysis")) source(file.path("R", paste0(m, ".R")))
args <- commandArgs(trailingOnly = TRUE)
B <- as.integer(args[1]); out_path <- args[2]
d <- read_responses()
r <- reshape_responses(d)
r$inc <- d$inc[r$respondent]
r$trow <- r$endorser != "a"
arm <- tapply(r$trow, r$respondent, mean)
r <- r[arm[as.character(r$respondent)] %in% c(0, 1), ]
r$dT <- as.numeric(arm[as.character(r$respondent)] == 1)
r$dC <- 1 - r$dT
r$endorser <- factor(r$endorser, levels = c("a", names(endorsers)), labels = c("control", endorsers))

decompose <- function(s) {
  s$respondent <- factor(s$respondent)
  fit <- lme(response ~ policy + endorser, data = s,
    random = list(respondent = pdDiag(~ 0 + dC + dT)),
    weights = varIdent(form = ~ 1 | trow), method = "REML", control = lmeControl(opt = "optim"))
  vc <- as.numeric(VarCorr(fit)[c("dC", "dT"), "Variance"])
  ratio <- coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  res <- fit$sigma^2 * ratio[c("FALSE", "TRUE")]^2
  lam <- fixef(fit)[grep("endorser", names(fixef(fit)))]
  omega_eta2 <- vc[2] - vc[1]
  w <- sqrt(max(omega_eta2, 1e-8))
  c(omega_eta2 = omega_eta2, omega_u2 = unname(res[2] - res[1]), lambda = unname(lam), p = unname(pnorm(lam / w)))
}
resample <- function(s) {
  ids <- unique(s$respondent)
  draw <- sample(ids, replace = TRUE)
  pieces <- lapply(seq_along(draw), function(i) {
    p <- s[s$respondent == draw[i], ]; p$respondent <- i; p
  })
  do.call(rbind, pieces)
}
groups <- list(all = 1:3, low = 1, middle = 2, high = 3)
rows <- list()
for (g in names(groups)) {
  s <- r[r$inc %in% groups[[g]], ]
  est <- decompose(s)
  set.seed(1)
  boot <- t(replicate(B, tryCatch(decompose(resample(s)), error = function(e) rep(NA_real_, length(est)))))
  rows[[g]] <- data.frame(income = g, quantity = names(est), estimate = est,
    lo = apply(boot, 2, quantile, 0.025, na.rm = TRUE), hi = apply(boot, 2, quantile, 0.975, na.rm = TRUE),
    failed = sum(is.na(boot[, 1])))
  message(g, " done")
}
out <- do.call(rbind, rows)
out[, 3:5] <- round(out[, 3:5], 4)
write.csv(out, out_path, row.names = FALSE)
print(out, row.names = FALSE)
