#!/usr/bin/env Rscript
# Fit the Bullock-Imai-Shapiro measurement model (endorse package) to the
# Blair et al. (2013) Pakistan data and summarize, per militant group, the
# population share of respondents whose reaction to the endorsement is
# positive, P(s > 0). Reactions are s ~ N(mu_i[k], omega_k^2) where mu_i[k]
# is lambda_k (primary), Z_i' lambda_k (covariates), or the division
# intercept lambda0_[k, div(i)] (hierarchical). P(s > 0) is invariant to the
# ideal-point scale; mean reactions are also reported in ideal-point SDs.
#
# Protocol follows Bullock, Imai, and Shapiro (2011): three parallel chains
# with overdispersed starting values, convergence monitored with
# Gelman-Rubin, effective sample sizes reported.
#
# Usage: Rscript scripts/fit_endorse.R <spec> [MCMC burn thin]
#   spec: primary | covariates | hierarchical
# Writes data/endorse_<spec>.csv (summaries, committed) and
# data/endorse_fits_<spec>.rds (draws, gitignored).

library(endorse)

args <- commandArgs(trailingOnly = TRUE)
spec <- if (length(args) >= 1) args[1] else "primary"
n_mcmc <- if (length(args) >= 2) as.integer(args[2]) else 200000
n_burn <- if (length(args) >= 3) as.integer(args[3]) else 20000
n_thin <- if (length(args) >= 4) as.integer(args[4]) else 40
stopifnot(spec %in% c("primary", "covariates", "hierarchical"))

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
root_dir <- dirname(dirname(normalizePath(script_path)))
out_csv <- file.path(root_dir, "data", sprintf("endorse_%s.csv", spec))
out_rds <- file.path(root_dir, "data", sprintf("endorse_fits_%s.rds", spec))

data(pakistan)

Y <- list(
  Q1 = c("Polio.a", "Polio.b", "Polio.c", "Polio.d", "Polio.e"),
  Q2 = c("FCR.a", "FCR.b", "FCR.c", "FCR.d", "FCR.e"),
  Q3 = c("Durand.a", "Durand.b", "Durand.c", "Durand.d", "Durand.e"),
  Q4 = c("Curriculum.a", "Curriculum.b", "Curriculum.c", "Curriculum.d",
         "Curriculum.e")
)

# Endorser order within each question (Blair et al. 2013, Table 1):
groups <- c("Kashmiri tanzeems", "Afghan Taliban", "Al-Qaeda",
            "Sectarian tanzeems")
K <- length(groups)

starts <- list(
  list(seed = 101, lambda = -0.5, omega2 = 0.05),
  list(seed = 102, lambda = 0, omega2 = 0.1),
  list(seed = 103, lambda = 0.5, omega2 = 0.5)
)

run_chain <- function(st) {
  set.seed(st$seed)
  common <- list(Y = Y, data = pakistan, identical.lambda = TRUE,
                 MCMC = n_mcmc, burn = n_burn, thin = n_thin,
                 lambda.start = st$lambda, omega2.start = st$omega2,
                 verbose = FALSE)
  extra <- switch(spec,
    primary = list(covariates = FALSE, hierarchical = FALSE),
    covariates = list(covariates = TRUE,
                      formula.indiv = ~ female + rural + edu + inc,
                      hierarchical = FALSE),
    hierarchical = list(
      covariates = FALSE, hierarchical = TRUE,
      data.village = data.frame(division = sort(unique(pakistan$division))),
      village = "division", formula.village = ~ 1
    )
  )
  do.call(endorse, c(common, extra))
}

cat(sprintf("spec = %s, MCMC = %d, burn = %d, thin = %d\n",
            spec, n_mcmc, n_burn, n_thin))
fits <- lapply(seq_along(starts), function(i) {
  cat(sprintf("chain %d...\n", i))
  run_chain(starts[[i]])
})
saveRDS(fits, out_rds)

lam_list <- coda::mcmc.list(lapply(fits, function(f) f$lambda))
om2_list <- coda::mcmc.list(lapply(fits, function(f) f$omega2))

gr <- function(x) coda::gelman.diag(x, multivariate = FALSE)$psrf[, 1]
rhat_lambda <- gr(lam_list)
rhat_omega2 <- gr(om2_list)
ess_lambda <- coda::effectiveSize(lam_list)
cat(sprintf("\nmax R-hat: lambda %.3f, omega2 %.3f; min ESS(lambda): %.0f\n",
            max(rhat_lambda), max(rhat_omega2), min(ess_lambda)))

lambda <- do.call(rbind, lapply(fits, function(f) as.matrix(f$lambda)))
omega2 <- do.call(rbind, lapply(fits, function(f) as.matrix(f$omega2)))
x_sd <- unlist(lapply(fits, function(f) as.numeric(f$x)))
n_draws <- nrow(lambda)

# Per-draw population mean reaction mu_k (latent scale) and P(s > 0).
if (spec == "primary") {
  mu <- lambda
  p_pos <- pnorm(lambda / sqrt(omega2))
} else if (spec == "covariates") {
  M <- fits[[1]]$model.matrix.indiv
  p_pos <- matrix(NA_real_, n_draws, K)
  mu <- matrix(NA_real_, n_draws, K)
  block <- 500
  for (k in 1:K) {
    cols <- ((k - 1) * ncol(M) + 1):(k * ncol(M))
    for (b in seq(1, n_draws, by = block)) {
      idx <- b:min(b + block - 1, n_draws)
      zl <- M %*% t(lambda[idx, cols, drop = FALSE])  # respondents x block
      mu[idx, k] <- colMeans(zl)
      p_pos[idx, k] <- colMeans(
        pnorm(sweep(zl, 2, sqrt(omega2[idx, k]), "/"))
      )
    }
  }
} else {
  div <- fits[[1]]$village.indicator
  w <- as.numeric(table(factor(div, levels = seq_len(max(div))))) / length(div)
  n_div <- length(w)
  p_pos <- matrix(NA_real_, n_draws, K)
  mu <- matrix(NA_real_, n_draws, K)
  for (k in 1:K) {
    cols <- ((k - 1) * n_div + 1):(k * n_div)
    l0 <- lambda[, cols]  # draws x divisions
    mu[, k] <- as.numeric(l0 %*% w)
    p_pos[, k] <- as.numeric(pnorm(l0 / sqrt(omega2[, k])) %*% w)
  }
}

mu_std <- mu / x_sd
omega_std <- sqrt(omega2) / x_sd

qsum <- function(d) c(mean(d), quantile(d, 0.025), quantile(d, 0.975))
rows <- lapply(1:K, function(k) {
  data.frame(
    spec = spec, group = groups[k],
    t(setNames(qsum(p_pos[, k]), c("p_mean", "p_lo", "p_hi"))),
    t(setNames(qsum(mu_std[, k]), c("mu_mean", "mu_lo", "mu_hi"))),
    omega_mean = mean(omega_std[, k]),
    p_mu_neg = mean(mu[, k] < 0),
    rhat_max_lambda = max(rhat_lambda), rhat_max_omega2 = max(rhat_omega2),
    ess_min_lambda = min(ess_lambda)
  )
})
avg <- data.frame(
  spec = spec, group = "Average",
  t(setNames(qsum(rowMeans(p_pos)), c("p_mean", "p_lo", "p_hi"))),
  t(setNames(qsum(rowMeans(mu_std)), c("mu_mean", "mu_lo", "mu_hi"))),
  omega_mean = mean(omega_std),
  p_mu_neg = mean(rowMeans(mu) < 0),
  rhat_max_lambda = max(rhat_lambda), rhat_max_omega2 = max(rhat_omega2),
  ess_min_lambda = min(ess_lambda)
)
out <- do.call(rbind, c(rows, list(avg)))
out[, 3:11] <- round(out[, 3:11], 4)
write.csv(out, out_csv, row.names = FALSE)

cat("\nP(s > 0) by group:\n")
print(out[, c("group", "p_mean", "p_lo", "p_hi", "mu_mean", "omega_mean")],
      row.names = FALSE)
cat("\nWrote", out_csv, "and", out_rds, "\n")
