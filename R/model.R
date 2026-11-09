model_sample <- function(respondents, spec) {
  stopifnot(spec %in% names(specifications))
  if (spec == "clean") respondents[!respondents$interviewer %in% suspect_interviewers, ] else respondents
}

fit_model <- function(respondents, spec, iterations = 200000L, burn = 20000L, thin = 40L, cores = 3L) {
  stopifnot(
    spec %in% names(specifications), iterations > burn, burn >= 0, thin >= 1,
    all(c(iterations, burn, thin) %% 1 == 0), (iterations - burn) %% thin == 0
  )
  respondents <- model_sample(respondents, spec)
  starts <- list(
    list(seed = 101, lambda = -0.5, omega2 = 0.05),
    list(seed = 102, lambda = 0, omega2 = 0.1),
    list(seed = 103, lambda = 0.5, omega2 = 0.5)
  )
  parallel::mclapply(starts, function(start) {
    set.seed(start$seed)
    endorse::endorse(
      Y = model_questions(), data = respondents, identical.lambda = TRUE,
      MCMC = iterations, burn = burn, thin = thin, covariates = FALSE, hierarchical = FALSE,
      lambda.start = start$lambda, omega2.start = start$omega2, verbose = FALSE
    )
  }, mc.cores = cores)
}

summarize_fits <- function(fits, spec) {
  stopifnot(spec %in% names(specifications), length(fits) == 3L)
  lambda_chains <- coda::mcmc.list(lapply(fits, function(fit) fit$lambda))
  omega_chains <- coda::mcmc.list(lapply(fits, function(fit) fit$omega2))
  rhat_lambda <- coda::gelman.diag(lambda_chains, multivariate = FALSE)$psrf[, 1]
  rhat_omega <- coda::gelman.diag(omega_chains, multivariate = FALSE)$psrf[, 1]
  ess_lambda <- coda::effectiveSize(lambda_chains)
  lambda <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$lambda)))
  omega2 <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$omega2)))
  x_sd <- unlist(lapply(fits, function(fit) as.numeric(fit$x)))
  n_groups <- length(endorsers)
  mu <- lambda
  p_positive <- pnorm(lambda / sqrt(omega2))
  mu_standardized <- mu / x_sd
  omega_standardized <- sqrt(omega2) / x_sd
  summarize_draws <- function(draws, prefix) {
    setNames(c(mean(draws), quantile(draws, c(0.025, 0.975))), paste0(prefix, c("_mean", "_lo", "_hi")))
  }
  make_row <- function(group, probability, reaction, dispersion, negative) {
    data.frame(spec, group, t(summarize_draws(probability, "p")), t(summarize_draws(reaction, "mu")),
      omega_mean = mean(dispersion), p_mu_neg = mean(negative),
      rhat_max_lambda = max(rhat_lambda), rhat_max_omega2 = max(rhat_omega), ess_min_lambda = min(ess_lambda)
    )
  }
  rows <- lapply(seq_len(n_groups), function(group) {
    make_row(
      unname(endorsers[group]), p_positive[, group], mu_standardized[, group],
      omega_standardized[, group], mu[, group] < 0
    )
  })
  average <- make_row(
    "Average", rowMeans(p_positive), rowMeans(mu_standardized),
    omega_standardized, rowMeans(mu) < 0
  )
  result <- do.call(rbind, c(rows, list(average)))
  result[, 3:11] <- round(result[, 3:11], 4)
  result
}
