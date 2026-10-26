test_that("posterior summaries average probabilities before reporting uncertainty", {
  set.seed(184)
  design <- cbind(1, c(0, 0, 1))
  division <- c(1L, 1L, 2L)
  for (spec in names(specifications)) {
    coefficients <- if (spec == "primary") 4L else 8L
    fits <- lapply(1:3, function(chain) {
      list(
        lambda = coda::mcmc(matrix(rnorm(200 * coefficients, sd = 0.5), ncol = coefficients)),
        omega2 = coda::mcmc(matrix(runif(800, 0.1, 0.5), ncol = 4)), x = rep(2, 200),
        model.matrix.indiv = design, village.indicator = division
      )
    })
    lambda <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$lambda)))
    omega2 <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$omega2)))
    probabilities <- reactions <- matrix(NA_real_, 600, 4)
    for (draw in 1:600) {
      for (group in 1:4) {
        if (spec == "primary") {
          person_means <- lambda[draw, group]
        } else if (spec == "covariates") {
          person_means <- design %*% lambda[draw, (group - 1L) * 2L + 1:2]
        } else {
          person_means <- lambda[draw, (group - 1L) * 2L + division]
        }
        probabilities[draw, group] <- mean(pnorm(person_means / sqrt(omega2[draw, group])))
        reactions[draw, group] <- mean(person_means) / 2
      }
    }
    actual <- summarize_fits(fits, spec)
    expect_equal(actual$p_mean, round(c(colMeans(probabilities), mean(probabilities)), 4))
    expect_equal(actual$mu_mean, round(c(colMeans(reactions), mean(reactions)), 4))
    expect_equal(actual$p_lo[5], unname(round(quantile(rowMeans(probabilities), 0.025), 4)))
    expect_equal(actual$p_hi[5], unname(round(quantile(rowMeans(probabilities), 0.975), 4)))
    expect_equal(actual$omega_mean, unname(round(c(colMeans(sqrt(omega2) / 2), mean(sqrt(omega2) / 2)), 4)))
  }
})

test_that("all model specifications fit the validated CSV data", {
  for (spec in names(specifications)) {
    fits <- suppressMessages(fit_model(respondents[1:150, ], spec, iterations = 200L, burn = 100L, thin = 2L))
    expect_length(fits, 3L)
    for (fit in fits) {
      expect_equal(nrow(fit$lambda), 50L)
      expect_true(all(is.finite(fit$lambda)))
    }
    if (spec == "covariates") expect_equal(ncol(fits[[1]]$model.matrix.indiv), 8L)
  }
})
