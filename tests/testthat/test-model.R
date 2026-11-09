test_that("posterior summaries average probabilities before reporting uncertainty", {
  set.seed(184)
  fits <- lapply(1:3, function(chain) {
    list(
      lambda = coda::mcmc(matrix(rnorm(800, sd = 0.5), ncol = 4)),
      omega2 = coda::mcmc(matrix(runif(800, 0.1, 0.5), ncol = 4)), x = rep(2, 200)
    )
  })
  lambda <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$lambda)))
  omega2 <- do.call(rbind, lapply(fits, function(fit) as.matrix(fit$omega2)))
  probabilities <- pnorm(lambda / sqrt(omega2))
  actual <- summarize_fits(fits, "full")
  expect_equal(actual$p_mean, unname(round(c(colMeans(probabilities), mean(probabilities)), 4)))
  expect_equal(actual$mu_mean, unname(round(c(colMeans(lambda / 2), mean(lambda / 2)), 4)))
  expect_equal(actual$p_lo[5], unname(round(quantile(rowMeans(probabilities), 0.025), 4)))
  expect_equal(actual$omega_mean, unname(round(c(colMeans(sqrt(omega2) / 2), mean(sqrt(omega2) / 2)), 4)))
})

test_that("the clean specification drops the three interviewers and both specifications fit", {
  expect_false(any(model_sample(respondents, "clean")$interviewer %in% suspect_interviewers))
  expect_equal(nrow(model_sample(respondents, "full")), nrow(respondents))
  for (spec in names(specifications)) {
    fits <- fit_model(respondents[1:300, ], spec, iterations = 200L, burn = 100L, thin = 2L, cores = 1L)
    expect_length(fits, 3L)
    for (fit in fits) {
      expect_equal(nrow(fit$lambda), 50L)
      expect_true(all(is.finite(fit$lambda)))
    }
  }
})
