test_that("moment restrictions leave infeasible fractions undefined", {
  expect_equal(symmetric_effects(c(-0.05, -0.02, 0, 0.02, 0.05), 0.02), c(NA, 0, 0.5, 1, NA))
  expect_equal(extreme_opponent(c(-1.1, -1, -0.013, 0, 0.01)), c(NA, 0, 0.987, 1, NA))
  expect_true(is.na(binary_switching(0.74, 0.3)))
  expect_true(is.na(discrete_scale(0.1, c(0, 0, 0, 0, 1), 1)$fraction))
  expect_error(symmetric_effects(0, 0), "positive")
})

test_that("strict-increase bounds handle ties and point masses", {
  uniform <- pairing_bounds(rep(0.2, 5), rep(0.2, 5))
  expect_equal(unname(uniform), c(0.4, 0.2, 0.4, 0, 0.8))
  for (treatment in 1:5) {
    for (control in 1:5) {
      bounds <- pairing_bounds(as.numeric(1:5 == treatment), as.numeric(1:5 == control))
      expect_equal(unname(bounds[c("lower", "upper")]), rep(as.numeric(treatment > control), 2))
    }
  }
  expect_error(pairing_bounds(c(1, 1, 0, 0, 0), rep(0.2, 5)), "Invalid")
})

test_that("bounds attain the extrema over all pairings of small samples", {
  permutations <- function(values) {
    if (length(values) == 1L) {
      return(matrix(values, nrow = 1L))
    }
    do.call(rbind, lapply(seq_along(values), function(index) {
      cbind(values[index], permutations(values[-index]))
    }))
  }
  treatment <- c(0, 0.25, 0.75, 1)
  control <- c(0.25, 0.5, 0.5, 1)
  fractions <- rowMeans(sweep(permutations(treatment), 2, control, ">"))
  bounds <- pairing_bounds(category_distribution(treatment), category_distribution(control))
  expect_equal(unname(bounds[c("lower", "upper")]), range(fractions))
})

test_that("mixed-arm cluster covariance matches the full matrix sandwich", {
  responses <- data.frame(
    respondent = rep(1:4, each = 2), endorser = c("a", "b", "a", "b", "a", "a", "b", "b"),
    response = (c(1, 2, 3, 5, 5, 4, 4, 5) - 1) / 4
  )
  design <- cbind(1, as.numeric(responses$endorser != "a"))
  bread <- solve(crossprod(design))
  residual <- responses$response - design %*% bread %*% crossprod(design, responses$response)
  scores <- rowsum(design * as.numeric(residual), responses$respondent)
  covariance <- bread %*% crossprod(scores) %*% bread
  expect_equal(clustered_ate_se(responses), sqrt(covariance[2, 2]))
})

test_that("empirical results survive the language migration", {
  analysis <- analyze_responses(respondents, list())
  expect_equal(length(analysis$control), 10619L)
  expect_equal(length(analysis$treated), 10229L)
  expect_equal(analysis$alpha, 0.735850833411809, tolerance = 1e-12)
  expect_equal(analysis$beta, -0.012832943099950533, tolerance = 1e-12)
  expect_equal(analysis$se, 0.00520879916129222, tolerance = 1e-12)
  expect_equal(unname(analysis$bounds[c("lower", "upper")]),
    c(0.011908176659755743, 0.5804689707128732),
    tolerance = 1e-12
  )
  expect_equal(unname(analysis$anomalies), c(35, 94))
  expect_equal(analysis$excluded, 129L)
  expect_equal(analysis$clean_ate, -0.012719900088725256, tolerance = 1e-12)
})

test_that("within-respondent moments separate shared reactions from independent noise", {
  make <- function(values, codes) {
    responses <- expand.grid(policy = policies, respondent = seq_len(nrow(values)), stringsAsFactors = FALSE)
    responses$response <- as.vector(t(values))
    responses$endorser <- codes[responses$respondent]
    responses[, c("respondent", "policy", "endorser", "response")]
  }
  set.seed(7)
  n <- 400L
  base <- matrix(rep(stats::rnorm(2L * n), each = 4L), ncol = 4L, byrow = TRUE)
  values <- (base + 0.5 * matrix(stats::rnorm(8L * n), ncol = 4L))
  values[(n + 1L):(2L * n), ] <- values[(n + 1L):(2L * n), ] + matrix(rep(stats::rnorm(n), 4L), ncol = 4L)
  codes <- rep(c("a", "b"), each = n)
  moments <- within_respondent_moments(make(values, codes), draws = 200L)
  expect_equal(nrow(moments), 6L)
  expect_true(all(moments$difference > 0))
  expect_true(all(moments$lower <= moments$difference & moments$difference <= moments$upper))
  # Independent noise added to the endorsed arm leaves the correlations unchanged in expectation.
  noisy <- values
  noisy[(n + 1L):(2L * n), ] <- base[(n + 1L):(2L * n), ] + 0.5 * matrix(stats::rnorm(4L * n), ncol = 4L)
  noise_only <- within_respondent_moments(make(noisy, codes), draws = 200L)
  expect_true(all(noise_only$lower < 0 & noise_only$upper > 0))
  mixed <- make(values, codes)
  mixed$endorser[mixed$respondent == 1L & mixed$policy == "FCR"] <- "c"
  expect_equal(nrow(respondent_matrix(mixed)$values), 2L * n - 1L)
})

test_that("person-mean distributions are shares that sum to one in each arm", {
  distribution <- person_mean_distribution(reshape_responses(respondents), draws = 50L)
  for (table in distribution) {
    expect_equal(sum(table$control), 1)
    expect_equal(sum(table$treated), 1)
    expect_equal(table$difference, table$treated - table$control)
    expect_true(all(table$lower <= table$upper))
  }
  expect_equal(nrow(distribution$full), 17L)
  expect_equal(distribution$bins$control[distribution$bins$bin == "1"], distribution$full$control[17L])
})

test_that("simulated opposition alone moves mass through the midpoint without emptying it", {
  worlds <- reaction_worlds(reshape_responses(respondents), simulations = 100L)
  one_sided <- worlds[worlds$world == "Opposition only", ]
  expect_true(all(abs(one_sided$midpoint) < 0.005))
  expect_true(all(diff(one_sided$oppose) < 0))
  both <- worlds[worlds$world == "Both directions", ]
  expect_true(all(diff(both$midpoint) < 0))
  expect_lt(worlds$midpoint_upper[worlds$world == "Observed"], 0)
})
