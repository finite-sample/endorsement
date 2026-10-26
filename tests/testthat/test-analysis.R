test_that("moment restrictions leave infeasible fractions undefined", {
  expect_equal(symmetric_effects(c(-0.05, -0.02, 0, 0.02, 0.05), 0.02), c(NA, 0, 0.5, 1, NA))
  expect_equal(extreme_opponent(c(-1.1, -1, -0.013, 0, 0.01)), c(NA, 0, 0.987, 1, NA))
  expect_true(is.na(binary_switching(0.74, 0.3)))
  expect_true(is.na(discrete_scale(0.1, c(0, 0, 0, 0, 1), 1)$fraction))
  expect_error(symmetric_effects(0, 0), "positive")
  curves <- sensitivity_data(0.74)
  symmetric <- curves[curves$restriction == "Symmetric effects (d = 0.02)", ]
  expect_true(all(is.na(symmetric$fraction[abs(symmetric$ate) > 0.02])))
  expect_equal(range(symmetric$fraction, na.rm = TRUE), c(0, 1))
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
  expect_equal(unname(analysis$bounds["All groups", c("lower", "upper")]),
    c(0.011908176659755743, 0.5804689707128732),
    tolerance = 1e-12
  )
  expect_equal(unname(analysis$anomalies), c(35, 94))
  expect_equal(analysis$excluded, 129L)
  expect_equal(analysis$clean_ate, -0.012719900088725256, tolerance = 1e-12)
})
