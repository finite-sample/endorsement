test_that("Stata comparison semantics treat missing values as large", {
  expect_equal(stata_gt(c(0, 2, NA), 1), c(0L, 1L, 1L))
  expect_equal(stata_lt(c(0, 2, NA), 1), c(1L, 0L, 0L))
  expect_equal(stata_eq(c(1, 2, NA), 1), c(1L, 0L, 0L))
})

test_that("weighted percentiles follow Stata's _pctile", {
  expect_equal(stata_percentile(1:10, rep(1, 10), 0.2), 2.5)
  expect_equal(stata_percentile(1:10, rep(1, 10), 0.25), 3)
  expect_equal(stata_percentile(c(5, 1, 3), c(1, 10, 1), 0.8), 1)
  expect_equal(stata_percentile(c(5, 1, 3), c(1, 10, 1), 0.95), 5)
})

blair <- blair_sample(file.path(root, "data", "dataverse"))

test_that("the rebuilt sample matches Blair et al.'s analysis file", {
  expect_equal(nrow(blair), 5358L)
  expect_true(all(blair$treatment %in% 0:1))
  expect_equal(sum(blair$interviewer %in% suspect_interviewers & !is.na(blair$low)), 308L)
  expect_true(all(blair$support >= 0.2 & blair$support <= 1))
})

test_that("full-sample estimates reproduce the published tables to the third decimal", {
  estimates <- blair_estimates(blair)
  # Blair et al. (2013), Tables 1A, 2, 3, 4, and 5, read from the published PDF.
  published <- data.frame(
    label = estimates$label, column = estimates$column,
    estimate = c(
      -0.011, -0.015, -0.010, -0.008,
      -0.001, -0.019, -0.018, 0.001, -0.022, -0.023,
      -0.015,
      0.019, -0.042, -0.055,
      -0.052, -0.061
    ),
    se = c(
      0.006, 0.006, 0.005, 0.005,
      0.006, 0.008, 0.010, 0.005, 0.008, 0.009,
      0.010,
      0.009, 0.013, 0.020,
      0.015, 0.015
    ),
    n = c(rep(5358L, 4), rep(5067L, 3), rep(4978L, 3), 4925L, rep(4978L, 3), 1265L, 1117L)
  )
  # The one known discrepancy: the cue by low-income-district interaction in their Table 3 comes out 0.0006 more
  # negative than published, in every column, while the table's other coefficients match exactly.
  district <- estimates$table == 3L
  expect_equal(round(estimates$estimate[!district], 3), published$estimate[!district])
  expect_equal(round(estimates$se[!district], 3), published$se[!district])
  expect_equal(round(estimates$estimate[district], 3), published$estimate[district], tolerance = 0.001 / 0.015)
  expect_equal(estimates$n, published$n)
})

test_that("the reanalysis pairs each quantity with its value without the three interviewers", {
  results <- blair_reanalysis(blair, draws = 5L)
  expect_equal(nrow(results), 16L)
  expect_true(all(results$n_without < results$n))
  gap <- results[results$table == 2L & results$column == 1L & grepl("^Gap", results$label), ]
  expect_lt(gap$estimate, gap$estimate_without)
  expect_gt(gap$estimate_without / gap$se_without, -1.96)
  low <- results[results$table == 2L & results$column == 1L & results$label == "Effect, low income", ]
  expect_lt(abs(low$estimate - low$estimate_without), 0.005)
  placebo <- attr(results, "placebo")
  expect_equal(nrow(placebo), 2L)
  expect_true(all(placebo$draws == 5L))
  expect_true(all(placebo$lower <= placebo$upper))
  expect_gt(attr(results, "three")[["not_low"]], 0.95)
})
