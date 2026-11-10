test_that("each kind of quantity prints at its configured precision", {
  expect_identical(fmt(-0.0128337, "scale"), "-0.013")
  expect_identical(fmt(-0.0017, "points", signed = TRUE), "$-0.2$")
  expect_identical(fmt(0.3812, "percent"), "38\\%")
  expect_identical(fmt(0.0567, "respondents"), "5.7\\%")
  expect_identical(fmt(93.14, "statistic"), "93.1")
  expect_identical(fmt(-0.0004, "points", signed = TRUE), "$0.0$")
  expect_error(fmt(0.0004, "p"), "inequality")
  expect_error(fmt(NA_real_, "percent"), "Missing")
  expect_identical(p_relation(0.0183), "=0.018")
  expect_identical(p_relation(8e-6), "<0.001")
})

test_that("only the formatter chooses decimal places", {
  code <- unlist(lapply(file.path(root, "R", c("paper.R", "analysis.R")), readLines))
  expect_false(any(grepl("%[+]?[.][0-9]f", code)))
  expect_false(any(grepl("\\bround\\([^)]*, [0-9]", code)))
})

test_that("no printed number carries more decimals than any configured kind", {
  files <- list.files(file.path(root, "tabs"), "\\.tex$", full.names = TRUE)
  text <- unlist(lapply(files, readLines))
  decimals <- nchar(unlist(regmatches(text, gregexpr("(?<=[0-9][.])[0-9]+", text, perl = TRUE))))
  expect_lte(max(decimals), max(precision))
})
