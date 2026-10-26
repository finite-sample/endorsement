test_that("CSV preserves package responses and categorical model coding", {
  original <- new.env(parent = emptyenv())
  utils::data("pakistan", package = "endorse", envir = original)
  rownames(original$pakistan) <- NULL
  expect_equal(nrow(respondents), 5212L)
  expect_equal(respondents, original$pakistan)
  formula <- ~ female + rural + edu + inc
  expect_equal(model.matrix(formula, respondents), model.matrix(formula, original$pakistan))
  responses <- reshape_responses(respondents)
  expect_equal(nrow(responses), 4L * nrow(respondents))
  expect_true(all(table(responses$respondent, responses$policy) == 1L))
  expect_equal(sort(unique(responses$response)), seq(0, 1, 0.25))
})

test_that("invalid categories and missing policy responses fail before analysis", {
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path))
  invalid <- respondents
  invalid$edu <- 6L
  write.csv(invalid, path, row.names = FALSE)
  expect_error(read_responses(path), "Invalid category")
  invalid <- respondents
  invalid[1, model_questions()[[1]]] <- NA
  write.csv(invalid, path, row.names = FALSE)
  expect_error(read_responses(path), "exactly one")
})
