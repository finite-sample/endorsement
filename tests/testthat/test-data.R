test_that("the analysis file reverses Blair et al.'s scale and keeps only complete, attributed interviews", {
  raw <- foreign::read.dta(file.path(root, "data", "dataverse", "fms09-data-weighted-20june.dta"),
    convert.factors = FALSE
  )
  expect_equal(nrow(respondents), 5479L)
  expect_equal(length(unique(respondents$interviewer)), 75L)
  source_rows <- raw[respondents$respondent, ]
  expect_equal(respondents$interviewer, source_rows$name)
  for (policy in policies) {
    for (code in c("a", names(endorsers))) {
      recorded <- as.integer(as.character(respondents[[paste(policy, code, sep = ".")]]))
      expect_equal(recorded, 6L - source_rows[[paste0(source_items[[policy]], code)]])
    }
  }
  responses <- reshape_responses(respondents)
  expect_equal(nrow(responses), 4L * nrow(respondents))
  expect_equal(sort(unique(responses$response)), seq(0, 1, 0.25))
})

test_that("the endorse package's respondents are in the analysis file or lack a recorded interviewer", {
  package <- new.env(parent = emptyenv())
  utils::data("pakistan", package = "endorse", envir = package)
  key <- function(data) {
    apply(data[response_columns()], 1, function(row) paste(ifelse(is.na(row), "", trimws(row)), collapse = "|"))
  }
  shortfall <- function(needed, available) {
    have <- as.numeric(available[names(needed)])
    have[is.na(have)] <- 0
    setNames(pmax(0, as.numeric(needed) - have), names(needed))
  }
  missing <- shortfall(table(key(package$pakistan)), table(key(respondents)))
  missing <- missing[missing > 0]
  raw <- foreign::read.dta(file.path(root, "data", "dataverse", "fms09-data-weighted-20june.dta"),
    convert.factors = FALSE
  )
  unattributed <- raw[is.na(raw$name), ]
  columns <- lapply(response_columns(), function(column) {
    parts <- strsplit(column, ".", fixed = TRUE)[[1]]
    value <- 6L - unattributed[[paste0(source_items[[parts[1]]], parts[2])]]
    ifelse(value %in% 1:5, value, NA)
  })
  unattributed_keys <- table(key(setNames(as.data.frame(columns), response_columns())))
  expect_lte(sum(missing), sum(is.na(raw$name)))
  expect_true(all(shortfall(missing, unattributed_keys) == 0))
})

test_that("invalid categories and missing policy responses fail before analysis", {
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path))
  invalid <- respondents
  invalid$Polio.a <- ifelse(is.na(invalid$Polio.a), NA, 6L)
  write.csv(invalid, path, row.names = FALSE)
  expect_error(read_responses(path), "Invalid category")
  invalid <- respondents
  invalid[1, model_questions()[[1]]] <- NA
  write.csv(invalid, path, row.names = FALSE)
  expect_error(read_responses(path), "exactly one")
})
