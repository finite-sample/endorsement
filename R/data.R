policies <- c("Polio", "FCR", "Durand", "Curriculum")
endorsers <- c(
  b = "Kashmiri tanzeems", c = "Afghan Taliban",
  d = "Al-Qaeda", e = "Sectarian tanzeems"
)
specifications <- c(primary = "Baseline", covariates = "Covariates", hierarchical = "Hierarchical")
response_categories <- seq(0, 1, by = 0.25)
response_columns <- function() as.vector(t(outer(policies, c("a", names(endorsers)), paste, sep = ".")))
model_questions <- function() {
  setNames(lapply(policies, function(policy) {
    paste(policy, c("a", names(endorsers)), sep = ".")
  }), paste0("Q", seq_along(policies)))
}

read_responses <- function(path = "data/pakistan_endorsement.csv") {
  respondents <- read.csv(path, check.names = FALSE)
  required <- c("division", "edu", "female", "inc", "rural", response_columns())
  stopifnot(all(required %in% names(respondents)))
  for (column in c("edu", response_columns())) {
    observed <- respondents[[column]][!is.na(respondents[[column]])]
    if (!all(observed %in% 1:5)) stop("Invalid category in ", column)
    respondents[[column]] <- factor(respondents[[column]], levels = 1:5)
  }
  for (columns in model_questions()) {
    if (any(rowSums(!is.na(respondents[columns])) != 1L)) {
      stop("Each respondent must have exactly one observed response per policy")
    }
  }
  if (anyNA(respondents[c("division", "edu", "female", "inc", "rural")])) {
    stop("Missing model covariates would change the analysis sample")
  }
  respondents
}

prepare_data <- function(path = "data/pakistan_endorsement.csv") {
  source_data <- new.env(parent = emptyenv())
  utils::data("pakistan", package = "endorse", envir = source_data)
  write.csv(source_data$pakistan, path, row.names = FALSE)
  invisible(read_responses(path))
}

reshape_responses <- function(respondents) {
  pieces <- lapply(response_columns(), function(column) {
    observed <- which(!is.na(respondents[[column]]))
    data.frame(
      respondent = observed,
      policy = sub("\\.[a-e]$", "", column),
      endorser = sub("^.*\\.", "", column),
      response = (as.integer(as.character(respondents[[column]][observed])) - 1) / 4
    )
  })
  responses <- do.call(rbind, pieces)
  rownames(responses) <- NULL
  stopifnot(nrow(responses) == length(policies) * nrow(respondents))
  responses
}

read_posteriors <- function() {
  lapply(setNames(names(specifications), names(specifications)), function(spec) {
    posterior <- read.csv(file.path("data", paste0("endorse_", spec, ".csv")))
    expected <- c(unname(endorsers), "Average")
    stopifnot(!anyDuplicated(posterior$group), setequal(posterior$group, expected))
    posterior[match(expected, posterior$group), ]
  })
}
