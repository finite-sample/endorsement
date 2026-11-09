policies <- c("Polio", "FCR", "Durand", "Curriculum")
endorsers <- c(
  b = "Kashmiri tanzeems", c = "Afghan Taliban",
  d = "Al-Qaeda", e = "Sectarian tanzeems"
)
# Survey items in Blair et al.'s file; suffixes a-e follow the endorser codes above.
source_items <- c(Polio = "q800", FCR = "q810", Durand = "q830", Curriculum = "q840")
# The three interviewers whose recorded answers produce the midpoint pattern (see the manuscript).
suspect_interviewers <- c(7L, 37L, 71L)
specifications <- c(full = "All interviews", clean = "Without interviewers 7, 37, and 71")
response_categories <- seq(0, 1, by = 0.25)
response_columns <- function() as.vector(t(outer(policies, c("a", names(endorsers)), paste, sep = ".")))
model_questions <- function() {
  setNames(lapply(policies, function(policy) {
    paste(policy, c("a", names(endorsers)), sep = ".")
  }), paste0("Q", seq_along(policies)))
}

read_responses <- function(path = "data/pakistan_endorsement.csv") {
  respondents <- read.csv(path, check.names = FALSE)
  required <- c("respondent", "province", "district", "interviewer", "female", "rural", response_columns())
  stopifnot(all(required %in% names(respondents)))
  for (column in response_columns()) {
    observed <- respondents[[column]][!is.na(respondents[[column]])]
    if (!all(observed %in% 1:5)) stop("Invalid category in ", column)
    respondents[[column]] <- factor(respondents[[column]], levels = 1:5)
  }
  for (columns in model_questions()) {
    if (any(rowSums(!is.na(respondents[columns])) != 1L)) {
      stop("Each respondent must have exactly one observed response per policy")
    }
  }
  if (anyNA(respondents[c("province", "district", "interviewer")])) {
    stop("Every respondent needs a province, district, and interviewer")
  }
  respondents
}

# Builds the analysis file from Blair et al.'s replication data. The survey codes 1 = "a great deal"
# through 5 = "not at all", with 98 and 99 for don't know and no response; the analysis file reverses
# the scale so that 5 is the most support. A respondent enters the sample with exactly one
# substantive answer to each policy and a recorded interviewer.
prepare_data <- function(source = "data/dataverse/fms09-data-weighted-20june.dta",
                         path = "data/pakistan_endorsement.csv") {
  raw <- foreign::read.dta(source, convert.factors = FALSE)
  labels <- attr(raw, "label.table")
  label <- function(variable) trimws(names(labels[[variable]])[match(raw[[variable]], labels[[variable]])])
  out <- data.frame(
    respondent = seq_len(nrow(raw)), province = label("d23"), district = label("d24"),
    interviewer = raw$name, female = as.integer(raw$d22 == 2), rural = as.integer(raw$d25 == 2)
  )
  keep <- !is.na(out$interviewer)
  for (policy in policies) {
    values <- as.matrix(raw[paste0(source_items[[policy]], c("a", names(endorsers)))])
    values[!(values %in% 1:5)] <- NA
    keep <- keep & rowSums(!is.na(values)) == 1L
    for (index in seq_len(ncol(values))) {
      out[[paste(policy, c("a", names(endorsers))[index], sep = ".")]] <- 6L - values[, index]
    }
  }
  write.csv(out[keep, ], path, row.names = FALSE)
  write_questionnaire(raw)
  invisible(read_responses(path))
}

# Every closed-ended question asked of at least half of respondents, for the near-duplicate check. Don't know
# and no response stay as their own codes, because a copied interview copies them too.
write_questionnaire <- function(raw, path = "data/questionnaire.csv") {
  items <- grep("^q", names(raw), value = TRUE)
  usable <- function(column) mean(!is.na(column)) >= 0.5 && length(unique(stats::na.omit(column))) > 1
  items <- items[sapply(raw[items], usable)]
  keep <- !is.na(raw$name)
  out <- data.frame(respondent = seq_len(nrow(raw)), point = raw$spn, interviewer = raw$name, raw[items])[keep, ]
  write.csv(out, path, row.names = FALSE)
}

reshape_responses <- function(respondents) {
  pieces <- lapply(response_columns(), function(column) {
    observed <- which(!is.na(respondents[[column]]))
    data.frame(
      respondent = observed,
      interviewer = respondents$interviewer[observed],
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
