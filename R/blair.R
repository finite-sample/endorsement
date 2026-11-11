# Blair et al.'s published estimates, rebuilt from their replication files (replication_dataprep.do and
# replication_analysis.do, ported from Stata) and recomputed without interviewers 7, 37, and 71. Every
# full-sample estimate is checked against the published tables in tests/testthat/test-blair.R.

# Stata treats a missing value as larger than any number, so `x > c` is true and `x < c` and `x == c` are
# false when x is missing. Blair et al.'s controls inherit that coding; these helpers reproduce it.
stata_gt <- function(x, cutoff) as.integer(is.na(x) | x > cutoff)
stata_lt <- function(x, cutoff) as.integer(!is.na(x) & x < cutoff)
stata_eq <- function(x, value) as.integer(!is.na(x) & x == value)

# Stata's _pctile with pweights: the first value whose cumulative weight passes p of the total, averaged with
# the next value when the cumulative weight lands exactly on p.
stata_percentile <- function(x, weights, p) {
  order <- order(x)
  x <- x[order]
  cumulative <- cumsum(weights[order])
  target <- p * cumulative[length(cumulative)]
  index <- which(cumulative > target)[1]
  if (index > 1 && isTRUE(all.equal(cumulative[index - 1], target))) (x[index - 1] + x[index]) / 2 else x[index]
}

# Blair et al. divide the 1-5 answer by 5, so one step on their outcome is 0.2; on this paper's 0-1 scale a
# step is 0.25. Multiplying their estimates by 1.25 puts them on the 0-1 scale, and by 125 on the 0-100 scale.
blair_scale <- 1.25

blair_demographics <- c("female", "married", "age", "online", "cellphone", "read", "write", "math", "educ2", "sunni")
blair_attitudes <- c("usviews1", "usviews2", "sharia_phys", "demvalues2")

# One row per respondent in Blair et al.'s `complete` sample, with their outcome, treatment indicator, income
# classes, controls, district measures, survey design variables, and the interviewer.
blair_sample <- function(directory = "data/dataverse") {
  path <- function(name) file.path(directory, name)
  raw <- foreign::read.dta(path("fms09-data-weighted-20june.dta"), convert.factors = FALSE)
  labels <- attr(raw, "label.table")
  district <- trimws(names(labels$d24)[match(raw$d24, labels$d24)])
  district[grepl("karachi", district)] <- "karachi"
  raw$spn[raw$spn %in% 98] <- 501
  raw$spn[raw$spn %in% 99] <- 502
  for (name in setdiff(names(raw), c("spn", "name"))) {
    if (is.numeric(raw[[name]])) raw[[name]][raw[[name]] %in% c(98, 99)] <- NA
  }
  items <- unname(source_items)
  arm <- function(code) as.matrix(raw[paste0(items, code)])
  substantive <- function(values) !is.na(values) & values < 6
  answered <- sapply(names(endorsers), function(code) rowSums(!is.na(arm(code))))
  any_endorsed <- rowSums(sapply(names(endorsers), function(code) rowSums(substantive(arm(code))))) > 0
  complete <- (rowSums(substantive(arm("a"))) == 4 | rowSums(answered == 1) == 4) &
    rowSums(answered > 1) == 0 & !(rowSums(substantive(arm("a"))) > 0 & any_endorsed)
  reversed <- function(code) 6 - arm(code)
  control_mean <- rowSums(reversed("a"), na.rm = TRUE) / 20
  control_mean[!(control_mean > 0 & complete)] <- NA
  # Each endorsed respondent rates one policy per group; Blair et al. divide that answer by 5 and weight it by
  # the policy that carried it. The control weight is the average of the four policy weights.
  policy_weights <- c(0.983415, 1.150536, 1.278917, 1.182913)
  group_outcome <- sapply(names(endorsers), function(code) {
    own <- rowSums(reversed(code), na.rm = TRUE) / 5
    own[!(own > 0 & complete)] <- NA
    rowSums(cbind(control_mean, own), na.rm = TRUE)
  })
  group_weight <- sapply(names(endorsers), function(code) {
    weight <- ifelse(!is.na(control_mean), 1.148945, NA)
    for (index in seq_along(items)) {
      carried <- is.na(weight) & !is.na(raw[[paste0(items[index], code)]]) & complete
      weight[carried] <- policy_weights[index]
    }
    weight * raw$wtpru
  })
  education <- raw$d11
  education[education %in% 7] <- 0
  agrees <- function(item) stata_lt(raw[[item]], 2)
  democratic <- c("q360", "q340", "q320", "q300", "q370", "q350")
  sample <- data.frame(
    support = rowMeans(group_outcome), treatment = as.integer(is.na(raw$q800a)),
    province = raw$d23, urban = stata_eq(raw$d25, 1), earnings = raw$d13, weight = raw$wtpru, point = raw$spn,
    interviewer = raw$name, district = district,
    female = raw$d22 - 1, married = 2 - raw$d1, age = (raw$d2 - 18) / 70, online = 2 - raw$d6,
    cellphone = 2 - raw$d7b, read = stata_gt(raw$d8, 1), write = stata_gt(raw$d9, 1), math = stata_eq(raw$d10, 1),
    educ2 = education / 6, sunni = stata_eq(raw$d19, 1),
    usviews1 = (raw$q500 - 1) / 4, usviews2 = (raw$q510 - 1) / 4, sharia_phys = stata_eq(raw$q160e, 1),
    demvalues2 = rowSums(sapply(democratic, agrees)) / 6
  )
  for (index in seq_along(endorsers)) {
    sample[[paste0("support_", index)]] <- group_outcome[, index]
    sample[[paste0("weight_", index)]] <- group_weight[, index]
  }
  # Low and high income: below the 20th and above the 80th weighted percentile of monthly cash earnings
  # within province and urban/rural.
  sample$low <- sample$high <- ifelse(is.na(sample$earnings), NA_integer_, 0L)
  for (urban in 0:1) {
    for (province in 1:4) {
      cell <- !is.na(sample$earnings) & sample$province == province & sample$urban == urban
      cuts <- sapply(c(0.2, 0.8), function(p) stata_percentile(sample$earnings[cell], sample$weight[cell], p))
      sample$low[cell & sample$earnings <= cuts[1]] <- 1L
      sample$high[cell & sample$earnings > cuts[2]] <- 1L
    }
  }
  income <- foreign::read.dta(path("district_income_data.dta"))
  matched <- match(sample$district, income$district_merge)
  sample$low_district <- income$low_inc_dist_median[matched]
  sample$high_district <- income$high_inc_dist_median[matched]
  # Districts absent from the violence file had no recorded incidents, as in Blair et al.'s code.
  violence <- foreign::read.dta(path("district_violence_data.dta"))
  matched <- match(sample$district, trimws(violence$d24))
  sample$incidents <- ifelse(is.na(matched), 0, violence$tot_milv[matched])
  sample$casualties <- ifelse(is.na(matched), 0, violence$tot_milv_cas[matched])
  sample[complete, ]
}

# Blair et al.'s specifications: province fixed effects (column 1), plus demographic controls (2), plus
# attitudinal controls (3).
blair_controls <- function(column) {
  stopifnot(column %in% 1:3)
  c("factor(province)", if (column >= 2) blair_demographics, if (column >= 3) blair_attitudes)
}

# A linear combination of coefficients from a survey-weighted regression with the design Blair et al. use:
# sampling points as clusters, provinces as strata, and their weights.
blair_estimate <- function(sample, outcome, terms, controls, weight = "weight", combination = terms) {
  formula <- stats::as.formula(paste(outcome, "~", paste(c(terms, controls), collapse = " + ")))
  used <- unique(c(all.vars(formula), weight, "point", "province"))
  sample <- sample[stats::complete.cases(sample[used]), ]
  design <- survey::svydesign(
    ids = ~point, strata = ~province, weights = stats::as.formula(paste("~", weight)), data = sample, nest = TRUE
  )
  model <- survey::svyglm(formula, design = design)
  contrast <- survey::svycontrast(model, stats::setNames(rep(1, length(combination)), combination))
  c(estimate = unname(stats::coef(contrast)), se = unname(survey::SE(contrast)), n = nrow(sample))
}

# The published estimates recomputed here: the table and column they come from, and the quantity.
blair_quantities <- function() {
  income <- c("treatment", "low", "high", "treatment:low", "treatment:high")
  urban <- "treatment * (low + high) * urban"
  quantity <- function(table, column, label, terms, controls = blair_controls(column), ...) {
    list(table = table, column = column, label = label, terms = terms, controls = controls, ...)
  }
  c(
    lapply(seq_along(endorsers), function(index) {
      quantity(1L, index, paste("Endorsement effect,", endorsers[[index]]), "treatment", character(),
        outcome = paste0("support_", index), weight = paste0("weight_", index)
      )
    }),
    unlist(lapply(c(1L, 3L), function(column) {
      list(
        quantity(2L, column, "Effect, middle income", income, combination = "treatment"),
        quantity(2L, column, "Effect, low income", income, combination = c("treatment", "treatment:low")),
        quantity(2L, column, "Gap: low minus middle income", income, combination = "treatment:low")
      )
    }), recursive = FALSE),
    list(quantity(3L, 1L, "Gap: low- minus middle-income district",
      c("treatment", "low_district", "high_district", "treatment:low_district", "treatment:high_district"),
      combination = "treatment:low_district"
    )),
    list(
      quantity(4L, 3L, "Effect, urban middle income", urban, combination = c("treatment", "treatment:urban")),
      quantity(4L, 3L, "Effect, urban low income", urban,
        combination = c("treatment", "treatment:low", "treatment:urban", "treatment:low:urban")
      ),
      quantity(4L, 3L, "Gap: low minus middle income, urban minus rural", urban, combination = "treatment:low:urban")
    ),
    list(
      quantity(5L, 1L, "Gap: low minus middle income, urban, violent district (incidents)", income,
        controls = blair_controls(3L), combination = "treatment:low",
        subset = function(sample) sample$incidents > 0 & sample$urban == 1
      ),
      quantity(5L, 4L, "Gap: low minus middle income, urban, violent district (casualties)", income,
        controls = blair_controls(3L), combination = "treatment:low",
        subset = function(sample) sample$casualties > 0 & sample$urban == 1
      )
    )
  )
}

blair_quantity <- function(sample, quantity) {
  if (!is.null(quantity$subset)) sample <- sample[quantity$subset(sample), ]
  default <- function(value, fallback) if (is.null(value)) fallback else value
  blair_estimate(
    sample, default(quantity$outcome, "support"), quantity$terms, quantity$controls,
    default(quantity$weight, "weight"), default(quantity$combination, quantity$terms)
  )
}

blair_estimates <- function(sample, quantities = blair_quantities()) {
  rows <- lapply(quantities, function(quantity) {
    cbind(data.frame(table = quantity$table, column = quantity$column, label = quantity$label),
      t(blair_quantity(sample, quantity))
    )
  })
  do.call(rbind, rows)
}

# Each published quantity with and without the three interviewers, and, for the central estimates, how far
# dropping three interviewers chosen at random moves them.
blair_reanalysis <- function(sample, draws = 1000L, seed = 1L) {
  quantities <- blair_quantities()
  full <- blair_estimates(sample, quantities)
  without <- blair_estimates(sample[!sample$interviewer %in% suspect_interviewers, ], quantities)
  stopifnot(identical(full$label, without$label))
  results <- data.frame(
    full[c("table", "column", "label")], estimate = full$estimate, se = full$se, n = full$n,
    estimate_without = without$estimate, se_without = without$se, n_without = without$n
  )
  gaps <- grepl("^Gap", results$label)
  central <- which(gaps & (results$table == 2L & results$column == 1L | results$table == 4L))
  interviewers <- unique(stats::na.omit(sample$interviewer))
  set.seed(seed)
  triples <- replicate(draws, sample(interviewers, 3L), simplify = FALSE)
  placebo <- do.call(rbind, lapply(central, function(index) {
    quantity <- quantities[[index]]
    estimates <- sapply(triples, function(dropped) {
      blair_quantity(sample[!sample$interviewer %in% dropped, ], quantity)[["estimate"]]
    })
    data.frame(
      label = quantity$label, draws = draws, lower = unname(stats::quantile(estimates, 0.05)),
      upper = unname(stats::quantile(estimates, 0.95)),
      share_as_close = mean(estimates >= results$estimate_without[index])
    )
  }))
  three <- sample[sample$interviewer %in% suspect_interviewers & !is.na(sample$low), ]
  attr(results, "placebo") <- placebo
  attr(results, "three") <- c(
    respondents = nrow(three), not_low = mean(three$low == 0), urban = mean(three$urban == 1),
    violent = mean(three$incidents > 0)
  )
  results
}
