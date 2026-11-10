# Decimal places for every number the paper prints, one setting per kind of quantity. Each is set near the
# first significant digit of the typical standard error for that kind, so printed digits carry information
# (Clymo 2014; Gelman's "digit soup"). Every table and macro formats through fmt() below.
precision <- c(
  scale = 3L, # effects, standard errors, and means on the 0-1 answer scale; SEs are about 0.005
  points = 1L, # changes in percentage points, and effects expressed as points of a 0-100 scale
  percent = 0L, # model shares and survey-wide shares, whose intervals span several points
  respondents = 1L, # shares of respondents printed beside a change in percentage points
  correlation = 2L,
  ideal = 2L, # measurement-model parameters in ideal-point standard deviations
  statistic = 1L, # z and chi-squared statistics
  p = 3L,
  rhat = 2L
)
hundredths <- c("points", "percent", "respondents")

fmt <- function(x, kind, signed = FALSE) {
  digits <- precision[[kind]]
  if (anyNA(x)) stop("Missing value passed to fmt()")
  rounded <- round(if (kind %in% hundredths) 100 * x else x, digits)
  if (kind == "p" && any(rounded == 0)) stop("A p-value rounds to zero; report it as an inequality")
  rounded[rounded == 0] <- 0
  out <- sprintf(paste0("%", if (signed) "+" else "", ".", digits, "f"), rounded)
  out[rounded == 0] <- sprintf(paste0("%.", digits, "f"), 0)
  if (kind %in% c("percent", "respondents")) out <- paste0(out, "\\%")
  if (signed) paste0("$", out, "$") else out
}

count <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)

# A p-value written with its relation, so that one too small for the configured precision prints as a bound.
p_relation <- function(p) {
  floor <- 10^-precision[["p"]]
  if (p < floor) paste0("<", fmt(floor, "p")) else paste0("=", fmt(p, "p"))
}
