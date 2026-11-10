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

test_that("headline estimates exclude the three interviewers and match recorded values", {
  analysis <- analyze_responses(respondents, list(), read.csv(file.path(root, "data", "questionnaire.csv")))
  expect_equal(analysis$respondents, 5166L)
  expect_equal(length(analysis$control), 10364L)
  expect_equal(length(analysis$treated), 10300L)
  expect_equal(analysis$alpha, 0.74628521806252346, tolerance = 1e-12)
  expect_equal(analysis$beta, -0.020751237480000517, tolerance = 1e-12)
  expect_equal(analysis$se, 0.0053202404833435557, tolerance = 1e-12)
  expect_equal(analysis$se_interviewer, 0.0050953325263102161, tolerance = 1e-12)
  expect_equal(analysis$summaries$ate[1:3], c(-0.012841191060277812, -0.020751237480000517, -0.028136735169026483),
    tolerance = 1e-12
  )
  expect_equal(unname(analysis$anomalies), c(34, 94))
  expect_equal(analysis$screen$interviewer[analysis$screen$suspect], suspect_interviewers)
  expect_true(all(analysis$screen$flagged[analysis$screen$suspect]))
  effects <- analysis$effects
  tested <- effects[effects$tested, ]
  expect_setequal(tested$interviewer[order(-abs(tested$z))][1:2], c(37L, 71L))
  expect_lt(attr(effects, "test")[["p"]], 0.01)
  expect_gt(attr(effects, "test_without_suspects")[["p"]], 0.1)
  expect_equal(attr(effects, "permutation")[["statistic"]], attr(effects, "test")[["statistic"]], ignore_attr = TRUE)
  expect_lt(attr(effects, "permutation")[["p"]], 0.05)
  expect_gt(attr(effects, "permutation_without_suspects")[["p"]], 0.1)
  # The text says no reshuffle reached the three, and that the other interviewers outside every reshuffle have
  # diffuse answers that move down under endorsement.
  switch <- analysis$switch
  floor <- 1 / (1 + attr(switch, "permutations"))
  expect_true(all(switch$p[switch$suspect] == floor))
  others <- switch[!switch$suspect & switch$p_family < 0.05, ]
  expect_true(all(others$change < 0))
  expect_lt(max(others$modal), min(switch$modal[switch$suspect]))
  expect_lt(max(analysis$blocks$log10_p), -10)
  expect_equal(analysis$matches$items, 92L)
  expect_equal(sum(analysis$matches$duplicates$pairs), 79L)
})

test_that("clustered standard errors reduce to respondent clustering and grow with shared cluster shocks", {
  responses <- reshape_responses(respondents)
  expect_equal(clustered_ate_se(responses, responses$respondent), clustered_ate_se(responses))
  set.seed(3)
  shock <- stats::rnorm(20)
  toy <- data.frame(
    respondent = 1:400, interviewer = rep(1:20, each = 20),
    endorser = rep(c("a", "b"), 200)
  )
  toy$response <- shock[toy$interviewer] * (toy$endorser == "b") + stats::rnorm(400, sd = 0.1)
  expect_gt(clustered_ate_se(toy, toy$interviewer), 3 * clustered_ate_se(toy))
})

test_that("the screen flags an interviewer who records one answer string and spares the rest", {
  set.seed(11)
  n <- 600L
  answer <- function() sample(1:5, n, replace = TRUE)
  toy <- data.frame(
    respondent = seq_len(n), province = "p", district = rep(c("d1", "d2"), each = n / 2),
    interviewer = rep(1:6, each = n / 6), female = 0L, rural = 0L
  )
  for (column in response_columns()) toy[[column]] <- NA_integer_
  for (policy in policies) toy[[paste(policy, "a", sep = ".")]] <- answer()
  copied <- toy$interviewer == 2L
  toy[copied, paste(policies, "a", sep = ".")] <- matrix(c(5L, 1L, 1L, 5L), sum(copied), 4L, byrow = TRUE)
  for (column in response_columns()) toy[[column]] <- factor(toy[[column]], levels = 1:5)
  screen <- interviewer_screen(toy, simulations = 200L)
  expect_equal(screen$interviewer[screen$p < 0.01], 2L)
  expect_equal(screen$pattern[screen$interviewer == 2L], "4004")
})

test_that("within-respondent moments separate shared reactions from independent noise", {
  make <- function(values, codes) {
    responses <- expand.grid(policy = policies, respondent = seq_len(nrow(values)), stringsAsFactors = FALSE)
    responses$response <- as.vector(t(values))
    responses$endorser <- codes[responses$respondent]
    responses$interviewer <- responses$respondent
    responses[, c("respondent", "interviewer", "policy", "endorser", "response")]
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
  expect_equal(distribution$bins$control[distribution$bins$bin == "16/16"], distribution$full$control[17L])
})

test_that("simulated opposition alone moves mass through the midpoint without emptying it", {
  worlds <- reaction_worlds(reshape_responses(respondents), simulations = 100L)
  one_sided <- worlds[worlds$world == "Opposition only", ]
  expect_true(all(abs(one_sided$midpoint) < 0.01))
  expect_true(all(diff(one_sided$oppose) < 0))
  both <- worlds[worlds$world == "Both directions", ]
  expect_lt(both$midpoint[3], both$midpoint[1])
  expect_lt(worlds$midpoint_lower[worlds$world == "Observed"], 0)
  expect_gt(worlds$midpoint_upper[worlds$world == "Observed"], 0)
})

toy_survey <- function(seed, effect = rep(0, 6)) {
  set.seed(seed)
  n <- 600L
  toy <- data.frame(
    respondent = seq_len(n), province = "p", district = rep(c("d1", "d2"), each = n / 2),
    interviewer = rep(1:6, each = n / 6), female = 0L, rural = 0L
  )
  for (column in response_columns()) toy[[column]] <- NA_integer_
  endorsed <- as.logical(stats::ave(seq_len(n), toy$interviewer, FUN = function(rows) sample(seq_along(rows)) %% 2))
  values <- matrix(sample(1:5, 4L * n, replace = TRUE), n, 4L)
  values[] <- pmin(5L, pmax(1L, values + round(effect[toy$interviewer] * endorsed)))
  list(toy = toy, endorsed = endorsed, values = values)
}

fill_survey <- function(survey) {
  toy <- survey$toy
  for (index in seq_along(policies)) {
    toy[[paste(policies[index], "a", sep = ".")]][!survey$endorsed] <- survey$values[!survey$endorsed, index]
    for (code in names(endorsers)) {
      rows <- survey$endorsed & names(endorsers)[seq_along(survey$endorsed) %% length(endorsers) + 1L] == code
      toy[[paste(policies[index], code, sep = ".")]][rows] <- survey$values[rows, index]
    }
  }
  for (column in response_columns()) toy[[column]] <- factor(toy[[column]], levels = 1:5)
  toy
}

test_that("the version-switch test singles out an interviewer whose block moves with the version", {
  survey <- toy_survey(5)
  copied <- survey$toy$interviewer == 2L
  block <- function(rows, pattern) matrix(pattern, sum(rows), 4L, byrow = TRUE)
  survey$values[copied & !survey$endorsed, ] <- block(copied & !survey$endorsed, c(5L, 1L, 1L, 5L))
  survey$values[copied & survey$endorsed, ] <- block(copied & survey$endorsed, c(5L, 2L, 1L, 5L))
  switch <- arm_switch_test(fill_survey(survey), permutations = 499L)
  expect_equal(switch$interviewer[switch$p_family < 0.05], 2L)
})

test_that("the reshuffled Q rejects an interviewer-specific effect and not a common one", {
  common <- reshape_responses(fill_survey(toy_survey(6, effect = rep(-1, 6))))
  planted <- reshape_responses(fill_survey(toy_survey(6, effect = c(-1, -1, 2, -1, -1, -1))))
  expect_gt(heterogeneity_permutation(common, 1:6, permutations = 499L)[["p"]], 0.05)
  expect_lt(heterogeneity_permutation(planted, 1:6, permutations = 499L)[["p"]], 0.01)
})
