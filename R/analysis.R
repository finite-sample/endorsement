feasible_share <- function(value) ifelse(value >= 0 & value <= 1, value, NA_real_)
binary_switching <- function(alpha, beta) feasible_share(alpha + beta)
symmetric_effects <- function(beta, magnitude) {
  if (magnitude <= 0) stop("The symmetric effect magnitude must be positive")
  feasible_share(beta / (2 * magnitude) + 0.5)
}
extreme_opponent <- function(beta) feasible_share(1 + beta)
normal_heterogeneity <- function(beta, sigma) stats::pnorm(beta / sigma)

category_distribution <- function(response) {
  if (!length(response) || anyNA(response) || !all(response %in% response_categories)) {
    stop("Responses must be observed values on the five-point scale")
  }
  tabulate(match(response, response_categories), nbins = length(response_categories)) / length(response)
}

discrete_scale <- function(beta, control_distribution, steps) {
  shift <- steps * 0.25
  up <- sum(control_distribution * (pmin(response_categories + shift, 1) - response_categories))
  down <- sum(control_distribution * (pmax(response_categories - shift, 0) - response_categories))
  data.frame(steps, up, down, ceiling_loss = shift - up, fraction = feasible_share((beta - down) / (up - down)))
}

pairing_bounds <- function(treatment_distribution, control_distribution) {
  for (distribution in list(treatment_distribution, control_distribution)) {
    invalid <- length(distribution) != 5L || anyNA(distribution) || any(distribution < 0) ||
      abs(sum(distribution) - 1) > 1e-10
    if (invalid) {
      stop("Invalid response marginals")
    }
  }
  joint <- outer(treatment_distribution, control_distribution)
  difference <- outer(response_categories, response_categories, "-")
  treatment_cdf <- cumsum(treatment_distribution)
  control_cdf <- cumsum(control_distribution)
  # Sharp bounds on a strict increase for ordered discrete outcomes.
  lower <- max(0, control_cdf - treatment_cdf)
  upper <- 1 - max(0, treatment_cdf - c(0, head(control_cdf, -1)))
  c(
    lower_pair = sum(joint[difference < 0]), tie = sum(diag(joint)),
    higher_pair = sum(joint[difference > 0]), lower = lower, upper = upper
  )
}

clustered_ate_se <- function(responses) {
  treated <- responses$endorser != "a"
  residual <- responses$response - ave(responses$response, treated, FUN = mean)
  score <- residual * ifelse(treated, 1 / sum(treated), -1 / sum(!treated))
  sqrt(sum(tapply(score, responses$respondent, sum)^2))
}

respondent_matrix <- function(responses) {
  wide <- stats::reshape(responses[, c("respondent", "policy", "response")],
    idvar = "respondent", timevar = "policy", direction = "wide"
  )
  share_endorsed <- tapply(responses$endorser != "a", responses$respondent, mean)[as.character(wide$respondent)]
  # Respondents with both control and endorsed items belong to neither arm.
  keep <- share_endorsed %in% c(0, 1)
  list(
    values = unname(as.matrix(wide[keep, paste0("response.", policies)])),
    treated = unname(share_endorsed[keep] == 1)
  )
}

arm_bootstrap <- function(matrix, statistic, draws, seed) {
  set.seed(seed)
  control <- matrix$values[!matrix$treated, , drop = FALSE]
  treated <- matrix$values[matrix$treated, , drop = FALSE]
  observed <- statistic(treated) - statistic(control)
  replicates <- replicate(draws, {
    statistic(treated[sample(nrow(treated), replace = TRUE), , drop = FALSE]) -
      statistic(control[sample(nrow(control), replace = TRUE), , drop = FALSE])
  })
  replicates <- matrix(replicates, ncol = draws)
  list(
    control = statistic(control), treated = statistic(treated), difference = observed,
    lower = apply(replicates, 1, stats::quantile, 0.025), upper = apply(replicates, 1, stats::quantile, 0.975)
  )
}

# Noise that is independent across questions raises variances but not covariances. A reaction
# carried by the respondent across all four endorsements raises how the answers move together.
within_respondent_moments <- function(responses, draws = 2000L, seed = 1L) {
  pairs <- utils::combn(length(policies), 2)
  correlations <- function(values) {
    r <- stats::cor(values)
    r[cbind(pairs[1, ], pairs[2, ])]
  }
  boot <- arm_bootstrap(respondent_matrix(responses), correlations, draws, seed)
  data.frame(
    first = policies[pairs[1, ]], second = policies[pairs[2, ]],
    control = boot$control, treated = boot$treated, difference = boot$difference,
    lower = boot$lower, upper = boot$upper
  )
}

# Distribution of each respondent's mean answer across the four policies, by arm. Indifference
# predicts no change; heterogeneous reactions in both directions move mass away from the centre.
person_mean_distribution <- function(responses, draws = 2000L, seed = 1L) {
  steps <- length(policies) * 4L
  shares <- function(values) {
    tabulate(round(rowMeans(values) * steps) + 1L, nbins = steps + 1L) / nrow(values)
  }
  matrix <- respondent_matrix(responses)
  boot <- arm_bootstrap(matrix, shares, draws, seed)
  # Coarser bins keep the midpoint and its neighbours separate and pool the sparse tails.
  edges <- c(-Inf, 6, 7, 8, 9, 10, 13, 15, steps) / steps
  bins <- function(values) {
    tabulate(findInterval(rowMeans(values), edges, left.open = TRUE), nbins = length(edges) - 1L) / nrow(values)
  }
  binned <- arm_bootstrap(matrix, bins, draws, seed)
  list(
    full = data.frame(
      mean = seq(0, 1, length.out = steps + 1L), control = boot$control, treated = boot$treated,
      difference = boot$difference, lower = boot$lower, upper = boot$upper
    ),
    bins = data.frame(
      bin = c("$\\leq 0.375$", "0.4375", "0.5", "0.5625", "0.625", "0.6875--0.8125", "0.875--0.9375", "1"),
      control = binned$control, treated = binned$treated,
      difference = binned$difference, lower = binned$lower, upper = binned$upper
    )
  )
}

# Simulated endorsed arms built from the real control respondents. Each world reproduces the observed
# average effect: a reacting respondent moves each answer one step in their direction with probability
# `moved`, and the opposing share is solved so the expected mean change equals the observed one.
# A world with opposition alone shows whether a one-directional reaction can empty the midpoint.
reaction_worlds <- function(responses, draws = 2000L, simulations = 500L, seed = 1L) {
  matrix <- respondent_matrix(responses)
  control <- matrix$values[!matrix$treated, , drop = FALSE]
  n_treated <- sum(matrix$treated)
  step <- 0.25
  statistic <- function(values) {
    sixteenths <- round(rowMeans(values) * 16)
    c(
      spread = stats::var(rowMeans(values)), midpoint = mean(sixteenths == 8),
      below = mean(sixteenths == 7), above = mean(sixteenths == 9)
    )
  }
  observed <- arm_bootstrap(matrix, statistic, draws, seed)
  ate <- mean(matrix$values[matrix$treated, ]) - mean(control)
  worlds <- data.frame(favour = c(0, 0, 0, 0.1, 0.2, 0.3), moved = c(0.25, 0.5, 1, 0.5, 0.5, 0.5))
  worlds$oppose <- (worlds$favour * worlds$moved * step * mean(control < 1) - ate) /
    (worlds$moved * step * mean(control > 0))
  set.seed(seed)
  simulated <- lapply(seq_len(nrow(worlds)), function(index) {
    world <- worlds[index, ]
    replicates <- replicate(simulations, {
      base <- control[sample(nrow(control), n_treated, replace = TRUE), , drop = FALSE]
      draw <- stats::runif(n_treated)
      direction <- ifelse(draw < world$oppose, -1, ifelse(draw < world$oppose + world$favour, 1, 0))
      moves <- direction * (stats::runif(length(base)) < world$moved)
      endorsed <- pmin(pmax(base + step * moves, 0), 1)
      statistic(endorsed) - statistic(control[sample(nrow(control), replace = TRUE), , drop = FALSE])
    })
    rbind(rowMeans(replicates), apply(replicates, 1, stats::quantile, c(0.025, 0.975)))
  })
  summaries <- c(list(rbind(observed$difference, observed$lower, observed$upper)), simulated)
  columns <- lapply(c(spread = "spread", midpoint = "midpoint", below = "below", above = "above"), function(name) {
    t(sapply(summaries, function(summary) summary[, name]))
  })
  result <- data.frame(
    world = c("Observed", rep("Opposition only", 3), rep("Both directions", 3)),
    oppose = c(NA, worlds$oppose), favour = c(NA, worlds$favour), moved = c(NA, worlds$moved)
  )
  for (name in names(columns)) {
    result[[name]] <- columns[[name]][, 1]
    result[[paste0(name, "_lower")]] <- columns[[name]][, 2]
    result[[paste0(name, "_upper")]] <- columns[[name]][, 3]
  }
  result
}

assignment_anomalies <- function(respondents) {
  counts <- sapply(c("a", names(endorsers)), function(code) {
    rowSums(!is.na(respondents[paste(policies, code, sep = ".")]))
  })
  data.frame(
    mixed = counts[, "a"] > 0 & rowSums(counts[, -1]) > 0,
    repeated = rowSums(counts[, -1] > 1) > 0
  )
}

analyze_responses <- function(respondents, posteriors) {
  responses <- reshape_responses(respondents)
  control <- responses$response[responses$endorser == "a"]
  treated <- responses$response[responses$endorser != "a"]
  alpha <- mean(control)
  beta <- mean(treated) - alpha
  control_distribution <- category_distribution(control)
  discrete <- do.call(rbind, lapply(1:3, function(steps) discrete_scale(beta, control_distribution, steps)))
  methods <- data.frame(
    restriction = c(
      "Extreme opponent reaction", "Binary switching", "5-point scale, 2-step shift",
      "5-point scale, 1-step shift", "Normal heterogeneity ($\\sigma = 0.05$)",
      "Symmetric effects ($d = 0.05$)", "Symmetric effects ($d = 0.02$)"
    ),
    assumption = c(
      "$\\delta^- = -1$, $\\delta^+ = 0$", "All supporters $\\to$ 1, opponents $\\to$ 0",
      "Shift $\\pm 2$ categories, truncated", "Shift $\\pm 1$ category, truncated",
      "$\\delta_i \\sim \\mathcal{N}(\\beta, 0.05^2)$", "$\\delta^+ = +0.05$, $\\delta^- = -0.05$",
      "$\\delta^+ = +0.02$, $\\delta^- = -0.02$"
    ),
    fraction = c(
      extreme_opponent(beta), binary_switching(alpha, beta), discrete$fraction[2:1],
      normal_heterogeneity(beta, 0.05), symmetric_effects(beta, 0.05), symmetric_effects(beta, 0.02)
    )
  )
  bounds <- pairing_bounds(category_distribution(treated), control_distribution)
  anomalies <- assignment_anomalies(respondents)
  excluded <- which(anomalies$mixed | anomalies$repeated)
  restricted <- responses[!responses$respondent %in% excluded, ]
  clean_ate <- mean(restricted$response[restricted$endorser != "a"]) -
    mean(restricted$response[restricted$endorser == "a"])
  list(
    respondents = nrow(respondents), responses = responses, control = control, treated = treated,
    alpha = alpha, beta = beta, se = clustered_ate_se(responses), control_distribution = control_distribution,
    discrete = discrete, methods = methods, bounds = bounds,
    anomalies = colSums(anomalies), excluded = length(excluded), clean_ate = clean_ate, posteriors = posteriors,
    moments = within_respondent_moments(responses), person_means = person_mean_distribution(responses),
    reaction_worlds = reaction_worlds(responses)
  )
}
