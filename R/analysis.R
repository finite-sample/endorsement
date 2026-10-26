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
  groups <- lapply(setNames(names(endorsers), unname(endorsers)), function(code) {
    responses$response[responses$endorser == code]
  })
  cells <- do.call(rbind, lapply(policies, function(policy) {
    baseline <- responses$response[responses$policy == policy & responses$endorser == "a"]
    do.call(rbind, lapply(names(endorsers), function(code) {
      endorsed <- responses$response[responses$policy == policy & responses$endorser == code]
      data.frame(policy,
        endorser = endorsers[[code]],
        alpha = mean(baseline), beta = mean(endorsed) - mean(baseline),
        variance_observed = var(endorsed), variance_predicted = mean(endorsed) * (1 - mean(endorsed))
      )
    }))
  }))
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
  group_effects <- do.call(rbind, lapply(names(groups), function(group) {
    effect <- mean(groups[[group]]) - alpha
    data.frame(
      endorser = group, ate = effect, binary = binary_switching(alpha, effect),
      symmetric = symmetric_effects(effect, 0.05), normal = normal_heterogeneity(effect, 0.05),
      extreme = extreme_opponent(effect)
    )
  }))
  bounds <- do.call(rbind, lapply(c(groups, list("All groups" = treated)), function(values) {
    pairing_bounds(category_distribution(values), control_distribution)
  }))
  anomalies <- assignment_anomalies(respondents)
  excluded <- which(anomalies$mixed | anomalies$repeated)
  restricted <- responses[!responses$respondent %in% excluded, ]
  clean_ate <- mean(restricted$response[restricted$endorser != "a"]) -
    mean(restricted$response[restricted$endorser == "a"])
  list(
    respondents = nrow(respondents), responses = responses, control = control, treated = treated,
    alpha = alpha, beta = beta, se = clustered_ate_se(responses), control_distribution = control_distribution,
    cells = cells, discrete = discrete, methods = methods, group_effects = group_effects, bounds = bounds,
    variance_ratio = mean(cells$variance_observed) / mean(cells$variance_predicted),
    anomalies = colSums(anomalies), excluded = length(excluded), clean_ate = clean_ate, posteriors = posteriors
  )
}
