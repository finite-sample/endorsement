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

clustered_ate_se <- function(responses, cluster = responses$respondent) {
  treated <- responses$endorser != "a"
  residual <- responses$response - ave(responses$response, treated, FUN = mean)
  score <- residual * ifelse(treated, 1 / sum(treated), -1 / sum(!treated))
  sqrt(sum(tapply(score, cluster, sum)^2))
}

respondent_matrix <- function(responses) {
  wide <- stats::reshape(responses[, c("respondent", "interviewer", "policy", "response")],
    idvar = c("respondent", "interviewer"), timevar = "policy", direction = "wide"
  )
  share_endorsed <- tapply(responses$endorser != "a", responses$respondent, mean)[as.character(wide$respondent)]
  # Respondents with both control and endorsed items belong to neither arm.
  keep <- share_endorsed %in% c(0, 1)
  list(
    values = unname(as.matrix(wide[keep, paste0("response.", policies)])),
    treated = unname(share_endorsed[keep] == 1),
    interviewer = wide$interviewer[keep]
  )
}

# Intervals resample interviewers, keeping all of an interviewer's respondents in both arms together,
# because answers recorded by the same interviewer are not independent.
arm_bootstrap <- function(matrix, statistic, draws, seed) {
  set.seed(seed)
  difference <- function(rows) {
    statistic(matrix$values[rows[matrix$treated[rows]], , drop = FALSE]) -
      statistic(matrix$values[rows[!matrix$treated[rows]], , drop = FALSE])
  }
  by_interviewer <- split(seq_len(nrow(matrix$values)), matrix$interviewer)
  replicates <- replicate(draws, {
    difference(unlist(by_interviewer[sample(length(by_interviewer), replace = TRUE)], use.names = FALSE))
  })
  observed <- difference(seq_len(nrow(matrix$values)))
  replicates <- matrix(replicates, ncol = draws, dimnames = list(names(observed), NULL))
  list(
    control = statistic(matrix$values[!matrix$treated, , drop = FALSE]),
    treated = statistic(matrix$values[matrix$treated, , drop = FALSE]),
    difference = observed,
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
      bin = c("$\\leq 6/16$", "7/16", "8/16", "9/16", "10/16", "11/16--13/16", "14/16--15/16", "16/16"),
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

# Each respondent's four answers on the 0-4 scale, in policy order, and their arm.
respondent_answers <- function(respondents) {
  values <- sapply(policies, function(policy) {
    columns <- respondents[paste(policy, c("a", names(endorsers)), sep = ".")]
    as.integer(as.character(columns[cbind(seq_len(nrow(columns)), max.col(!is.na(columns)))])) - 1L
  })
  control <- rowSums(!is.na(respondents[paste(policies, "a", sep = ".")]))
  data.frame(
    interviewer = respondents$interviewer, district = respondents$district,
    pattern = apply(values, 1, paste, collapse = ""), total = rowSums(values),
    arm = ifelse(control == 4L, "control", ifelse(control == 0L, "endorsed", "mixed"))
  )
}

# For each interviewer, how often their single most common answer pattern appears, against how often it
# would appear if their respondents were drawn at random from the respondents in the same districts.
# Top marks on all four policies are the modal pattern in the survey and are left out, so that the
# screen does not flag interviewers for working where support is high.
interviewer_screen <- function(respondents, simulations = 1000L, seed = 1L) {
  answers <- respondent_answers(respondents)
  top_count <- function(patterns) {
    patterns <- patterns[patterns != "4444"]
    if (length(patterns)) max(table(patterns)) else 0L
  }
  pools <- split(answers$pattern, answers$district)
  set.seed(seed)
  rows <- lapply(split(answers, answers$interviewer), function(own) {
    counts <- table(own$district)
    simulated <- replicate(simulations, top_count(unlist(lapply(names(counts), function(district) {
      sample(pools[[district]], counts[[district]], replace = TRUE)
    }))))
    observed <- top_count(own$pattern)
    patterns <- table(own$pattern[own$pattern != "4444"])
    data.frame(
      interviewer = own$interviewer[1], interviews = nrow(own), districts = length(counts),
      control = sum(own$arm == "control"), endorsed = sum(own$arm == "endorsed"),
      pattern = if (length(patterns)) names(patterns)[which.max(patterns)] else "",
      share = observed / nrow(own), expected = mean(simulated) / nrow(own),
      block_control = mean(own$pattern[own$arm == "control"] == "4004"),
      block_endorsed = mean(own$pattern[own$arm == "endorsed"] == "4104"),
      p = mean(simulated >= observed)
    )
  })
  screen <- do.call(rbind, rows)
  screen$suspect <- screen$interviewer %in% suspect_interviewers
  screen$flagged <- screen$p < 0.001
  screen[order(screen$interviewer), ]
}

# Respondents whose answers average the midpoint, by arm, and how many of them the suspect interviewers recorded.
midpoint_counts <- function(respondents) {
  answers <- respondent_answers(respondents)
  answers <- answers[answers$arm != "mixed", ]
  suspect <- answers$interviewer %in% suspect_interviewers
  at_midpoint <- answers$total == 8L
  c(
    control = sum(answers$arm == "control" & at_midpoint),
    control_suspect = sum(answers$arm == "control" & at_midpoint & suspect),
    endorsed = sum(answers$arm == "endorsed" & at_midpoint),
    endorsed_suspect = sum(answers$arm == "endorsed" & at_midpoint & suspect),
    suspect_control = sum(answers$arm == "control" & suspect),
    suspect_endorsed = sum(answers$arm == "endorsed" & suspect),
    suspect_pattern_control = sum(answers$arm == "control" & suspect & answers$pattern == "4004"),
    suspect_pattern_endorsed = sum(answers$arm == "endorsed" & suspect & answers$pattern == "4104")
  )
}

# Each interviewer's own treatment effect, its z-score against the pooled effect, and how far the pooled effect
# moves when that interviewer is left out. If interviewers differed only by sampling noise, the squared z-scores
# would sum to a chi-square with one fewer degree of freedom than interviewers. Interviewers with fewer than
# `minimum` respondents, or with no variation in recorded answers, are left out of the test.
interviewer_effects <- function(responses, minimum = 20L) {
  treated <- responses$endorser != "a"
  pooled <- mean(responses$response[treated]) - mean(responses$response[!treated])
  rows <- lapply(split(responses, responses$interviewer), function(own) {
    own_treated <- own$endorser != "a"
    rest <- responses[responses$interviewer != own$interviewer[1], ]
    rest_treated <- rest$endorser != "a"
    effect <- if (any(own_treated) && any(!own_treated)) {
      mean(own$response[own_treated]) - mean(own$response[!own_treated])
    } else {
      NA_real_
    }
    se <- if (is.na(effect)) NA_real_ else clustered_ate_se(own)
    data.frame(
      interviewer = own$interviewer[1], respondents = length(unique(own$respondent)), effect = effect, se = se,
      z = (effect - pooled) / se,
      shift = mean(rest$response[rest_treated]) - mean(rest$response[!rest_treated]) - pooled
    )
  })
  effects <- do.call(rbind, rows)
  tested <- effects$respondents >= minimum & is.finite(effects$z)
  effects$tested <- tested
  attr(effects, "pooled") <- pooled
  # Cochran's Q: spread of the interviewer effects around their inverse-variance weighted mean.
  test <- function(rows) {
    weight <- 1 / effects$se[rows]^2
    centre <- sum(weight * effects$effect[rows]) / sum(weight)
    statistic <- sum(weight * (effects$effect[rows] - centre)^2)
    c(statistic = statistic, df = sum(rows) - 1L, p = stats::pchisq(statistic, sum(rows) - 1L, lower.tail = FALSE))
  }
  attr(effects, "test") <- test(tested)
  attr(effects, "test_without_suspects") <- test(tested & !effects$interviewer %in% suspect_interviewers)
  attr(effects, "permutation") <- heterogeneity_permutation(responses, effects$interviewer[tested], seed = 1L)
  attr(effects, "permutation_without_suspects") <- heterogeneity_permutation(
    responses, effects$interviewer[tested & !effects$interviewer %in% suspect_interviewers], seed = 2L
  )
  attr(effects, "zero_variance") <- sum(effects$se == 0, na.rm = TRUE)
  effects
}

# Cochran's Q against reshuffles of the randomized version labels. Each respondent's set of four labels moves
# as a unit to another respondent of the same interviewer, after the common effect is removed from endorsed
# answers, so the reshuffled effects differ across interviewers only by chance. Effects and standard errors are
# recomputed exactly as for the observed data. This reference does not rely on the chi-square approximation,
# which understates Q's spread when interviewers have few respondents per version.
heterogeneity_permutation <- function(responses, interviewers, permutations = 19999L, seed = 1L) {
  set.seed(seed)
  cells <- lapply(interviewers, function(id) {
    own <- responses[responses$interviewer == id, ]
    own <- own[order(own$respondent, own$policy), ]
    treated <- matrix(own$endorser != "a", ncol = length(policies), byrow = TRUE)
    answers <- matrix(own$response, ncol = length(policies), byrow = TRUE)
    list(answers = answers, treated = treated)
  })
  draw <- function(cell, labels) {
    n_treated <- sum(cell$treated)
    n_control <- length(cell$treated) - n_treated
    pieces <- lapply(seq_len(ncol(cell$answers)), function(item) {
      list(y = cell$answers[, item], t = matrix(cell$treated[, item][labels], nrow = nrow(labels)))
    })
    treated_sum <- Reduce(`+`, lapply(pieces, function(piece) colSums(piece$y * piece$t)))
    treated_mean <- treated_sum / n_treated
    control_mean <- (sum(cell$answers) - treated_sum) / n_control
    score <- Reduce(`+`, lapply(pieces, function(piece) {
      piece$t * outer(piece$y, treated_mean, "-") / n_treated -
        (1 - piece$t) * outer(piece$y, control_mean, "-") / n_control
    }))
    list(effect = treated_mean - control_mean, se = sqrt(colSums(score^2)))
  }
  q <- function(effect, se) {
    weight <- 1 / se^2
    centre <- colSums(weight * effect) / colSums(weight)
    statistic <- colSums(weight * sweep(effect, 2, centre)^2)
    statistic[colSums(se == 0) > 0] <- Inf
    statistic
  }
  observed <- lapply(cells, function(cell) draw(cell, matrix(seq_len(nrow(cell$answers)))))
  effect <- matrix(sapply(observed, `[[`, "effect"), ncol = 1L)
  se <- matrix(sapply(observed, `[[`, "se"), ncol = 1L)
  statistic <- q(effect, se)
  centre <- sum(effect / se^2) / sum(1 / se^2)
  null <- lapply(cells, function(cell) {
    cell$answers <- cell$answers - centre * cell$treated
    draw(cell, replicate(permutations, sample.int(nrow(cell$answers))))
  })
  reshuffled <- q(t(sapply(null, `[[`, "effect")), t(sapply(null, `[[`, "se")))
  c(statistic = statistic, permutations = permutations, p = (1 + sum(reshuffled >= statistic)) / (1 + permutations))
}

# How often each of the three recorded the 4-0-0-4 block without endorsements and 4-1-0-4 with them, against
# the rate among other interviewers' respondents in the same districts and version. One match is added to each
# comparison group so that no rate is zero. The binomial tail treats respondents as independent.
block_test <- function(respondents) {
  answers <- respondent_answers(respondents)
  others <- answers[!answers$interviewer %in% suspect_interviewers, ]
  cases <- expand.grid(interviewer = suspect_interviewers, arm = c("control", "endorsed"), stringsAsFactors = FALSE)
  cases$pattern <- ifelse(cases$arm == "control", "4004", "4104")
  rows <- lapply(seq_len(nrow(cases)), function(index) {
    case <- cases[index, ]
    own <- answers[answers$interviewer == case$interviewer & answers$arm == case$arm, ]
    comparison <- others$pattern[others$district %in% own$district & others$arm == case$arm]
    recorded <- sum(own$pattern == case$pattern)
    rate <- (sum(comparison == case$pattern) + 1) / (length(comparison) + 1)
    cbind(case,
      recorded = recorded, interviews = nrow(own), comparison = length(comparison),
      matches = sum(comparison == case$pattern),
      log10_p = stats::pbinom(recorded - 1L, nrow(own), rate, lower.tail = FALSE, log.p = TRUE) / log(10)
    )
  })
  tests <- do.call(rbind, rows)
  districts <- unique(answers$district[answers$interviewer %in% suspect_interviewers & answers$arm == "control"])
  nearby <- others$pattern[others$district %in% districts & others$arm == "control"]
  # Five answer options on each of four policies, two versions, every interviewer.
  attr(tests, "searched") <- 5^length(policies) * 2 * length(unique(answers$interviewer))
  attr(tests, "nearby") <- c(interviews = length(nearby), matches = sum(nearby == "4004"))
  attr(tests, "survey_rate") <- mean(others$pattern[others$arm == "control"] == "4004")
  tests
}

# Within each interviewer, how different the answer patterns of the two versions are, against reshuffles of the
# randomized version labels. The distance is the share of interviews that would have to change pattern for the
# two versions to match (total variation distance). No pattern is named in advance, and the reshuffles need no
# assumption about how respondents resemble one another. Interviewers are reshuffled independently, so under
# the null the chance that any of m interviewers reaches p is 1 - (1 - p)^m.
arm_switch_test <- function(respondents, permutations = 9999L, seed = 1L) {
  answers <- respondent_answers(respondents)
  answers <- answers[answers$arm != "mixed", ]
  set.seed(seed)
  rows <- lapply(split(answers, answers$interviewer), function(own) {
    control <- own$arm == "control"
    if (all(control) || !any(control)) {
      return(NULL)
    }
    indicator <- outer(own$pattern, unique(own$pattern), "==") * 1
    distance <- function(labels) {
      in_control <- crossprod(indicator, labels)
      0.5 * colSums(abs(in_control / sum(control) - (colSums(indicator) - in_control) / sum(!control)))
    }
    observed <- distance(matrix(control))
    null <- distance(replicate(permutations, sample(control)))
    data.frame(
      interviewer = own$interviewer[1], interviews = nrow(own), distance = observed,
      lower = stats::quantile(null, 0.025, names = FALSE), middle = stats::median(null),
      upper = stats::quantile(null, 0.975, names = FALSE),
      p = (1 + sum(null >= observed - 1e-9)) / (1 + permutations),
      change = (mean(own$total[!control]) - mean(own$total[control])) / 16,
      modal = max(table(own$pattern)) / nrow(own)
    )
  })
  tests <- do.call(rbind, rows)
  tests$p_family <- 1 - (1 - tests$p)^nrow(tests)
  tests$suspect <- tests$interviewer %in% suspect_interviewers
  attr(tests, "permutations") <- permutations
  tests
}

# For each respondent, the highest share of jointly answered questions on which another respondent in the same
# sampling point gave the identical answer (Kuriakose and Robbins's percent match, within sampling points).
# A pair counts only when both answered at least half the questions.
village_matches <- function(questionnaire) {
  items <- setdiff(names(questionnaire), c("respondent", "point", "interviewer"))
  indicators <- do.call(cbind, lapply(items, function(item) {
    values <- questionnaire[[item]]
    sapply(sort(unique(stats::na.omit(values))), function(level) as.numeric(!is.na(values) & values == level))
  }))
  answered <- !is.na(as.matrix(questionnaire[items])) * 1
  best <- rep(NA_real_, nrow(questionnaire))
  duplicates <- data.frame(interviewer = integer(), pairs = integer())
  for (rows in split(seq_len(nrow(questionnaire)), questionnaire$point)) {
    agree <- tcrossprod(indicators[rows, , drop = FALSE])
    both <- tcrossprod(answered[rows, , drop = FALSE])
    share <- ifelse(both >= 0.5 * length(items), agree / pmax(both, 1), NA)
    diag(share) <- NA
    if (length(rows) > 1) best[rows] <- suppressWarnings(apply(share, 1, max, na.rm = TRUE))
    exact <- sum(share[upper.tri(share)] >= 1, na.rm = TRUE)
    if (exact > 0) {
      duplicates <- rbind(duplicates, data.frame(interviewer = questionnaire$interviewer[rows[1]], pairs = exact))
    }
  }
  best[!is.finite(best)] <- NA
  list(
    items = length(items), best = best, median = stats::median(best, na.rm = TRUE),
    at_85 = mean(best >= 0.85, na.rm = TRUE), at_90 = mean(best >= 0.90, na.rm = TRUE),
    twins = mean(best >= 1, na.rm = TRUE),
    duplicates = stats::aggregate(pairs ~ interviewer, duplicates, sum)
  )
}

sample_summary <- function(respondents, draws = 1000L, seed = 1L) {
  responses <- reshape_responses(respondents)
  treated <- responses$endorser != "a"
  matrix <- respondent_matrix(responses)
  bins <- function(values) {
    sixteenths <- round(rowMeans(values) * 16)
    c(below = mean(sixteenths == 7), midpoint = mean(sixteenths == 8), above = mean(sixteenths == 9))
  }
  boot <- arm_bootstrap(matrix, bins, draws, seed)
  correlation_change <- stats::cor(matrix$values[matrix$treated, ]) - stats::cor(matrix$values[!matrix$treated, ])
  data.frame(
    respondents = nrow(respondents), interviewers = length(unique(respondents$interviewer)),
    ate = mean(responses$response[treated]) - mean(responses$response[!treated]),
    se_respondent = clustered_ate_se(responses), se_interviewer = clustered_ate_se(responses, responses$interviewer),
    below = boot$difference[["below"]], below_lower = boot$lower[["below"]], below_upper = boot$upper[["below"]],
    midpoint = boot$difference[["midpoint"]],
    midpoint_lower = boot$lower[["midpoint"]], midpoint_upper = boot$upper[["midpoint"]],
    above = boot$difference[["above"]], above_lower = boot$lower[["above"]], above_upper = boot$upper[["above"]],
    correlations_rising = sum(correlation_change[upper.tri(correlation_change)] > 0)
  )
}

analyze_responses <- function(respondents, posteriors, questionnaire = read.csv("data/questionnaire.csv"),
                              blair = blair_reanalysis(blair_sample())) {
  screen <- interviewer_screen(respondents)
  flagged <- screen$interviewer[screen$flagged]
  matches <- village_matches(questionnaire)
  duplicated <- matches$duplicates$interviewer
  samples <- list(
    full = respondents, clean = respondents[!respondents$interviewer %in% suspect_interviewers, ],
    unflagged = respondents[!respondents$interviewer %in% flagged, ],
    unduplicated = respondents[!respondents$interviewer %in% c(suspect_interviewers, duplicated), ]
  )
  # Interviewer effects compare respondents who saw one version; the few recorded with both belong to neither arm.
  full_responses <- reshape_responses(samples$full)
  mixed <- assignment_anomalies(samples$full)$mixed
  single_version <- full_responses[!full_responses$respondent %in% which(mixed), ]
  summaries <- do.call(rbind, lapply(names(samples), function(name) {
    cbind(sample = name, sample_summary(samples[[name]]))
  }))
  respondents <- samples$clean
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
      "Symmetric effects ($d = 0.05$)", "Symmetric effects ($d = 0.03$)"
    ),
    assumption = c(
      "$\\delta^- = -1$, $\\delta^+ = 0$", "All supporters $\\to$ 1, opponents $\\to$ 0",
      "Shift $\\pm 2$ categories, truncated", "Shift $\\pm 1$ category, truncated",
      "$\\delta_i \\sim \\mathcal{N}(\\beta, 0.05^2)$", "$\\delta^+ = +0.05$, $\\delta^- = -0.05$",
      "$\\delta^+ = +0.03$, $\\delta^- = -0.03$"
    ),
    fraction = c(
      extreme_opponent(beta), binary_switching(alpha, beta), discrete$fraction[2:1],
      normal_heterogeneity(beta, 0.05), symmetric_effects(beta, 0.05), symmetric_effects(beta, 0.03)
    )
  )
  bounds <- pairing_bounds(category_distribution(treated), control_distribution)
  anomalies <- assignment_anomalies(respondents)
  excluded <- which(anomalies$mixed | anomalies$repeated)
  restricted <- responses[!responses$respondent %in% excluded, ]
  clean_ate <- mean(restricted$response[restricted$endorser != "a"]) -
    mean(restricted$response[restricted$endorser == "a"])
  list(
    screen = screen, midpoint_counts = midpoint_counts(samples$full), summaries = summaries, blair = blair,
    blocks = block_test(samples$full), switch = arm_switch_test(samples$full),
    effects = interviewer_effects(single_version), effects_with_mixed = interviewer_effects(full_responses),
    mixed = sum(mixed), matches = matches,
    respondents = nrow(respondents), responses = responses, control = control, treated = treated,
    alpha = alpha, beta = beta, se = clustered_ate_se(responses),
    se_interviewer = clustered_ate_se(responses, responses$interviewer), control_distribution = control_distribution,
    discrete = discrete, methods = methods, bounds = bounds,
    anomalies = colSums(anomalies), excluded = length(excluded), clean_ate = clean_ate, posteriors = posteriors,
    moments = within_respondent_moments(responses), person_means = person_mean_distribution(responses),
    reaction_worlds = reaction_worlds(responses)
  )
}
