number <- function(value, digits = 3L) sprintf(paste0("%.", digits, "f"), value)
percent <- function(value, digits = 0L) paste0(number(100 * value, digits), "\\%")
interval <- function(posterior) paste0("[", percent(posterior$p_lo), ", ", percent(posterior$p_hi), "]")

write_table <- function(name, headings, rows, align, breaks = integer()) {
  body <- unlist(lapply(seq_len(nrow(rows)), function(index) {
    c(if (index %in% breaks) "\\midrule", paste0(paste(rows[index, ], collapse = " & "), " \\\\"))
  }))
  writeLines(c(
    paste0("\\begin{tabular}{", align, "}"), "\\toprule",
    paste0(paste(headings, collapse = " & "), " \\\\"), "\\midrule", body,
    "\\bottomrule", "\\end{tabular}"
  ), file.path("tabs", paste0(name, ".tex")))
}

write_tables <- function(analysis) {
  cells <- analysis$cells
  observed <- c(cells$variance_observed, mean(cells$variance_observed))
  predicted <- c(cells$variance_predicted, mean(cells$variance_predicted))
  write_table(
    "variance_test", c("Policy", "Endorser", "Var(T) Obs", "Var(T) Pred", "Ratio"),
    cbind(
      c(cells$policy, "\\textbf{Average}"), c(cells$endorser, ""),
      number(observed), number(predicted), number(observed / predicted, 2)
    ), "llrrr", 17L
  )
  methods <- analysis$methods
  write_table(
    "method_comparison", c("Restriction", "Assumption", "Implied fraction"),
    cbind(methods$restriction, methods$assumption, percent(methods$fraction, 1)), "lll"
  )
  discrete <- analysis$discrete
  write_table(
    "discrete_scale", c("Shift", "Eff.\\ $\\delta^+$", "Eff.\\ $\\delta^-$", "Ceiling Loss", "Implied $p$"),
    cbind(
      c("1 category", "2 categories", "3 categories"), number(discrete$up), number(discrete$down),
      number(discrete$ceiling_loss), percent(discrete$fraction, 1)
    ), "lrrrr"
  )
  effects <- analysis$group_effects
  write_table(
    "empirical_support", c("Endorser", "ATE", "Binary", "Symmetric", "Normal", "Extreme"),
    cbind(
      effects$endorser, sprintf("$%+.4f$", effects$ate), percent(effects$binary, 1),
      percent(effects$symmetric, 1), percent(effects$normal, 1), percent(effects$extreme, 1)
    ), "lrrrrr"
  )
  bounds <- analysis$bounds
  write_table(
    "pairing_bounds", c(
      "Endorser", "$P(T<C)$", "$P(T=C)$", "$P(T>C)$",
      "Min.\\ Positive", "Max.\\ Positive"
    ),
    cbind(rownames(bounds), matrix(percent(bounds), nrow = nrow(bounds))), "lrrrrr", 5L
  )
  moments <- analysis$moments
  write_table(
    "within_correlation", c("Policy pair", "Control", "Endorsed", "Difference", "95\\% interval"),
    cbind(
      paste(moments$first, moments$second, sep = " -- "), number(moments$control, 2), number(moments$treated, 2),
      sprintf("$%+.2f$", moments$difference), sprintf("[$%+.2f$, $%+.2f$]", moments$lower, moments$upper)
    ), "lrrrr"
  )
  bins <- analysis$person_means$bins
  points <- function(value) sprintf("$%+.1f$", 100 * value)
  write_table(
    "person_means", c("Mean answer", "Control", "Endorsed", "Difference (pp)", "95\\% interval"),
    cbind(
      bins$bin, percent(bins$control, 1), percent(bins$treated, 1), points(bins$difference),
      paste0("[", points(bins$lower), ", ", points(bins$upper), "]")
    ), "lrrrr"
  )
  worlds <- analysis$reaction_worlds
  cell <- function(name, scale, digits) {
    form <- function(value) sprintf(paste0("$%+.", digits, "f$"), scale * value)
    bounds <- paste0(form(worlds[[paste0(name, "_lower")]]), ", ", form(worlds[[paste0(name, "_upper")]]))
    paste0(form(worlds[[name]]), " [", bounds, "]")
  }
  share <- function(value) ifelse(is.na(value), "", percent(value))
  write_table(
    "reaction_worlds", c(
      "Endorsed arm", "Oppose", "Favor", "Answers moved", "Spread ($\\times 100$)", "Midpoint (pp)",
      "One below (pp)", "One above (pp)"
    ),
    cbind(
      worlds$world, share(worlds$oppose), share(worlds$favour), share(worlds$moved),
      cell("spread", 100, 2), cell("midpoint", 100, 1), cell("below", 100, 1), cell("above", 100, 1)
    ), "lrrrrrrr", c(2L, 5L)
  )
  primary <- analysis$posteriors$primary
  write_table(
    "endorse_results", c(
      "Endorser", "$P(s > 0)$", "95\\% CrI",
      "\\shortstack[r]{Mean reaction\\\\(ideal-point SDs)}",
      "\\shortstack[r]{Reaction SD\\\\(ideal-point SDs)}"
    ),
    cbind(
      primary$group, percent(primary$p_mean), interval(primary),
      sprintf("$%+.2f$", primary$mu_mean), number(primary$omega_mean, 2)
    ), "lrrrr", 5L
  )
  robustness <- sapply(analysis$posteriors, function(posterior) {
    paste(percent(posterior$p_mean), interval(posterior))
  })
  write_table(
    "endorse_robustness", c("Endorser", unname(specifications)),
    cbind(primary$group, robustness), "lrrr", 5L
  )
}

paper_numbers <- function(analysis) {
  primary <- analysis$posteriors$primary
  average <- primary[primary$group == "Average", ]
  afghan <- primary[primary$group == "Afghan Taliban", ]
  bounds <- analysis$bounds["All groups", ]
  count <- function(value) format(value, big.mark = ",", scientific = FALSE, trim = TRUE)
  range <- function(posterior) paste0(percent(posterior$p_lo), "--", percent(posterior$p_hi))
  values <- c(
    SampleSize = count(analysis$respondents), ControlCount = count(length(analysis$control)),
    TreatmentCount = count(length(analysis$treated)), BaselineMean = number(analysis$alpha),
    PooledATE = number(analysis$beta), PooledATEPrecise = number(analysis$beta, 5), PooledSE = number(analysis$se, 4),
    TreatmentMean = number(mean(analysis$treated)), TreatmentSD = number(sd(analysis$treated), 2),
    BinarySD = number(sqrt(mean(analysis$treated) * (1 - mean(analysis$treated))), 2),
    VarianceRatio = number(analysis$variance_ratio, 2), PrimaryShare = percent(average$p_mean),
    PrimaryInterval = range(average),
    MaxMeanReaction = number(max(abs(primary$mu_mean[primary$group != "Average"])), 2),
    AfghanShare = percent(afghan$p_mean), AfghanInterval = range(afghan),
    AfghanNegativeProbability = number(afghan$p_mu_neg, 2),
    MaxRhat = number(max(average$rhat_max_lambda, average$rhat_max_omega2)), MinESS = number(average$ess_min_lambda, 0),
    BinaryShare = percent(analysis$methods$fraction[2], 1), SymmetricShare = percent(analysis$methods$fraction[6]),
    SmallSymmetricShare = percent(analysis$methods$fraction[7]), ExtremeShare = percent(analysis$methods$fraction[1]),
    NormalShare = percent(analysis$methods$fraction[5]), CeilingShare = percent(tail(analysis$control_distribution, 1)),
    TopTwoShare = percent(sum(tail(analysis$control_distribution, 2)), 1),
    DiscreteOneShare = percent(analysis$discrete$fraction[1]),
    DiscreteTwoShare = percent(analysis$discrete$fraction[2]),
    DiscreteThreeShare = percent(analysis$discrete$fraction[3]), LowerBound = percent(bounds[["lower"]]),
    UpperBound = percent(bounds[["upper"]]), RandomHigher = percent(bounds[["higher_pair"]]),
    RandomEqual = percent(bounds[["tie"]]), RandomLower = percent(bounds[["lower_pair"]]),
    MixedCount = as.character(analysis$anomalies[["mixed"]]),
    RepeatedCount = as.character(analysis$anomalies[["repeated"]]), AnomalyCount = as.character(analysis$excluded),
    CleanATE = number(analysis$clean_ate, 5), AwareLower = percent(max(0, (mean(analysis$treated) - 0.3) / 0.7))
  )
  points <- function(value) sprintf("%+.1f", 100 * value)
  change <- function(row) {
    paste0("$", points(row$difference), "$ [$", points(row$lower), "$, $", points(row$upper), "$]")
  }
  moments <- analysis$moments
  strong <- moments[moments$first == "FCR" & moments$second == "Durand", ]
  weak <- moments[moments$first != "FCR" | moments$second != "Durand", ]
  bins <- analysis$person_means$bins
  bin <- function(label) bins[bins$bin == label, ]
  values <- c(values,
    StrongPairControl = number(strong$control, 2), StrongPairEndorsed = number(strong$treated, 2),
    StrongPairChange = sprintf("$%+.2f$", strong$difference),
    StrongPairInterval = sprintf("[$%+.2f$, $%+.2f$]", strong$lower, strong$upper),
    WeakPairChangeMin = sprintf("%.2f", min(weak$difference)),
    WeakPairChangeMax = sprintf("%.2f", max(weak$difference)),
    WeakPairLowerMin = sprintf("%.2f", min(weak$lower)),
    MidpointControl = percent(bin("0.5")$control, 1), MidpointEndorsed = percent(bin("0.5")$treated, 1),
    MidpointChange = change(bin("0.5")), BelowMidpointChange = change(bin("0.4375")),
    AboveMidpointChange = change(bin("0.5625")), NextAboveChange = change(bin("0.625")),
    LowerTailChange = change(bin("$\\leq 0.375$")), TopChange = change(bin("0.875--0.9375")),
    AllCeilingControl = percent(bin("1")$control, 1), AllCeilingChange = change(bin("1")),
    TopThirdShare = percent(bin("0.875--0.9375")$control + bin("1")$control, 1)
  )
  worlds <- analysis$reaction_worlds
  one_sided <- worlds[worlds$world == "Opposition only", ]
  two_sided <- worlds[worlds$world == "Both directions", ]
  signed <- function(value) sprintf("$%+.1f$", 100 * value)
  values <- c(values,
    OneSidedOpposeMin = percent(min(one_sided$oppose)), OneSidedOpposeMax = percent(max(one_sided$oppose)),
    OneSidedMidpointMin = signed(min(one_sided$midpoint)), OneSidedMidpointMax = signed(max(one_sided$midpoint)),
    OneSidedMidpointLowest = number(100 * abs(min(one_sided$midpoint_lower)), 1),
    TwoSidedMidpointMin = signed(max(two_sided$midpoint)), TwoSidedMidpointMax = signed(min(two_sided$midpoint)),
    TwoSidedTopFavour = percent(max(two_sided$favour)), TwoSidedTopOppose = percent(max(two_sided$oppose)),
    TwoSidedAboveMax = signed(max(two_sided$above))
  )
  # Sharp lower bounds on the shares whose mean answer fell or rose, from the two marginals alone.
  full <- analysis$person_means$full
  gap <- cumsum(full$treated) - cumsum(full$control)
  values <- c(values,
    MeanFallBound = percent(max(0, gap), 1), MeanRiseBound = percent(max(0, -gap), 1)
  )
  for (spec in c("covariates", "hierarchical")) {
    posterior <- analysis$posteriors[[spec]]
    label <- if (spec == "covariates") "CovariateShare" else "HierarchicalShare"
    values[label] <- percent(posterior$p_mean[posterior$group == "Average"])
  }
  values["HierarchicalAfghanShare"] <- percent(analysis$posteriors$hierarchical$p_mean[
    analysis$posteriors$hierarchical$group == "Afghan Taliban"
  ])
  values
}

write_readme_results <- function(analysis, values) {
  plain <- gsub("\\\\%", "%", values)
  value <- function(name) plain[[name]]
  posterior <- analysis$posteriors$primary
  lines <- c(
    paste0(
      "In the unweighted analysis sample of ", value("SampleSize"),
      " respondents, the fitted measurement model implies a **", value("PrimaryShare"),
      " probability of a positive latent reaction** to an endorsement, averaged equally across four militant ",
      "groups (95% credible interval ", gsub("--", "–", value("PrimaryInterval")), ")."
    ), "",
    "| Endorser | Positive latent reaction | 95% credible interval |", "|---|---:|---:|",
    gsub("\\\\%", "%", paste0(
      "| ", posterior$group, " | ", percent(posterior$p_mean), " | ", interval(posterior), " |"
    )), "",
    paste0(
      "The pooled outcome mean is ", value("BaselineMean"), " without an endorsement and ",
      value("TreatmentMean"), " with one, an average difference of ", value("PooledATE"),
      " on the rescaled 0–1 response scale (respondent-clustered SE ", value("PooledSE"), "). ",
      "Bounds calculated from the pooled response marginals run from ", value("LowerBound"), " to ",
      value("UpperBound"), " for the fraction of person–policy responses strictly increased by endorsement. ",
      "These are plug-in bounds, not confidence intervals or bounds on the latent reaction probability."
    )
  )
  readme <- readLines("README.md", warn = FALSE)
  start <- which(readme == "<!-- BEGIN GENERATED RESULTS -->")
  end <- which(readme == "<!-- END GENERATED RESULTS -->")
  stopifnot(length(start) == 1L, length(end) == 1L, start < end)
  writeLines(c(readme[seq_len(start)], "", lines, "", readme[end:length(readme)]), "README.md")
}

sensitivity_data <- function(alpha) {
  ate <- sort(unique(c(seq(-0.1, 0.1, length.out = 200L), -0.05, -0.02, 0, 0.02, 0.05)))
  curves <- list(
    "Binary switching*" = binary_switching(alpha, ate),
    "Extreme opponent reaction" = extreme_opponent(ate),
    "Symmetric effects (d = 0.05)" = symmetric_effects(ate, 0.05),
    "Symmetric effects (d = 0.02)" = symmetric_effects(ate, 0.02)
  )
  do.call(rbind, lapply(names(curves), function(restriction) {
    data.frame(ate, fraction = curves[[restriction]], restriction = factor(restriction, levels = names(curves)))
  }))
}

write_sensitivity <- function(analysis) {
  curves <- sensitivity_data(analysis$alpha)
  plot <- ggplot2::ggplot(curves[!is.na(curves$fraction), ], ggplot2::aes(ate, fraction,
    colour = restriction, linetype = restriction
  )) +
    ggplot2::geom_hline(yintercept = 0.5, colour = "grey70", linetype = "dashed") +
    ggplot2::geom_vline(xintercept = analysis$beta, colour = "grey55", linetype = "dashed") +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::annotate("text",
      x = analysis$beta + 0.003, y = 0.07, hjust = 0, size = 3,
      label = paste0("Observed ATE\n", number(analysis$beta)), colour = "grey35"
    ) +
    ggplot2::scale_colour_manual(values = c("#D45500", "#228833", "#AA8822", "#3377BB")) +
    ggplot2::scale_linetype_manual(values = c("solid", "dotted", "dotdash", "dashed")) +
    ggplot2::scale_x_continuous(limits = c(-0.1, 0.1), breaks = seq(-0.1, 0.1, 0.05)) +
    ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    ggplot2::labs(
      x = "Average treatment effect (0–1 response scale)",
      y = "Implied fraction (definition varies by restriction)", colour = NULL, linetype = NULL
    ) +
    ggplot2::theme_minimal(base_size = 10) +
    ggplot2::theme(
      legend.position = "bottom", legend.text = ggplot2::element_text(size = 8),
      panel.grid.minor = ggplot2::element_blank()
    ) +
    ggplot2::guides(colour = ggplot2::guide_legend(nrow = 2))
  ggplot2::ggsave("figs/sensitivity.pdf", plot, width = 6.5, height = 4.5, device = grDevices::cairo_pdf)
}

write_paper <- function(analysis) {
  dir.create("tabs", showWarnings = FALSE)
  dir.create("figs", showWarnings = FALSE)
  write_tables(analysis)
  values <- paper_numbers(analysis)
  writeLines(paste0("\\newcommand{\\", names(values), "}{", values, "}"), "tabs/numbers.tex")
  write_readme_results(analysis, values)
  write_sensitivity(analysis)
  cat("Pooled ATE:", number(analysis$beta, 5), "; respondent-clustered SE:", number(analysis$se, 5), "\n")
  print(analysis$posteriors$primary, row.names = FALSE)
  cat("Assignment anomalies:", analysis$excluded, "; ATE after exclusion:", number(analysis$clean_ate, 5), "\n")
}
