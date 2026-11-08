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
  methods <- analysis$methods
  write_table(
    "method_comparison", c("Restriction", "Assumption", "Implied fraction"),
    cbind(methods$restriction, methods$assumption, percent(methods$fraction, 1)), "lll"
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
}

paper_numbers <- function(analysis) {
  primary <- analysis$posteriors$primary
  average <- primary[primary$group == "Average", ]
  afghan <- primary[primary$group == "Afghan Taliban", ]
  bounds <- analysis$bounds
  count <- function(value) format(value, big.mark = ",", scientific = FALSE, trim = TRUE)
  range <- function(posterior) paste0(percent(posterior$p_lo), "--", percent(posterior$p_hi))
  values <- c(
    SampleSize = count(analysis$respondents), ControlCount = count(length(analysis$control)),
    TreatmentCount = count(length(analysis$treated)), BaselineMean = number(analysis$alpha),
    PooledATE = number(analysis$beta), PooledATEPrecise = number(analysis$beta, 5), PooledSE = number(analysis$se, 4),
    TreatmentMean = number(mean(analysis$treated)), PrimaryShare = percent(average$p_mean),
    PrimaryInterval = range(average),
    MaxMeanReaction = number(max(abs(primary$mu_mean[primary$group != "Average"])), 2),
    AfghanShare = percent(afghan$p_mean), AfghanInterval = range(afghan),
    AfghanNegativeProbability = number(afghan$p_mu_neg, 2),
    MaxRhat = number(max(average$rhat_max_lambda, average$rhat_max_omega2)), MinESS = number(average$ess_min_lambda, 0),
    SmallSymmetricShare = percent(analysis$methods$fraction[7]), ExtremeShare = percent(analysis$methods$fraction[1]),
    CeilingShare = percent(tail(analysis$control_distribution, 1)),
    DiscreteOneShare = percent(analysis$discrete$fraction[1]),
    LowerBound = percent(bounds[["lower"]]),
    UpperBound = percent(bounds[["upper"]]), MixedCount = as.character(analysis$anomalies[["mixed"]]),
    RepeatedCount = as.character(analysis$anomalies[["repeated"]]), AnomalyCount = as.character(analysis$excluded),
    CleanATE = number(analysis$clean_ate, 5)
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
    MidpointControl = percent(bin("0.5")$control, 1), MidpointEndorsed = percent(bin("0.5")$treated, 1),
    MidpointChange = change(bin("0.5")), BelowMidpointChange = change(bin("0.4375")),
    AboveMidpointChange = change(bin("0.5625")), NextAboveChange = change(bin("0.625")),
    LowerTailChange = change(bin("$\\leq 0.375$")),
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
    TwoSidedMidpointMax = signed(min(two_sided$midpoint)),
    TwoSidedTopFavour = percent(max(two_sided$favour)), TwoSidedTopOppose = percent(max(two_sided$oppose)),
    TwoSidedAboveMax = signed(max(two_sided$above))
  )
  # Sharp lower bounds on the shares whose mean answer fell or rose, from the two marginals alone.
  full <- analysis$person_means$full
  gap <- cumsum(full$treated) - cumsum(full$control)
  values <- c(values,
    MeanFallBound = percent(max(0, gap), 1), MeanRiseBound = percent(max(0, -gap), 1)
  )
  values <- c(values,
    MidpointDrop = number(-100 * analysis$person_means$bins$difference[analysis$person_means$bins$bin == "0.5"], 1)
  )
  for (spec in c("covariates", "hierarchical")) {
    posterior <- analysis$posteriors[[spec]]
    label <- if (spec == "covariates") "CovariateShare" else "HierarchicalShare"
    values[label] <- percent(posterior$p_mean[posterior$group == "Average"])
  }
  values
}

write_readme_results <- function(analysis, values) {
  plain <- gsub("\\\\%", "%", values)
  value <- function(name) plain[[name]]
  lines <- c(
    paste0(
      "In the unweighted analysis sample of ", value("SampleSize"), " respondents, endorsements lower stated ",
      "policy support by ", value("PooledATE"), " on the 0–1 scale (respondent-clustered SE ", value("PooledSE"),
      "). That small average is the net of reactions in both directions, not the absence of reactions."
    ), "",
    paste0(
      "- **Answers move together differently.** Under endorsement, the correlation between answers rises for five ",
      "of six policy pairs (by ", value("WeakPairChangeMin"), " to ", value("WeakPairChangeMax"),
      ") and falls for FCR and Durand (", value("StrongPairControl"), " to ", value("StrongPairEndorsed"), ")."
    ),
    paste0(
      "- **The midpoint empties in both directions.** Respondents whose four answers average the midpoint fall ",
      "from ", value("MidpointControl"), " to ", value("MidpointEndorsed"), ", with gains one step below and one ",
      "step above."
    ),
    paste0(
      "- **Opposition alone cannot do this.** Endorsed arms simulated from control respondents, tuned to the ",
      "observed average effect, leave the midpoint share unchanged when reactions run only against the endorser."
    ), "",
    paste0(
      "How many respondents react favorably is not identified. The standard measurement model gives ",
      value("PrimaryShare"), " (95% credible interval ", gsub("--", "–", value("PrimaryInterval")), "), but it ",
      "treats a respondent's reactions as independent across questions, which the correlation changes contradict."
    )
  )
  readme <- readLines("README.md", warn = FALSE)
  start <- which(readme == "<!-- BEGIN GENERATED RESULTS -->")
  end <- which(readme == "<!-- END GENERATED RESULTS -->")
  stopifnot(length(start) == 1L, length(end) == 1L, start < end)
  writeLines(c(readme[seq_len(start)], "", lines, "", readme[end:length(readme)]), "README.md")
}

write_paper <- function(analysis) {
  dir.create("tabs", showWarnings = FALSE)
  write_tables(analysis)
  values <- paper_numbers(analysis)
  writeLines(paste0("\\newcommand{\\", names(values), "}{", values, "}"), "tabs/numbers.tex")
  write_readme_results(analysis, values)
  cat("Pooled ATE:", number(analysis$beta, 5), "; respondent-clustered SE:", number(analysis$se, 5), "\n")
  print(analysis$posteriors$primary, row.names = FALSE)
  cat("Assignment anomalies:", analysis$excluded, "; ATE after exclusion:", number(analysis$clean_ate, 5), "\n")
}
