interval <- function(posterior) paste0("[", fmt(posterior$p_lo, "percent"), ", ", fmt(posterior$p_hi, "percent"), "]")
points <- function(value) fmt(value, "points", signed = TRUE)

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
  summaries <- analysis$summaries
  change <- function(name) {
    paste0(
      points(summaries[[name]]), " [", points(summaries[[paste0(name, "_lower")]]), ", ",
      points(summaries[[paste0(name, "_upper")]]), "]"
    )
  }
  write_table(
    "interviewer_samples", c(
      "Sample", "Respondents", "Interviewers", "Average effect", "Midpoint (pp)", "One above (pp)",
      "One below (pp)"
    ),
    cbind(
      c(
        "All interviews", "Without interviewers 7, 37, 71", "Without all flagged interviewers",
        "Without 7, 37, 71 and duplicating interviewers"
      ),
      count(summaries$respondents), summaries$interviewers,
      paste0("$", fmt(summaries$ate, "scale"), "$ (", fmt(summaries$se_interviewer, "scale"), ")"),
      change("midpoint"), change("above"), points(summaries$below)
    ), "lrrrrrr"
  )
  effects <- analysis$effects
  top <- effects[effects$tested, ]
  top <- top[order(-abs(top$z)), ][1:8, ]
  write_table(
    "interviewer_effects", c("Interviewer", "Respondents", "Effect (points)", "SE", "$z$", "Shift when left out"),
    cbind(
      top$interviewer, top$respondents, points(top$effect), fmt(top$se, "points"),
      fmt(top$z, "statistic", signed = TRUE), points(top$shift)
    ), "rrrrrr"
  )
  methods <- analysis$methods
  write_table(
    "method_comparison", c("Restriction", "Assumption", "Implied fraction"),
    cbind(methods$restriction, methods$assumption, fmt(methods$fraction, "percent")), "lll"
  )
  moments <- analysis$moments
  write_table(
    "within_correlation", c("Policy pair", "Control", "Endorsed", "Difference", "95\\% interval"),
    cbind(
      paste(moments$first, moments$second, sep = " -- "), fmt(moments$control, "correlation"),
      fmt(moments$treated, "correlation"), fmt(moments$difference, "correlation", signed = TRUE),
      paste0(
        "[", fmt(moments$lower, "correlation", signed = TRUE), ", ",
        fmt(moments$upper, "correlation", signed = TRUE), "]"
      )
    ), "lrrrr"
  )
  bins <- analysis$person_means$bins
  write_table(
    "person_means", c("Mean answer", "Control", "Endorsed", "Difference (pp)", "95\\% interval"),
    cbind(
      bins$bin, fmt(bins$control, "respondents"), fmt(bins$treated, "respondents"), points(bins$difference),
      paste0("[", points(bins$lower), ", ", points(bins$upper), "]")
    ), "lrrrr"
  )
  worlds <- analysis$reaction_worlds
  # Spread is a variance on the 0-1 scale, printed times 100 so that it shares the points precision.
  cell <- function(name) {
    bounds <- paste0(points(worlds[[paste0(name, "_lower")]]), ", ", points(worlds[[paste0(name, "_upper")]]))
    paste0(points(worlds[[name]]), " [", bounds, "]")
  }
  share <- function(value) replace(rep("", length(value)), !is.na(value), fmt(value[!is.na(value)], "percent"))
  write_table(
    "reaction_worlds", c(
      "Endorsed arm", "Oppose", "Favor", "Answers moved", "Spread ($\\times 100$)", "Midpoint (pp)",
      "One below (pp)", "One above (pp)"
    ),
    cbind(
      worlds$world, share(worlds$oppose), share(worlds$favour), share(worlds$moved),
      cell("spread"), cell("midpoint"), cell("below"), cell("above")
    ), "lrrrrrrr", c(2L, 5L)
  )
  rows <- lapply(names(specifications), function(spec) {
    posterior <- analysis$posteriors[[spec]]
    cbind(
      posterior$group, fmt(posterior$p_mean, "percent"), interval(posterior),
      fmt(posterior$mu_mean, "ideal", signed = TRUE), fmt(posterior$omega_mean, "ideal")
    )
  })
  write_table(
    "endorse_results", c(
      "Endorser", "$P(s > 0)$", "95\\% CrI",
      "\\shortstack[r]{Mean reaction\\\\(ideal-point SDs)}",
      "\\shortstack[r]{Reaction SD\\\\(ideal-point SDs)}"
    ),
    rows[[2]], "lrrrr", 5L
  )
}

# Four views of the interviewers: how many interviews each recorded, how often each recorded a single answer
# pattern against how often respondents in the same districts give it, how each interviewer's interviews split
# between the two versions of the questionnaire, and how often each recorded the 4-0-0-4 block and its
# endorsed counterpart.
write_interviewer_figure <- function(screen, path = "figs/interviewers.pdf") {
  ink <- "#2b2b2b"
  muted <- "#8a8984"
  accent <- "#c2410c"
  screen$group <- ifelse(screen$flagged, "Other flagged", "Not flagged")
  screen$group[screen$suspect] <- "Interviewers 7, 37, 71"
  screen$group <- factor(screen$group, c("Not flagged", "Other flagged", "Interviewers 7, 37, 71"))
  colours <- c("Not flagged" = "#c9c8c3", "Other flagged" = muted, "Interviewers 7, 37, 71" = accent)
  screen$rank <- rank(screen$interviews, ties.method = "first")
  screen$control_share <- screen$control / (screen$control + screen$endorsed)
  overall <- sum(screen$control) / sum(screen$control + screen$endorsed)
  funnel <- data.frame(n = seq(5, max(screen$control + screen$endorsed), by = 1))
  funnel$lower <- stats::qbinom(0.025, funnel$n, overall) / funnel$n
  funnel$upper <- stats::qbinom(0.975, funnel$n, overall) / funnel$n
  theme <- ggplot2::theme_minimal(base_size = 9) + ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(), panel.grid.major.x = ggplot2::element_blank(),
    legend.position = "none", plot.title = ggplot2::element_text(colour = ink, size = 9, face = "bold"),
    axis.text = ggplot2::element_text(colour = muted), axis.title = ggplot2::element_text(colour = muted)
  )
  screen$side <- ifelse(screen$interviewer == 7L, 1.6, -0.4)
  label <- function(data) {
    ggplot2::geom_text(
      data = data[data$suspect, ], ggplot2::aes(label = interviewer, hjust = side), colour = accent,
      size = 2.6, vjust = 0.4
    )
  }
  volume <- ggplot2::ggplot(screen, ggplot2::aes(rank, interviews, colour = group)) +
    ggplot2::geom_point(size = 1.6) +
    label(screen) +
    ggplot2::scale_colour_manual(values = colours) +
    ggplot2::labs(
      title = "A. Interviews per interviewer", x = "Interviewers, ordered by interviews", y = "Interviews"
    ) +
    theme
  repeats <- ggplot2::ggplot(screen, ggplot2::aes(100 * expected, 100 * share, colour = group)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, colour = muted, linewidth = 0.3, linetype = "dashed") +
    ggplot2::geom_point(size = 1.6) +
    label(screen) +
    ggplot2::scale_colour_manual(values = colours) +
    ggplot2::coord_cartesian(xlim = c(0, 70), ylim = c(0, 70)) +
    ggplot2::labs(
      title = "B. Most repeated answer pattern", x = "Expected from same districts (% of interviews)",
      y = "Recorded (% of interviews)"
    ) +
    theme
  balance <- ggplot2::ggplot(screen, ggplot2::aes(control + endorsed, 100 * control_share, colour = group)) +
    ggplot2::geom_ribbon(
      data = funnel, ggplot2::aes(n, ymin = 100 * lower, ymax = 100 * upper), inherit.aes = FALSE,
      fill = "#ecebe7"
    ) +
    ggplot2::geom_point(size = 1.6) +
    label(screen) +
    ggplot2::scale_colour_manual(values = colours) +
    ggplot2::labs(
      title = "C. Interviews without endorsements", x = "Interviews in either version", y = "Share without (%)"
    ) +
    theme
  block <- ggplot2::ggplot(screen, ggplot2::aes(100 * block_control, 100 * block_endorsed, colour = group)) +
    ggplot2::geom_point(size = 1.6) +
    label(screen) +
    ggplot2::scale_colour_manual(values = colours) +
    ggplot2::coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
    ggplot2::labs(
      title = "D. The 4-0-0-4 block", x = "Without endorsements: 4-0-0-4 (%)", y = "With endorsements: 4-1-0-4 (%)"
    ) +
    theme
  panels <- list(volume, repeats, balance, block)
  draw <- function() {
    grid::grid.newpage()
    grid::pushViewport(grid::viewport(layout = grid::grid.layout(2, 2)))
    for (index in seq_along(panels)) {
      print(panels[[index]], vp = grid::viewport(layout.pos.row = (index + 1) %/% 2, layout.pos.col = 2 - index %% 2))
    }
  }
  dir.create(dirname(path), showWarnings = FALSE)
  grDevices::pdf(path, width = 7.5, height = 6.2, useDingbats = FALSE)
  draw()
  grDevices::dev.off()
  # A raster copy for the README; the manuscript uses the PDF.
  grDevices::png(sub("\\.pdf$", ".png", path), width = 7.5, height = 6.2, units = "in", res = 150)
  draw()
  grDevices::dev.off()
  invisible(path)
}

paper_numbers <- function(analysis) {
  summaries <- analysis$summaries
  full <- summaries[summaries$sample == "full", ]
  clean <- summaries[summaries$sample == "clean", ]
  unflagged <- summaries[summaries$sample == "unflagged", ]
  screen <- analysis$screen
  suspect <- screen[screen$suspect, ]
  counts <- analysis$midpoint_counts
  primary <- analysis$posteriors$clean
  average <- primary[primary$group == "Average", ]
  afghan <- primary[primary$group == "Afghan Taliban", ]
  full_average <- analysis$posteriors$full[analysis$posteriors$full$group == "Average", ]
  bounds <- analysis$bounds
  range <- function(posterior) paste0(fmt(posterior$p_lo, "percent"), "--", fmt(posterior$p_hi, "percent"))
  signed_interval <- function(row, name) {
    bound <- function(side) points(row[[paste0(name, "_", side)]])
    paste0(points(row[[name]]), " [", bound("lower"), ", ", bound("upper"), "]")
  }
  balance <- stats::binom.test(
    sum(suspect$control), sum(suspect$control + suspect$endorsed),
    sum(screen$control) / sum(screen$control + screen$endorsed)
  )
  values <- c(
    FullSampleSize = count(full$respondents), SampleSize = count(analysis$respondents),
    InterviewerCount = as.character(nrow(screen)), CleanInterviewerCount = as.character(clean$interviewers),
    UnflaggedSampleSize = count(unflagged$respondents),
    UnflaggedInterviewerCount = as.character(unflagged$interviewers),
    FlaggedCount = as.character(sum(screen$flagged)),
    MedianInterviews = as.character(stats::median(screen$interviews)),
    MinInterviews = as.character(min(screen$interviews)), MaxInterviews = as.character(max(screen$interviews)),
    SuspectInterviews = count(sum(suspect$interviews)),
    SuspectShare = fmt(sum(suspect$interviews) / sum(screen$interviews), "respondents"),
    SuspectControl = as.character(counts[["suspect_control"]]),
    SuspectEndorsed = as.character(counts[["suspect_endorsed"]]),
    SuspectPatternControl = as.character(counts[["suspect_pattern_control"]]),
    SuspectPatternEndorsed = as.character(counts[["suspect_pattern_endorsed"]]),
    SuspectBalanceP = fmt(balance$p.value, "p"),
    MidpointCountControl = as.character(counts[["control"]]),
    MidpointCountSuspect = as.character(counts[["control_suspect"]]),
    MidpointCountEndorsed = as.character(counts[["endorsed"]]),
    MidpointCountEndorsedSuspect = as.character(counts[["endorsed_suspect"]]),
    SuspectTopShareMin = fmt(min(suspect$share), "percent"), SuspectTopShareMax = fmt(max(suspect$share), "percent"),
    SuspectExpectedMax = fmt(max(suspect$expected), "percent"),
    FullATE = fmt(full$ate, "scale"),
    FullATEPoints = fmt(-full$ate, "points"), FullSE = fmt(full$se_interviewer, "scale"),
    PooledATE = fmt(clean$ate, "scale"), PooledATEPoints = fmt(-clean$ate, "points"),
    PooledSE = fmt(analysis$se, "scale"), PooledInterviewerSE = fmt(analysis$se_interviewer, "scale"),
    UnflaggedATE = fmt(unflagged$ate, "scale"),
    FullMidpointChange = signed_interval(full, "midpoint"), FullAboveChange = signed_interval(full, "above"),
    CleanMidpointChange = signed_interval(clean, "midpoint"), CleanAboveChange = signed_interval(clean, "above"),
    FullMidpointDrop = fmt(-full$midpoint, "points"),
    FullCorrelationsRising = as.character(full$correlations_rising),
    BaselineMean = fmt(analysis$alpha, "scale"), TreatmentMean = fmt(mean(analysis$treated), "scale"),
    ControlCount = count(length(analysis$control)), TreatmentCount = count(length(analysis$treated)),
    PrimaryShare = fmt(average$p_mean, "percent"), PrimaryInterval = range(average),
    FullPrimaryShare = fmt(full_average$p_mean, "percent"), FullPrimaryInterval = range(full_average),
    MaxMeanReaction = fmt(max(abs(primary$mu_mean[primary$group != "Average"])), "ideal"),
    AfghanShare = fmt(afghan$p_mean, "percent"), AfghanInterval = range(afghan),
    MaxRhat = fmt(max(sapply(analysis$posteriors, function(p) max(p$rhat_max_lambda, p$rhat_max_omega2))), "rhat"),
    MinESS = count(round(min(sapply(analysis$posteriors, function(p) min(p$ess_min_lambda))))),
    SmallSymmetricShare = fmt(analysis$methods$fraction[7], "percent"),
    ExtremeShare = fmt(analysis$methods$fraction[1], "percent"),
    CeilingShare = fmt(tail(analysis$control_distribution, 1), "percent"),
    DiscreteOneShare = fmt(analysis$discrete$fraction[1], "percent"),
    LowerBound = fmt(bounds[["lower"]], "percent"), UpperBound = fmt(bounds[["upper"]], "percent"),
    MixedCount = as.character(analysis$anomalies[["mixed"]]),
    RepeatedCount = as.character(analysis$anomalies[["repeated"]]), AnomalyCount = as.character(analysis$excluded),
    CleanATE = fmt(analysis$clean_ate, "scale")
  )
  change <- function(row) paste0(points(row$difference), " [", points(row$lower), ", ", points(row$upper), "]")
  moments <- analysis$moments
  rising <- moments[moments$difference > 0, ]
  bins <- analysis$person_means$bins
  bin <- function(label) bins[bins$bin == label, ]
  worlds <- analysis$reaction_worlds
  one_sided <- worlds[worlds$world == "Opposition only", ]
  two_sided <- worlds[worlds$world == "Both directions", ]
  effects <- analysis$effects
  tested <- effects[effects$tested, ]
  test <- attr(effects, "test")
  without <- attr(effects, "test_without_suspects")
  z_rank <- rank(-abs(tested$z))[match(suspect_interviewers, tested$interviewer)]
  shift_rank <- rank(-abs(effects$shift))[match(suspect_interviewers, effects$interviewer)]
  matches <- analysis$matches
  unduplicated <- summaries[summaries$sample == "unduplicated", ]
  values <- c(values,
    HeterogeneityStat = fmt(test[["statistic"]], "statistic"), HeterogeneityDF = as.character(test[["df"]]),
    HeterogeneityP = fmt(test[["p"]], "p"), HeterogeneityTested = as.character(nrow(tested)),
    HeterogeneityStatClean = fmt(without[["statistic"]], "statistic"),
    HeterogeneityDFClean = as.character(without[["df"]]),
    HeterogeneityPClean = fmt(without[["p"]], "p"),
    ZeroVarianceCount = as.character(attr(effects, "zero_variance")),
    SuspectZRanks = paste(sort(z_rank), collapse = ", "), SuspectShiftRanks = paste(sort(shift_rank), collapse = ", "),
    SuspectEffectMin = points(min(effects$effect[effects$interviewer %in% suspect_interviewers])),
    SuspectEffectMax = points(max(effects$effect[effects$interviewer %in% suspect_interviewers])),
    QuestionnaireItems = as.character(matches$items),
    MatchMedian = fmt(matches$median, "percent"), MatchEightyFive = fmt(matches$at_85, "percent"),
    MatchNinety = fmt(matches$at_90, "percent"),
    MatchTwins = fmt(matches$twins, "respondents"), DuplicatePairs = as.character(sum(matches$duplicates$pairs)),
    DuplicatingInterviewers = as.character(nrow(matches$duplicates)),
    TopDuplicators = paste(head(matches$duplicates$interviewer[order(-matches$duplicates$pairs)], 3), collapse = ", "),
    TopDuplicatorPairs = paste(head(sort(matches$duplicates$pairs, decreasing = TRUE), 3), collapse = ", "),
    UnduplicatedATE = fmt(unduplicated$ate, "scale"), UnduplicatedSampleSize = count(unduplicated$respondents)
  )
  values <- c(values,
    CorrelationsRising = as.character(nrow(rising)),
    CorrelationsExcludingZero = as.character(sum(moments$lower > 0 | moments$upper < 0)),
    MidpointControl = fmt(bin("8/16")$control, "respondents"),
    MidpointEndorsed = fmt(bin("8/16")$treated, "respondents"),
    MidpointChange = change(bin("8/16")), BelowMidpointChange = change(bin("7/16")),
    AboveMidpointChange = change(bin("9/16")), LowerTailChange = change(bin("$\\leq 6/16$")),
    UpperChange = change(bin("11/16--13/16")),
    OneSidedOpposeMin = fmt(min(one_sided$oppose), "percent"),
    OneSidedOpposeMax = fmt(max(one_sided$oppose), "percent"),
    OneSidedBelowMin = points(min(one_sided$below)),
    OneSidedBelowMax = points(max(one_sided$below)),
    OneSidedMidpointMin = points(min(one_sided$midpoint)),
    OneSidedMidpointMax = points(max(one_sided$midpoint)),
    OneSidedAboveMin = points(min(one_sided$above)),
    OneSidedAboveMax = points(max(one_sided$above)),
    TwoSidedMidpointMin = points(min(two_sided$midpoint)),
    TwoSidedMidpointMax = points(max(two_sided$midpoint)),
    WorldBelowMax = points(max(worlds$below[-1])),
    TwoSidedTopFavour = fmt(max(two_sided$favour), "percent"),
    OtherBlockMax = fmt(max(screen$block_control[!screen$suspect]), "percent"),
    OtherBlockEndorsedMax = fmt(max(screen$block_endorsed[!screen$suspect]), "percent"),
    SuspectBlockMin = fmt(min(suspect$block_control), "percent"),
    SuspectBlockMax = fmt(max(suspect$block_control), "percent"),
    SuspectBlockEndorsedMin = fmt(min(suspect$block_endorsed), "percent"),
    SuspectBlockEndorsedMax = fmt(max(suspect$block_endorsed), "percent")
  )
  values
}

write_readme_results <- function(values) {
  plain <- gsub("\\\\%", "%", gsub("\\$", "", values))
  value <- function(name) plain[[name]]
  lines <- c(
    paste0(
      "Three of the survey's ", value("InterviewerCount"), " interviewers (", value("SuspectShare"), " of interviews) ",
      "recorded one answer pattern in most of their interviews. Their interviews make respondents look warmer ",
      "toward militant groups and produce the apparent movement in both directions that the published data show."
    ), "",
    paste0(
      "- **With all interviews,** endorsements lower policy support by ", value("FullATE"), " on the 0–1 scale ",
      "(interviewer-clustered SE ", value("FullSE"), ")."
    ),
    paste0(
      "- **Without the three interviewers,** the measurement model's share of favorable reactions falls from ",
      value("FullPrimaryShare"), " to ", value("PrimaryShare"), ", the effect is ", value("PooledATE"),
      " (SE ", value("PooledInterviewerSE"), "), and respondents leave the middle of the scale only downward."
    ),
    paste0(
      "- **How many respondents react favorably is not identified by the data.** The model's share rests on its ",
      "assumption that reactions are normally distributed."
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
  write_interviewer_figure(analysis$screen)
  values <- paper_numbers(analysis)
  writeLines(paste0("\\newcommand{\\", names(values), "}{", values, "}"), "tabs/numbers.tex")
  write_readme_results(values)
  print(analysis$summaries, row.names = FALSE)
  print(analysis$posteriors$clean, row.names = FALSE)
}
