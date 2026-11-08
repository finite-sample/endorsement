# Linear mixed-model version of the shared-affect decomposition, on the 0-1 response scale.
# Person variance and residual variance are each allowed to differ between arms; the
# excess under treatment is omega_eta^2 (shared affect) and omega_u^2 (independent spread).
# Fast cross-check of the Stan model; ignores item slopes and the ceiling, so approximate.
suppressPackageStartupMessages(library(nlme))
for (m in c("data", "analysis")) source(file.path("R", paste0(m, ".R")))
d <- read_responses()
r <- reshape_responses(d)
r$inc <- d$inc[r$respondent]
r$trow <- r$endorser != "a"
arm <- tapply(r$trow, r$respondent, mean)
r <- r[arm[as.character(r$respondent)] %in% c(0, 1), ]
r$dT <- as.numeric(arm[as.character(r$respondent)] == 1)
r$dC <- 1 - r$dT
r$endorser <- factor(r$endorser, levels = c("a", names(endorsers)), labels = c("control", endorsers))
r$respondent <- factor(r$respondent)

fit_arm_model <- function(s) {
  lme(response ~ policy + endorser, data = s,
    random = list(respondent = pdDiag(~ 0 + dC + dT)),
    weights = varIdent(form = ~ 1 | trow), method = "REML", control = lmeControl(opt = "optim"))
}
report <- function(fit, label) {
  vc <- as.numeric(VarCorr(fit)[c("dC", "dT"), "Variance"])
  sig <- fit$sigma^2
  ratio <- coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  res <- sig * ratio[c("FALSE", "TRUE")]^2
  lam <- fixef(fit)[grep("endorser", names(fixef(fit)))]
  omega_eta2 <- vc[2] - vc[1]
  omega_u2 <- res[2] - res[1]
  cat(sprintf("\n== %s (n respondents = %d)\n", label, nlevels(droplevels(fit$data$respondent))))
  cat(sprintf("person variance   control %.4f  treated %.4f  excess (omega_eta^2) %+.4f\n", vc[1], vc[2], omega_eta2))
  cat(sprintf("residual variance control %.4f  treated %.4f  excess (omega_u^2)   %+.4f\n", res[1], res[2], omega_u2))
  w_eta <- sqrt(max(omega_eta2, 0)); w_all <- sqrt(max(omega_eta2, 0) + max(omega_u2, 0))
  out <- data.frame(endorser = sub("endorser", "", names(lam)), lambda = round(lam, 4),
    p_conservative = round(pnorm(lam / w_eta), 3), p_liberal = round(pnorm(lam / w_all), 3))
  print(out, row.names = FALSE)
  invisible(c(omega_eta2 = omega_eta2, omega_u2 = omega_u2))
}
fit <- fit_arm_model(r)
report(fit, "all respondents")
print(intervals(fit, which = "var-cov"))
for (g in 1:3) report(fit_arm_model(r[r$inc == g, ]), paste("income group", g))
