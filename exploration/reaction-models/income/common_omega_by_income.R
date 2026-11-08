for (m in c("data", "analysis", "model")) source(file.path("R", paste0(m, ".R")))
fits <- readRDS("data/endorse_fits_covariates.rds")
d <- read_responses()
design <- fits[[1]]$model.matrix.indiv
lambda <- do.call(rbind, lapply(fits, function(f) as.matrix(f$lambda)))
omega2 <- do.call(rbind, lapply(fits, function(f) as.matrix(f$omega2)))
draws <- seq(1, nrow(lambda), length.out = 1500)
rows <- list()
for (g in seq_along(endorsers)) {
  cols <- (g - 1L) * ncol(design) + seq_len(ncol(design))
  reac <- design %*% t(lambda[draws, cols])
  for (inc in 1:3) {
    idx <- d$inc == inc
    p <- colMeans(pnorm(sweep(reac[idx, ], 2, sqrt(omega2[draws, g]), "/")))
    rows[[length(rows) + 1]] <- data.frame(endorser = endorsers[[g]], inc,
      p = mean(p), lo = quantile(p, .025), hi = quantile(p, .975))
  }
}
out <- do.call(rbind, rows); out[, 3:5] <- round(out[, 3:5], 3)
print(out, row.names = FALSE)
print(colnames(design))
