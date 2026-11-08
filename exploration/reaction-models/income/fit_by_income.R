# Separate primary-spec fits by income group so omega can differ across groups.
args <- commandArgs(trailingOnly = TRUE)
inc <- as.integer(args[1]); settings <- as.integer(args[2:4]); out <- args[5]
for (m in c("data", "analysis", "model")) source(file.path("R", paste0(m, ".R")))
d <- read_responses()
d <- d[d$inc == inc, ]
t0 <- Sys.time()
fits <- fit_model(d, "primary", settings[1], settings[2], settings[3])
saveRDS(fits, out)
s <- summarize_fits(fits, "primary")
s$inc <- inc
write.csv(s, sub("\\.rds$", ".csv", out), row.names = FALSE)
print(s, row.names = FALSE)
message("elapsed: ", format(Sys.time() - t0))
