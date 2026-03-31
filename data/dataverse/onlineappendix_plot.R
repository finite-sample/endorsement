
## ###################
## Replication Stata Do File
## Blair, Fair, Malhotra, and Shapiro (2012),
## Poverty and Support for Militant Politics: Evidence from Pakistan.
## Forthcoming, American Journal of Political Science.
## Created 24 March 2011
## Modified 3 April 2012
## 
## Contact Graeme Blair, gblair@princeton.edu with questions
##
## This file replicates Online Appendix Figure 1
## ####################

library(foreign)
library(survey)

names <- c("Kashmir Tanzeem", "Afghan Taliban", "al Qaeda", "Secretarian Tanzeem")

## !!
## run replication_dataprep.do in Stata first, the following line
## uses the output file from that set of commands

pakistan <- read.dta("data_for_analysis.dta", convert.factors = F, convert.underscore = T)
pakistan <- subset(pakistan, is.na(low.urprov) ==F)

q600.names <- c("q600a", "q610ai", "q610aii", "q610aiii", "q610aiv", "q610av", "q620ai", "q620aii", "q620aiii", "q620aiv", "q620av", "q600b", "q610bi", "q610bii", "q610biii", "q610biv", "q620bi", "q620bii", "q620biii", "q620biv", "q620bv", "q600c", "q610ci", "q610cii", "q610ciii", "q610civ", "q620ci", "q620cii", "q620ciii", "q620civ", "q620cv", "q600d", "q610di", "q610dii", "q610diii", "q610div", "q620di", "q620dii", "q620diii", "q620div", "q620dv")

mean.urban.low <- mean.urban.high <- mean.urban.middle <- list()
for(var in q600.names) {
  print(paste("name",var))
  temp <- (pakistan[,c(var)] - 5) / -4

  for(i in c("q600a", "q610ai", "q610aii", "q610aiii", "q610aiv", "q610av", "q620ai", "q620aii", "q620aiii", "q620aiv", "q620av"))
    if(i==var) pakistan$curr.weights <- pakistan$group1weight
  for(i in c("q600b", "q610bi", "q610bii", "q610biii", "q610biv", "q620bi", "q620bii", "q620biii", "q620biv", "q620bv"))
    if(i==var) pakistan$curr.weights <- pakistan$group2weight
  for(i in c("q600c", "q610ci", "q610cii", "q610ciii", "q610civ", "q620ci", "q620cii", "q620ciii", "q620civ", "q620cv"))
    if(i==var) pakistan$curr.weights <- pakistan$group3weight
  for(i in c("q600d", "q610di", "q610dii", "q610diii", "q610div", "q620di", "q620dii", "q620diii", "q620div", "q620dv"))
    if(i==var) pakistan$curr.weights <- pakistan$group4weight

  mean.urban.low[[var]] <- svymean(temp[pakistan$low.urprov==1 & is.na(pakistan$curr.weights)==F], design = svydesign(weights = pakistan$curr.weights[pakistan$low.urprov==1 & is.na(pakistan$curr.weights)==F], strata = pakistan$province[pakistan$low.urprov==1 & is.na(pakistan$curr.weights)==F], ids = pakistan$spn[pakistan$low.urprov==1 & is.na(pakistan$curr.weights)==F]), na.rm = T)
  mean.urban.high[[var]] <- svymean(temp[pakistan$low.urprov==0 & is.na(pakistan$curr.weights)==F], design = svydesign(weights = pakistan$curr.weights[pakistan$low.urprov==0 & is.na(pakistan$curr.weights)==F], strata = pakistan$province[pakistan$low.urprov==0 & is.na(pakistan$curr.weights)==F], ids = pakistan$spn[pakistan$low.urprov==0 & is.na(pakistan$curr.weights)==F]), na.rm = T)

}

names <- c("Kashmir Tanzeem", "Afghan Taliban", "Al Qaeda", "Sectarian Tanzeem")

q600.ordered <- c("q610ai", "q610aii", "q610aiii", "q610aiv", "q610av", "q620ai", "q620aii", "q620aiii", "q620aiv", "q620av", "q610bi", "q610bii", "q610biii", "q610biv", "q620bi", "q620bii", "q620biii", "q620biv", "q620bv", "q610ci", "q610cii", "q610ciii", "q610civ", "q620ci", "q620cii", "q620ciii", "q620civ", "q620cv", "q610di", "q610dii", "q610diii", "q610div", "q620di", "q620dii", "q620diii", "q620div", "q620dv")

q600.labels <- c("Justice","Democracy","Protect Muslims","Rid of Apostates", "Free Kashmir","Social Services","Social Awareness","Religious Education","Secular Education", "Fighting Jihad")

pdf(file = "plots.objectivesprovides.pdf", width = 10.5, height = 8)
par(mfcol = c(2, 4) , mar=c(10,1.4,1.4,1.4), oma=c(3,7,3,0))

for(i in 1:4) {

  plot(0,0, xlim = c(0,5.3), ylim = c(0.77,1), type = "n", axes = F, main = paste(names[i]), xlab = "", ylab = "mean agreement")

  if(i==1) {
    end <- 5
  }  else {
    end <- 4
  }
  for(j in 1:end) {
    if(i==1) {
      first <- 0
    } else {
      first <- 10 + (i-2)*9
    }
    var <- q600.ordered[first + j]
    lines(rep(j-.15,2),c(mean.urban.low[[var]][1]-1.96*SE(mean.urban.low[[var]]), mean.urban.low[[var]][1]+1.96*SE(mean.urban.low[[var]])))
    lines(rep(j+.15,2),c(mean.urban.high[[var]][1]-1.96*SE(mean.urban.high[[var]]), mean.urban.high[[var]][1]+1.96*SE(mean.urban.high[[var]])))
    points(c(j - .15, j + .15), c(mean.urban.low[[var]], mean.urban.high[[var]]), pch = c(1,2), cex = 1.3)

  }

  axis(1, at  = 1:5, labels = q600.labels[1:5], las = 2)
  axis(2, at = c(0.8, 0.85, 0.9, 0.95, 1), labels = c(".8", ".85", ".9",".95","1"))

  plot(0,0, xlim = c(0,5.3), ylim = c(0.77,1), type = "n", axes = F, main = paste(names[i]), xlab = "", ylab = "mean agreement")

  if(i==1){
    start<-6
    end<-10
  } else {
    start<-5
    end<-9
  }
  for(j in start:end) {
    if(i==1) first <- 0
    else first <- 10 + (i-2)*9
    var <- q600.ordered[first + j]
    if(i==1) x.point <- j-5
    else x.point <- j - 4
    lines(rep(x.point - .15, 2),c(mean.urban.low[[var]][1]-1.96*SE(mean.urban.low[[var]]), mean.urban.low[[var]][1]+1.96*SE(mean.urban.low[[var]])))
    lines(rep(x.point + .15, 2),c(mean.urban.high[[var]][1]-1.96*SE(mean.urban.high[[var]]), mean.urban.high[[var]][1]+1.96*SE(mean.urban.high[[var]])))
    points(c(x.point-.15,x.point+.15), c(mean.urban.low[[var]][1], mean.urban.high[[var]][1]), pch = c(1,2), cex = 1.3)##pch = c("L", "H"))
  }

  axis(1, at  = 1:5, labels = q600.labels[6:10], las = 2)
  axis(2, at = c(0.8, 0.85, 0.9, 0.95, 1), labels = c(".8", ".85", ".9",".95","1"))
   
}
mtext("Proportion of Responses", outer = TRUE, line = 1.5, at = 0.85, side = 2, cex = .75)
mtext("Proportion of Responses", outer = TRUE, line = 1.5, at = 0.35, side = 2, cex = .75)


mtext("Group Objectives", outer = TRUE, line = 4.5, at = 0.85, side = 2, cex = 1, font = 2)
mtext("Group Service Provision", outer = TRUE, line = 4.5, at = 0.35, side = 2, cex = 1, font = 2)
par(xpd = NA)
legend("bottomright", c("Low Income Respondents", "Middle & High Income Respondents"), pch = c(1,2), bty = "n", horiz = TRUE, cex = 1, xjust = 0.5)
dev.off()


