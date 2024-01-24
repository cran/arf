## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----arf----------------------------------------------------------------------
# Load libraries
library(arf)
library(data.table)
library(ggplot2)

# Set seed
set.seed(123, "L'Ecuyer-CMRG")

# Train ARF
arf_iris <- adversarial_rf(iris)

## ----arf2, fig.height=5, fig.width=5------------------------------------------
# Train ARF with no printouts
arf_iris <- adversarial_rf(iris, verbose = FALSE)

# Plot accuracy against iterations (model converges when accuracy <= 0.5)
tmp <- data.frame('Accuracy' = arf_iris$acc, 
                  'Iteration' = seq_len(length(arf_iris$acc)))
ggplot(tmp, aes(Iteration, Accuracy)) + 
  geom_point() + 
  geom_path() +
  geom_hline(yintercept = 0.5, linetype = 'dashed', color = 'red') 

## ----par, eval=FALSE----------------------------------------------------------
#  # Register cores - Unix
#  library(doParallel)
#  registerDoParallel(cores = 2)

## ----par2, eval=FALSE---------------------------------------------------------
#  # Register cores - Windows
#  library(doParallel)
#  cl <- makeCluster(2)
#  registerDoParallel(cl)

## ----arf3---------------------------------------------------------------------
# Rerun ARF, now in parallel and with more trees
arf_iris <- adversarial_rf(iris, num_trees = 100)

## ----forde--------------------------------------------------------------------
# Compute leaf and distribution parameters
params_iris <- forde(arf_iris, iris)

## ----forde_unif---------------------------------------------------------------
# Recompute with uniform density
params_unif <- forde(arf_iris, iris, family = 'unif')

## ----dirichlet----------------------------------------------------------------
# Recompute with additive smoothing
params_alpha <- forde(arf_iris, iris, alpha = 0.1)

# Compare results
head(params_iris$cat)
head(params_alpha$cat)

## ----unity--------------------------------------------------------------------
# Sum probabilities
summary(params_iris$cat[, sum(prob), by = .(f_idx, variable)]$V1)
summary(params_alpha$cat[, sum(prob), by = .(f_idx, variable)]$V1)

## ----forde2-------------------------------------------------------------------
params_iris

## ----lik----------------------------------------------------------------------
# Compute likelihood under truncated normal and uniform distributions
ll <- lik(params_iris, iris, arf = arf_iris)
ll_unif <- lik(params_unif, iris, arf = arf_iris)

# Compare average negative log-likelihood (lower is better)
-mean(ll)
-mean(ll_unif)

## ----iris---------------------------------------------------------------------
head(iris)

## ----iris2, fig.height=5, fig.width=5-----------------------------------------
# Compute likelihoods after marginalizing over Species
iris_without_species <- iris[, -5]
ll_cnt <- lik(params_iris, iris_without_species)

# Compare results
tmp <- data.frame(Total = ll, Partial = ll_cnt)
ggplot(tmp, aes(Total, Partial)) + 
  geom_point() + 
  geom_abline(slope = 1, intercept = 0, color = 'red')

## ----smiley, fig.height=5, fig.width=8----------------------------------------
# Simulate training data
library(mlbench)
x <- mlbench.smiley(1000)
x <- data.frame(x$x, x$classes)
colnames(x) <- c('X', 'Y', 'Class')

# Fit ARF
arf_smiley <- adversarial_rf(x, mtry = 2)

# Estimate parameters
params_smiley <- forde(arf_smiley, x)

# Simulate data
synth <- forge(params_smiley, n_synth = 1000)

# Compare structure
str(x)
str(synth)

# Put it all together
x$Data <- 'Original'
synth$Data <- 'Synthetic'
df <- rbind(x, synth)

# Plot results
ggplot(df, aes(X, Y, color = Class, shape = Class)) + 
  geom_point() + 
  facet_wrap(~ Data)

## ----conditional, fig.height=5, fig.width=5-----------------------------------
# Compute conditional likelihoods
evi <- data.frame(Species = 'setosa')
ll_conditional <- lik(params_iris, query = iris_without_species, evidence = evi)

# Compare NLL across species (shifting to positive range for visualization)
tmp <- iris
tmp$NLL <- -ll_conditional + max(ll_conditional) + 1
ggplot(tmp, aes(Species, NLL, fill = Species)) + 
  geom_boxplot() + 
  scale_y_log10() + 
  ylab('Negative Log-Likelihood') + 
  theme(legend.position = 'none')

## ----cond2--------------------------------------------------------------------
# Data frame of conditioning events
evi <- data.frame(variable = c('Species', 'Petal.Width'),
                  relation = c('==', '>'), 
                  value = c('setosa', 0.3))
evi

## ----cond3--------------------------------------------------------------------
evi <- data.frame(variable = c('Species', 'Petal.Width', 'Petal.Width'),
                  relation = c('==', '>', '<='), 
                  value = c('setosa', 0.3, 0.5))
evi

## ----cond4--------------------------------------------------------------------
# Drawing random weights
evi <- data.frame(f_idx = params_iris$forest$f_idx,
                  wt = rexp(nrow(params_iris$forest)))
evi$wt <- evi$wt / sum(evi$wt)
head(evi)

## ----smiley2, fig.height=5, fig.width=8---------------------------------------
# Simulate class-conditional data for smiley example
evi <- data.frame(Class = 4)
synth2 <- forge(params_smiley, n_synth = 250, evidence = evi)

# Put it all together
synth2$Data <- 'Synthetic'
df <- rbind(x, synth2)

# Plot results
ggplot(df, aes(X, Y, color = Class, shape = Class)) + 
  geom_point() + 
  facet_wrap(~ Data)

## ----cond6--------------------------------------------------------------------
expct(params_smiley, evidence = evi)

