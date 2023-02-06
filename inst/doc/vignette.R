## ---- include = FALSE---------------------------------------------------------
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
arf <- adversarial_rf(iris)

## ----arf2, fig.height=5, fig.width=5------------------------------------------
# Train ARF with no printouts
arf <- adversarial_rf(iris, verbose = FALSE)

# Plot accuracy against iterations (model converges when accuracy <= 0.5)
tmp <- data.frame('acc' = arf$acc, 'iter' = seq_len(length(arf$acc)))
ggplot(tmp, aes(iter, acc)) + 
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
arf <- adversarial_rf(iris, num_trees = 100)

## ----forde--------------------------------------------------------------------
# Compute leaf and distribution parameters
params <- forde(arf, iris)

## ----forde_unif---------------------------------------------------------------
# Recompute with uniform density
params_unif <- forde(arf, iris, family = 'unif')

## ----dirichlet----------------------------------------------------------------
# Recompute with additive smoothing
params_alpha <- forde(arf, iris, alpha = 0.1)

# Compare results
head(params$cat)
head(params_alpha$cat)

## ----unity--------------------------------------------------------------------
# Sum probabilities
summary(params$cat[, sum(prob), by = .(f_idx, variable)]$V1)
summary(params_alpha$cat[, sum(prob), by = .(f_idx, variable)]$V1)

## ----forde2-------------------------------------------------------------------
params

## ----lik----------------------------------------------------------------------
# Compute likelihood under truncated normal and uniform distributions
ll <- lik(arf, params, iris)
ll_unif <- lik(arf, params_unif, iris)

# Compare average negative log-likelihood (lower is better)
-mean(ll)
-mean(ll_unif)

## ----lik2---------------------------------------------------------------------
# Compute likelihood in batches of 50
ll_50 <- lik(arf, params, iris, batch = 50)

# Identical results?
identical(ll, ll_50)

## ----smiley, fig.height=5, fig.width=8----------------------------------------
# Simulate training data
library(mlbench)
x <- mlbench.smiley(1000)
x <- data.frame(x$x, x$classes)
colnames(x) <- c('X', 'Y', 'Class')

# Fit ARF
arf <- adversarial_rf(x, mtry = 2)

# Estimate parameters
params <- forde(arf, x)

# Simulate data
synth <- forge(params, n_synth = 1000)

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

## ----price, fig.height=5, fig.width=5-----------------------------------------
# Check data
head(diamonds)

# View the distribution
hist(diamonds$price)

# How many unique prices?
length(unique(diamonds$price))

## ----price2, fig.height=5, fig.width=5----------------------------------------
# Re-class 
diamonds$price <- as.numeric(diamonds$price)

# Take a random subsample of size 2000
s_idx <- sample(1:nrow(diamonds), 2000)

# Train ARF
arf <- adversarial_rf(diamonds[s_idx, ])

# Estimate parameters
params <- forde(arf, diamonds[s_idx, ])

# Check distributional families
params$meta

# Forge data, check histogram
synth <- forge(params, n_synth = 1000)
hist(synth$price)

## ----lb, fig.height=5, fig.width=5--------------------------------------------
# Set price minimum to empirical lower bound
params$cnt[variable == 'price', min := min(diamonds$price)]

# Re-forge, check histogram
synth <- forge(params, n_synth = 1000)
hist(synth$price)

## ----lprice, fig.height=5, fig.width=5----------------------------------------
# Transform price variable
tmp <- as.data.table(diamonds[s_idx, ])
tmp[, price := log(price)]

# Retrain ARF
arf <- adversarial_rf(tmp)

# Estimate parameters
params <- forde(arf, tmp)

# Forge, check histogram
synth <- forge(params, n_synth = 1000)
hist(exp(synth$price))

