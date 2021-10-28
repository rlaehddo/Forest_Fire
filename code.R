
library(dplyr)
library(plyr)
library(tidyr)
library(ggplot2)
library(forecast)
library(MASS)
library(glmnet)
library(jtools)
library(ggstance)
library(ggpubr)
library(gridExtra)
library(grid)

fire <- read.csv(file = "~/Desktop/school/stat151A/project/forestfires.csv", header = TRUE)
fire

# Unique values from `month`
fire %>% pull(month) %>% unique

# Unique values from `day`
fire %>% pull(day) %>% unique

# Specify order of `month` possible values
month.order <- c("jan", "feb", "mar",
                 "apr", "may", "jun",
                 "jul", "aug", "sep",
                 "oct", "nov", "dec")
# Specify order of `day` possible values
day.order <- c("sun", "mon", "tue", "wed", "thu", "fri", "sat")

# Implement specified order of `month` and `day` possible values
fire <- fire %>%
  mutate(
    month = factor(month, levels = month.order),
    day = factor(day, levels = day.order)
  )

# Duplicate `month` variable column
fire$season <- fire$month

# Assign `season` values according to month
fire$season <- revalue(fire$season, c("jan" = "winter",
                                      "feb"="winter",
                                      "mar"="spring",
                                      "apr"="spring",
                                      "may"="spring",
                                      "jun"="summer",
                                      "jul"="summer",
                                      "aug"="summer",
                                      "sep"="fall",
                                      "oct"="fall",
                                      "nov"="fall",
                                      "dec"="winter"))

# Duplicate `day` variable column
fire$day.type <- fire$day

# Assign `day.type` values according to day of week
fire$day.type <- revalue(fire$day.type,
                         c("mon" = "weekday",
                           "tue" = "weekday",
                           "wed"= "weekday",
                           "thu" = "weekday",
                           "fri" = "weekend",
                           "sat" = "weekend",
                           "sun" = "weekend"))

# Find optimal lambda paramater for Box-Cox transformation
bc.lambda <- BoxCox.lambda(fire$area) # Equals 0.02935387

# Implement Box-Cox transformation of response variable `area`
# Add transformed `area` to a copy of the dataset
fire.copy <- data.frame(fire)
fire.copy$bc.area <- (((fire$area)^bc.lambda) - 1)/bc.lambda

# Perform ad-hoc transformation log(area + 1)
# For brevity, we call this new variable `log.area`
fire$log.area <- log(fire$area + 1)

# Separate observations where log.area.plus.1 = log(1) = 0
# These observations correspond to fires where the original `area`=0
fire.1 <- fire[fire$log.area != log(1), ]
fire.0 <- fire[fire$log.area == log(1), ]

# Set seed to ensure reproducible data split
set.seed(12345)

# Select 80% of the data
selected.obs <- sample(dim(fire.1)[1],
                       size=((0.8)*nrow(fire.1)),
                       replace=FALSE)

# Execute the data split
train.fire <- fire.1[selected.obs, ] # 216 training set observations
test.fire <- fire.1[-selected.obs, ] # 54 test set observations


fires.plot <- train.fire %>%
  pivot_longer(
    cols = c("FFMC", "DMC", "DC",
             "ISI", "temp", "RH",
             "wind", "rain"),
    names_to = "data_col",
    values_to = "value"
  )


fig.1 <- fires.plot %>%
  ggplot(aes(x=value, y=area)) +
  geom_point(alpha=0.5, size=0.75) +
  facet_wrap(vars(data_col), scales = "free_x", nrow=2) +
  labs(
    caption="Figure 1: Bivariate relationships between numerical regressor variables
and area burned (in hectares, original units)",
    x = " ",
    y = "Area burned (hectares)") +
  theme(plot.title = element_text(hjust=0.5), plot.margin = unit(c(2.5,0.1,2.5,0.1),"cm"))

print(fig.1)

fires.plot <- train.fire %>%
  pivot_longer(
    cols = c("FFMC", "DMC", "DC",
             "ISI", "temp", "RH",
             "wind"),
    names_to = "data_col",
    values_to = "value"
  )
fig.2 <- fires.plot %>%
  ggplot(aes(x=value, y=log.area, color=season)) +
  geom_point(alpha=0.5, size=0.75) +
  facet_wrap(vars(data_col), scales = "free_x", nrow=2) +
  labs(
    caption = "Figure 2: Bivariate relationships between numerical regressor variables and \n log(y+1) area burned grouped by season",
    x = " ",
    y = "log(y+1) area burned") +
  theme(plot.title = element_text(hjust=0.5),plot.margin = unit(c(2,0.5,2,0.5),"cm"))

print(fig.2)

# Intercept-only model
null.model <- log.area ~ 1
# Full model (all predictors with interactions)
full.model <- log.area ~ season + day.type + FFMC + DMC + DC +
  ISI + temp + RH + wind + season:FFMC + season:DMC +
  season:DC + season:ISI + season:temp

# Summary of full model (all predictors with interactions)
summary(lm(full.model, data=train.fire))

# Resulting model is log.area ~ season + day.type + DMC + DC + season:DC
aic.backward <- stepAIC(lm(full.model, data=train.fire),
                        scope=list(lower=null.model, upper=full.model),
                        direction="backward", trace=FALSE, k=2)
aic.backward$anova

# Summary of model (Backward selection of AIC)
summary(lm(log.area ~ season + day.type + DMC + DC + temp + season:DC,
           data=train.fire))

AIC(lm(log.area ~ season + day.type + DMC + DC + temp + season:DC,
       data=train.fire))

X <- model.matrix(full.model, data=train.fire)[,-1]
y <- train.fire$log.area
lambda.grid <- 10^seq(5, -5, length = 100)
lasso <- glmnet(y=y, x=X, alpha=1, lambda=lambda.grid, standardize=TRUE)
set.seed(12345)
cv.lasso <- cv.glmnet(x=X, y=y, alpha=1, nfolds=10)
best.lambda <- cv.lasso$lambda.min
lasso.plot <- plot(lasso, xvar = "lambda", sub="Figure 3: LASSO coefficient trails", cex.lab = 0.9, cex.axis = 0.9, cex.sub = 0.7)
lines(c(log(best.lambda), log(best.lambda)), c(-1000, 1000), lty = "dashed", lwd = 1.5)


# LASSO coefficients using the optimal lambda hyperparameter
best.lasso.coefs <- predict(lasso, type = 'coefficients', s=best.lambda)
# Find number of nonzero LASSO estimated coefficients
num.nonzero.lasso.coefs <- sum(best.lasso.coefs != 0)
# Find total number of coefficients (i.e. from the full model)
num.total.coefs <- length(best.lasso.coefs)
c(num.nonzero.lasso.coefs, num.total.coefs)

rownames(coef(lasso, s=best.lambda))[coef(lasso, s=best.lambda)[,1] != 0]

summary(lm(log.area ~ season + day.type,
           data=train.fire))

# Extract value of model's AIC (LASSO)
AIC(lm(log.area ~ season + day.type,
       data = train.fire))

final.model <- lm(log.area ~ season + day.type + DMC + DC + temp + season:DC, data=train.fire)

# Diagnostic plots
par(mfrow = c(1,4))
plot(lm(log.area ~ season + day.type + DMC + DC + temp + season:DC,
        data=train.fire),
     cex.caption=1, cex.oma.main=0.7, cex.lab=1, cex=0.5, cex.main = 0.5)
title(main="Figure 4: Diagnostic Plots for Backward-Stepwise Selected Model (AIC)",
      outer=TRUE, line=-1, cex.main=1.5)

# Note the very high value of `area`
# All the values for `area` except these two outliers were less than 600
# Refer to Figure 1 in Exploratory Data Analysis
fire[239,]

# Note the very high value of `area`
# All the values for `area` except these two outliers were less than 600
# Refer to Figure 1 in Exploratory Data Analysis
fire[416,]

coeff.abs <- sort(abs(final.model$coefficients))
coeff.sorted <- final.model$coefficients[names(coeff.abs)]
neg.coeffs.plot <- plot_summs(final.model, coefs = c("seasonsummer:DC", "seasonspring:DC",
                                                     "seasonfall:DC", "day.typeweekday",
                                                     "seasonsummer"),colors="magenta", ci_level=0.95) + labs(title = "Neg. Coeffs.")
pos.coeffs.plot <- plot_summs(final.model, coefs = c("DC", "DMC", "temp", "seasonspring", "seasonfall"), ci_level=0.95) + labs(title = "Pos. Coeffs.")
grid.arrange(neg.coeffs.plot, pos.coeffs.plot, nrow=1, bottom = textGrob("Figure 4: Coefficient plot with 95% confidence intervals",
                                                                         gp=gpar(fontface=1, fontsize=8),
                                                                         hjust=1, x=1))

# Evaluate how well our Final Model fits to the values `log.area`=0
# Recall `log.area` represents a log(y+1) transformation of `area`
# Model predictions for `log.area`=0 at 95% confidence level
zero.pred <- predict(final.model, newdata=fire.0,
                     interval="prediction", level=0.95)
# Calculate RMSE
sqrt(mean((zero.pred[,1] - rep(0, length(zero.pred[,1])))^2))

# Test set predictions for `log.area` at 95% confidence level
test.pred <- predict(final.model, newdata=test.fire,
                     interval="prediction", level=0.95)
# Append test set 95% prediction intervals to test set dataframe
test.pred.df <- cbind(test.fire, test.pred)
# Predicted (fitted) `log.area`,= values from training set
train.pred <- final.model$fitted.values
# Create data visualization
fire.predict.plot <- ggplot(test.pred.df, aes(x=log.area, y=fit)) +
  geom_point(alpha=0.5) +
  geom_smooth(aes(y=lwr), color="red", linetype="dashed") +
  geom_smooth(aes(y=upr, color="upr"),color="red", linetype="dashed") + 
  geom_smooth(aes(color="model"), color="blue", stat="smooth", method="gam", formula=y~s(x, bs="cs")) +
  labs(x="Actual Values", y="Predicted values",
       caption="Figure 5: Model fit, predicting new values for log-transformed area burned") +
  scale_color_manual("linear relation", values=c("red", "blue")) +
  theme(plot.title=element_text(hjust=0.5), legend.position=c(0.25, 0.8),text=element_text(size=10)) +
  ggtitle("Linear model: Predicting log-transformed area burned") + 
  annotate(geom="text", size=4, x=1, y=6, label=paste("Train RMSE:", round(sqrt(mean((train.pred - train.fire$log.area)^2)), 2)), color="black") + 
  annotate(geom="text", size=4, x=3, y=6, label=paste("Test RMSE:", round(sqrt(mean((test.pred.df$fit - test.fire$log.area)^2)), 2)), color="black")

print(fire.predict.plot)

# Recall `log.area` represents a log(y+1) transformation of `area`
# Model predictions for `log.area`=0 at 95% confidence level
zero.pred <- predict(final.model, newdata=fire.0,
                     interval="prediction", level=0.95)
# Calculate RMSE
sqrt(mean((zero.pred[,1] - rep(0, length(zero.pred[,1])))^2))

# Compile predicted values `log.area`=0 into dataframe
# Include their true values (all of which are 0) as a second column
zero.df <- data.frame(zero.pred[,1], rep(0, length(zero.pred[,1])))

# Rename columns for brevity and convenience
names(zero.df)[1] <- "Predicted Zero"
names(zero.df)[2] <- "Actual Zero"
zero.df$Residual <- zero.df$`Actual Zero` - zero.df$`Predicted Zero`
zero.residual.plot <- ggplot(data=zero.df) +
  geom_histogram(bins=30, aes(x=`Predicted Zero`), color="white") +
  geom_vline(xintercept=0, linetype="dashed", size=0.5, color="black") +
  labs(x="Predicted Value of Log(Area+1)=0", y="Count",
       caption="Figure 7: Distribution of 'Predicted Zero' Values of log.area") +
  theme(text=element_text(size=8), plot.title = element_text(hjust = 0.5))

print(zero.residual.plot)


# Test Set 1 predictions for `log.area` at 95% confidence level
# This test set contains all true values of `log.area` that are nonzero
# Convert predicted values into original `area` units (in hectares or ha)
test1.pred.area <- exp(predict(final.model, newdata=test.fire))-1
# Actual values of `area` for Test Set 1 (in hectares or ha)
test1.actual.area <- test.fire$area
# Test Set 2 predictions for `log.area` at 95% confidence level
# This test set contains all true values of `log.area` that are zero
# Convert predicted values into original `area` units (in hectares or ha)
test2.pred.area <- exp(predict(final.model, newdata=fire.0))-1
# Actual values of `area` for Test Set 2 (in hectares or ha)
test2.actual.area <- fire.0$area
# Create 2 dataframes for values based on Test Set 1 and Test Set 2
# Each dataframe contains 1 column of actual values and 1 column of predicted values
# Again, these are in original units of `area` (in hectares or ha)
# For Test Set 1
burn.df <- data.frame(
  'Burned predicted' = test1.pred.area,
  'Burned actual' = test1.actual.area)
# For Test Set 2
no.burn.df <- data.frame(
  'No Burn predicted' = test2.pred.area,
  'No Burn actual' = test2.actual.area)

# RMSE of Test Set 1, in original units of `area` (hectares or ha)
round(sqrt(mean((burn.df$Burned.predicted - burn.df$Burned.actual)^2)), 2)

# RMSE of Test Set 2, in original units of `area` (hectares or ha)
round(sqrt(mean((no.burn.df$No.Burn.predicted - no.burn.df$No.Burn.actual)^2)),2)

# From burn.df, we find the ratio of each predicted `area` value
# over its corresponding actual `area` value
pred.actual.ratio <- (burn.df$Burned.predicted)/(burn.df$Burned.actual)
# Percentage of points that had predicted > actual (overestimated)
sum(pred.actual.ratio > 1)/length(pred.actual.ratio)

# Total sum of the true area burned by all fires described in `burn.df`
sum(burn.df$Burned.actual)
# Total sum of the predicted area burned by all fires described in `burn.df`
sum(burn.df$Burned.predicted)
# Total sum of the true area by all fires described in `no.burn.df`
sum(no.burn.df$Burned.actual)
# Total sum of the predicted area by all fires described in `no.burn.df`
sum(no.burn.df$No.Burn.predicted)
