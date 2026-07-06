setwd("C:\\Users\\ngonyams\\Downloads") 
getwd()
vehicle <- read.csv(file="train.csv", stringsAsFactors=FALSE)
vehicle <- vehicle[, -1,-5]
vehicle <- vehicle[vehicle$Policy_Sales_Channel %in% c(152, 124, 26, 160, 156, 122), ]
vehicle$Policy_Sales_Channel <- as.factor(vehicle$Policy_Sales_Channel)
vehicle$Gender <- as.factor(vehicle$Gender)
vehicle$Vehicle_Age <- as.factor(vehicle$Vehicle_Age)
vehicle$Vehicle_Damage <- as.factor(vehicle$Vehicle_Damage)
vehicle$Response <- as.factor(vehicle$Response)
vehicle$Previously_Insured <- as.factor(vehicle$Previously_Insured)
vehicle$Driving_License <- as.factor(vehicle$Driving_License)
boxplot(vehicle$Age, xlab="Age", col="yellow")
boxplot(vehicle$Annual_Premium, xlab="premium", col="yellow") #outlier present above the maximum
boxplot(vehicle$Policy_Sales_Channel, xlab="salespolicy", col="yellow")
boxplot(vehicle$Vintage, xlab="vintage", col="yellow")
table(vehicle$Response)
prop.table(table(vehicle$Response))
table(vehicle$Gender)
prop.table(table(vehicle$Gender))
table(vehicle$Driving_License)
prop.table(table(vehicle$Driving_License))
summary(vehicle)
boxplot(Age ~ Response, data=vehicle, main="Age", col=c("blue","yellow"))
boxplot(Annual_Premium ~ Response, data=vehicle, main="AnnualPremium", col=c("blue","red"))
boxplot(Vintage ~ Response, data=vehicle, main="Vintage", col=c("blue","yellow"))                                                                                                                                                                                                                                                                                                                                                                                                   
library(ggplot2)

ggplot(vehicle, aes(x = Policy_Sales_Channel, fill = Response)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    name = "Response",
    labels = c("0 = Not interested", "1 = interested")
  ) +
  labs(
    title = "PSC vs VehicleInsurance",
    x = "PSC",
    y = "Count"
  ) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    position = position_dodge(width = 0.9),
    vjust = -0.3
  )
ggplot(vehicle, aes(x = Vehicle_Age, fill = Response)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    name = "Response",
    labels = c("0 = Not interested", "1 = interested")
  ) +
  labs(
    title = "VehicleAge vs VehicleInsurance",
    x = "VehicleAge",
    y = "Count"
  ) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    position = position_dodge(width = 0.9),
    vjust = -0.3
  )
ggplot(vehicle, aes(x = Vehicle_Damage, fill = Response)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    name = "Response",
    labels = c("0 = Not interested", "1 = interested")
  ) +
  labs(
    title = "VehicleDamage vs VehicleInsurance",
    x = "Damage",
    y = "Count"
  ) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    position = position_dodge(width = 0.9),
    vjust = -0.3
  )
ggplot(vehicle, aes(x = Driving_License, fill = Response)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    name = "Response",
    labels = c("0 = Not interested", "1 = interested")
  ) +
  labs(
    title = "DrivingLicence vs VehicleInsurance",
    x = "Driverslicence",
    y = "Count"
  ) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    position = position_dodge(width = 0.9),
    vjust = -0.3
  )
ggplot(vehicle, aes(x = Previously_Insured, fill = Response)) +
  geom_bar(position = "dodge", color = "black") +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    name = "Response",
    labels = c("0 = Not interested", "1 = interested")
  ) +
  labs(
    title = "PreviouslyInsured vs VehicleInsurance",
    x = "PreviouslyInsured",
    y = "Count"
  ) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    position = position_dodge(width = 0.9),
    vjust = -0.3
  )
library(ggcorrplot)
num_data <- vehicle[, c("Age","Annual_Premium","Vintage")]
cormat <- cor(num_data, use = "complete.obs")
ggcorrplot(cormat,
           type = "lower",
           lab = TRUE,
           title = "Correlation Heat Map",
           tl.cex = 15,
           lab_size = 5)
str(vehicle)
summary(vehicle)
library(lattice)
library(caret)

set.seed(1350)
index <- createDataPartition(vehicle$Response, p = 0.8, list = FALSE)
train_data <- vehicle[index, ]
test_data  <- vehicle[-index, ]
prop.table(table(vehicle$Response))

prop.table(table(train_data$Response))

prop.table(table(test_data$Response))

#####LOGISTIC REGRESSION
log_model <- glm(Response ~ ., 
                 data = train_data,
                 family = "binomial")
summary(log_model)
coef(log_model)
z <- predict(log_model, test_data, type = "link")
logistic_prob <- 1 / (1 + exp(-z))
head(logistic_prob)
x <- seq(-10, 10, length=100)
y <- 1 / (1 + exp(-x))
plot(x, y, type="l",
     col="blue",
     lwd=2,
     xlab="Linear Predictor (z)",
     ylab="Probability",
     main="Logistic Function")

library(caret)
pred_prob <- predict(log_model, test_data, type="response")
pred_class <- ifelse(pred_prob > 0.2, 1, 0)
confusionMatrix(as.factor(pred_class),
                as.factor(test_data$Response),
                positive = "1")

library(pROC)

roc_curve <- roc(test_data$Response, pred_prob)
plot(roc_curve)

auc(roc_curve)

plot(roc_curve,
     col = "blue",
     lwd = 3,
     main = "ROC Curve for Logistic Regression Model",
     print.auc = TRUE,          # show AUC value
     auc.polygon = TRUE,        # shade AUC
     auc.polygon.col = rgb(0, 0, 1, 0.2))  # transparent blue shading

abline(a = 0, b = 1, lty = 2, col = "gray")

library(ggplot2)
imp_df <- data.frame(
  Variable = names(coef(log_model))[-1],
  Importance = abs(coef(log_model)[-1])
)
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
ggplot(imp_df[1:10, ], 
       aes(x = Importance, 
           y = reorder(Variable, Importance))) +
  geom_point(size = 4) +
  labs(title = "Variable Importance (Logistic Regression)",
       x = "Absolute Coefficient",
       y = "Variables") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
#####PROBIT REGRESSION
probit_model <- glm(Response ~ ., 
                    data = train_data,
                    family = binomial(link = "probit"))

summary(probit_model)
coef(probit_model)
pred_prob_probit <- predict(probit_model, test_data, type = "response")
head(pred_prob_probit)
pred_class_probit <- ifelse(pred_prob_probit > 0.2, 1, 0)
confusionMatrix(as.factor(pred_class_probit),
                as.factor(test_data$Response),
                positive = "1")
library(pROC)

roc_curve_probit <- roc(test_data$Response, pred_prob_probit)
plot(roc_curve_probit,
     col = "red",
     lwd = 3,
     main = "ROC Curve for Probit Model",
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = rgb(0, 0, 1, 0.2))

abline(a = 0, b = 1, lty = 2, col = "gray")
auc(roc_curve_probit)
library(ggplot2)
imp_df <- data.frame(
  Variable = names(coef(probit_model))[-1],
  Importance = abs(coef(probit_model)[-1])
)
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
library(ggplot2)
ggplot(imp_df[1:10, ], 
       aes(x = Importance, 
           y = reorder(Variable, Importance))) +
  geom_point(size = 4) +
  labs(title = "Variable Importance (Probit Regression)",
       x = "Absolute Coefficient",
       y = "Variables") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
########RANDOM FOREST
train_data$Region_Code <- NULL
test_data$Region_Code  <- NULL
# 1. Load libraries
library(randomForest)
library(pROC)
library(caret)

train_data$Response <- as.factor(train_data$Response)
test_data$Response  <- as.factor(test_data$Response)


colSums(is.na(train_data))

set.seed(1350)

rf_model <- randomForest(
  Response ~ .,
  data = train_data,
  ntree = 100,
  importance = TRUE,
  classwt = c("0" = 1, "1" = 5)
)

varImpPlot(rf_model)
rf_prob <- predict(rf_model, test_data, type = "prob")[,2]

roc_rf <- roc(test_data$Response, rf_prob)
plot(roc_rf,
     col = "blue",
     lwd = 3,
     main = "ROC Curve - Random Forest",
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = rgb(0, 0, 1, 0.2))

abline(a = 0, b = 1, lty = 2, col = "gray")
auc(roc_rf)


rf_class <- ifelse(rf_prob > 0.5, 1, 0)
confusionMatrix(
  as.factor(rf_class),
  test_data$Response,
  positive = "1"
)
imp_df <- na.omit(imp_df)

imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
imp <- importance(rf_model)
imp_df <- data.frame(
  Variable = rownames(imp),
  Importance = imp[, "MeanDecreaseGini"]
)
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
library(ggplot2)

ggplot(imp_df[1:9, ], 
       aes(x = Importance, 
           y = reorder(Variable, Importance))) +
  geom_point(size = 4) +   # circles
  labs(title = "Variable Importance (Random Forest)",
       x = "Importance",
       y = "Variables") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

############################
############################
# C5.0 MODEL
############################


########################################
###########xgboost

library(xgboost)
library(caret)
library(pROC)


train_y <- as.numeric(as.character(train_data$Response))
test_y  <- as.numeric(as.character(test_data$Response))

train_matrix <- model.matrix(Response ~ . - 1, data = train_data)
test_matrix  <- model.matrix(Response ~ . - 1, data = test_data)

dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_y)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(1350)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 20
)

xgb_pred_prob <- predict(xgb_model, dtest)
roc_xgb <- roc(test_y, xgb_pred_prob)
best_t <- coords(roc_xgb, "best", ret = "threshold", transpose = FALSE)
best_t <- as.numeric(best_t)
cat("Optimal threshold:", best_t, "\n")
xgb_pred_class <- ifelse(xgb_pred_prob > best_t, 1, 0)
conf_mat <- confusionMatrix(
  as.factor(xgb_pred_class),
  as.factor(test_y),
  positive = "1"
)
print(conf_mat)
plot(roc_xgb,
     col = "purple",
     lwd = 3,
     main = "ROC Curve - XGBoost",
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = rgb(0.5, 0, 0.5, 0.2))

abline(a = 0, b = 1, lty = 2, col = "gray")
library(xgboost)
library(ggplot2)

# Get importance matrix
imp <- xgb.importance(model = xgb_model)
imp_df <- imp[1:9, ]
ggplot(imp_df, 
       aes(x = Gain, 
           y = reorder(Feature, Gain))) +
  geom_segment(aes(x = 0, 
                   xend = Gain, 
                   y = reorder(Feature, Gain), 
                   yend = reorder(Feature, Gain))) +
  geom_point(size = 4) +
  labs(title = "Variable Importance (XGBoost)",
       x = "Importance (Gain)",
       y = "Variables") +
  theme_minimal()

####################################
model <- c("Logistic", "Probit", "Random Forest", "XGBoost")

sensitivity <- c(0.783, 0.832, 0.950, 0.934)
specificity <- c(0.750, 0.751, 0.638, 0.688)
balanced_acc <- c(0.791, 0.792, 0.794, 0.811)

metrics_df <- data.frame(model, sensitivity, specificity, balanced_acc)


library(reshape2)
metrics_long <- melt(metrics_df, id.vars="model")


library(ggplot2)

ggplot(metrics_long, aes(x=model, y=value, fill=variable)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="Model Performance Comparison",
       y="Score",
       x="Model",
       fill="Metric") +
  theme_minimal()



