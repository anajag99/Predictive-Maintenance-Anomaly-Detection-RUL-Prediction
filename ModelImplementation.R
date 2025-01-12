# Load necessary libraries
library(xgboost)
library(data.table)  # For faster data manipulation
library(caret)       # For data splitting and evaluation
library(ggplot2)     # For plotting

# Load the training data (TrainDataFinal.csv)
train_data <- read.csv("TrainDataFinal.csv")

# Load the test data (TestDataFinal.csv) for testing
test_data <- read.csv("TestDataFinal.csv")

# Load the evaluation data (EvaluationDataFinal.csv) for transfer learning evaluation
evaluation_data <- read.csv("EvaluationDataFinal.csv")

# Check for missing values in the train, test, and evaluation data (optional, remove if not needed)
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
evaluation_data <- na.omit(evaluation_data)

# Separate features and target (RUL) in the training data
train_target <- train_data$RUL
train_features <- train_data[, -which(names(train_data) == "RUL")]

# Convert the training and test data to the DMatrix format required by XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_features), label = train_target)

# Prepare the test data for prediction (no target column)
test_features <- test_data

# Train the XGBoost model
params <- list(
  objective = "reg:squarederror",   # For regression task
  eval_metric = "rmse",              # Use RMSE as evaluation metric
  max_depth = 6,                     # Depth of each tree
  eta = 0.1,                         # Learning rate
  nthread = 2,                       # Number of threads to use
  subsample = 0.8,                   # Subsample ratio of the training data
  colsample_bytree = 0.8             # Subsample ratio of features
)

# Train the model (use 100 iterations for now)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 1
)

# Save the trained model for future use
saveRDS(xgb_model, "xgboost_rul_model.rds")

# Prediction on the test data
dtest <- xgb.DMatrix(data = as.matrix(test_features))
rul_predictions <- predict(xgb_model, dtest)

# Print the first few predictions
print(head(rul_predictions))

#------------------------------ Prediction Analysis ------------------------------

# Prediction Summary: Descriptive statistics of the predicted RUL values
cat("Prediction Summary:\n")
summary(rul_predictions)

# Visualize the distribution of predicted RUL values (Histogram)
ggplot(data.frame(predictions = rul_predictions), aes(x = predictions)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Predicted Remaining Useful Life (RUL)", x = "Predicted RUL", y = "Frequency") +
  theme_minimal()

# Boxplot for the predicted RUL values to check for outliers
ggplot(data.frame(predictions = rul_predictions), aes(y = predictions)) +
  geom_boxplot(fill = "blue", alpha = 0.7) +
  labs(title = "Boxplot of Predicted RUL", y = "Predicted RUL") +
  theme_minimal()

#------------------------------ Feature Importance ------------------------------

# Feature importance: To understand which features are most important for the prediction
importance_matrix <- xgb.importance(feature_names = colnames(train_features), model = xgb_model)
xgb.plot.importance(importance_matrix)

#------------------------------ Transfer Learning Evaluation ------------------------------

# Prepare the evaluation data for prediction
eval_features <- evaluation_data

# Use the trained model to make predictions on the evaluation data
deval <- xgb.DMatrix(data = as.matrix(eval_features))
rul_eval_predictions <- predict(xgb_model, deval)

# Print the first few predictions for evaluation data
print(head(rul_eval_predictions))

# Prediction Summary for the evaluation data:
cat("Prediction Summary for Evaluation Data:\n")
summary(rul_eval_predictions)

# Visualize the distribution of predicted RUL values for evaluation data (Histogram)
ggplot(data.frame(predictions = rul_eval_predictions), aes(x = predictions)) +
  geom_histogram(bins = 30, fill = "red", alpha = 0.7) +
  labs(title = "Predicted Remaining Useful Life (RUL) - Evaluation Data", x = "Predicted RUL", y = "Frequency") +
  theme_minimal()

# Boxplot for the predicted RUL values in the evaluation data to check for outliers
ggplot(data.frame(predictions = rul_eval_predictions), aes(y = predictions)) +
  geom_boxplot(fill = "red", alpha = 0.7) +
  labs(title = "Boxplot of Predicted RUL - Evaluation Data", y = "Predicted RUL") +
  theme_minimal()

#------------------------------ Model Analysis ------------------------------

# Model Summary: Display the model's parameters and training settings
cat("Model Parameters and Settings:\n")
print(params)

cat("The predicted RUL values range from", min(rul_eval_predictions), "to", max(rul_eval_predictions), "\n")

# Number of boosting rounds (iterations)
cat("Number of boosting rounds (iterations): ", xgb_model$bestIteration, "\n")

#------------------------------ Save Predictions ------------------------------

# Save the predictions on the evaluation data to a CSV file
write.csv(data.frame(Predicted_RUL = rul_eval_predictions), "predicted_rul_evaluation.csv", row.names = FALSE)

