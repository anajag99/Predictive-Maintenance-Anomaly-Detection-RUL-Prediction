# Load the pre-trained models
one_class_svm <- readRDS("one_class_svm_model.rds")  # Assuming the model is saved as RDS
xgb_model <- readRDS("xgboost_rul_model.rds")        # XGBoost model for RUL prediction

# Step 1: Anomaly Detection using One-Class SVM
# Load the test features (or evaluation features)
test_features <- read.csv("EvaluationDataFinal.csv")

# Predict anomalies (1 for normal, -1 for anomalous)
anomalies <- predict(one_class_svm, test_features)

# Step 2: RUL Prediction using XGBoost
# Remove the 'anomaly_flag' column for prediction (XGBoost needs the original feature set)
test_features_without_flag <- test_features[, colnames(test_features) != "anomaly_flag"]

# Prepare the test data for prediction (no RUL in the test data, only features)
dtest <- xgb.DMatrix(data = as.matrix(test_features_without_flag))

# Predict RUL with XGBoost
rul_predictions <- predict(xgb_model, dtest)

# Combine the anomaly flags with the RUL predictions
test_features$anomaly_flag <- ifelse(anomalies == 1, 0, 1)  # 1 = anomaly, 0 = normal
test_features$rul_predictions <- rul_predictions

# Check the results (anomaly flags and RUL predictions)
head(test_features)

# ---------------------------- Prediction Analysis ----------------------------

# Descriptive Statistics for Predicted RUL values
cat("Prediction Summary:\n")
summary(rul_predictions)

# Add a column for anomaly detection summary
cat("\nAnomaly Detection Summary:\n")
table(test_features$anomaly_flag)

# Visualizations of predicted RUL values

# Histogram of predicted RUL values
ggplot(data.frame(predictions = rul_predictions), aes(x = predictions)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Predicted Remaining Useful Life (RUL)", x = "Predicted RUL", y = "Frequency") +
  theme_minimal()

# Boxplot for predicted RUL values to check for outliers
ggplot(data.frame(predictions = rul_predictions), aes(y = predictions)) +
  geom_boxplot(fill = "blue", alpha = 0.7) +
  labs(title = "Boxplot of Predicted RUL", y = "Predicted RUL") +
  theme_minimal()

# Visualize Anomalies in RUL predictions (highlight anomalies)
ggplot(test_features, aes(x = rul_predictions, color = factor(anomaly_flag))) +
  geom_density(alpha = 0.6) +
  scale_color_manual(values = c("red", "green")) +
  labs(title = "RUL Predictions with Anomaly Flag", x = "Predicted RUL", y = "Density") +
  theme_minimal()

# ---------------------------- Model and Data Summary ----------------------------

# Model Summary: Display the model's parameters and training settings
cat("\nModel Parameters and Settings:\n")
print(xgb_model$params)

# Summary of predictions and anomalies
cat("\nThe predicted RUL values range from", min(rul_predictions), "to", max(rul_predictions), "\n")
cat("The number of anomalies detected is", sum(test_features$anomaly_flag == 1), "\n")
cat("The number of normal instances is", sum(test_features$anomaly_flag == 0), "\n")

# Number of boosting rounds (iterations)
cat("\nNumber of boosting rounds (iterations): ", xgb_model$bestIteration, "\n")

# ---------------------------- Save Predictions ----------------------------

# Save the predictions on the evaluation data to a CSV file
write.csv(test_features[, c("anomaly_flag", "rul_predictions")], "predicted_rul_evaluation_with_anomalies.csv", row.names = FALSE)

# ---------------------------- Time Series Forecasting Visualization ----------------------------

# Create a time index (if no time column is available)
test_features$time_index <- 1:nrow(test_features)

# Plotting time series of RUL predictions with anomalies
library(ggplot2)

# Time series plot for RUL predictions with anomalies highlighted
ggplot(test_features, aes(x = time_index, y = rul_predictions, color = factor(anomaly_flag))) +
  geom_line(size = 1, alpha = 0.7) +  # Line plot for RUL predictions
  geom_point(aes(color = factor(anomaly_flag)), size = 3, alpha = 0.7) +  # Highlight anomalies
  scale_color_manual(values = c("green", "Blue")) +  # Green for normal, red for anomalies
  labs(title = "Time Series of Predicted RUL with Anomalies",
       x = "Time Step / Instance",
       y = "Predicted RUL",
       color = "Anomaly Flag") +
  theme_minimal() +
  theme(legend.position = "bottom")
