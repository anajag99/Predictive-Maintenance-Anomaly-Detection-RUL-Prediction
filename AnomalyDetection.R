# Load necessary libraries
library(e1071)
library(ggplot2)

# Load data
train_data <- read.csv("TrainDataFinal.csv")
test_data <- read.csv("TestDataFinal.csv")

# Handle missing values (optional)
train_data <- na.omit(train_data)  # Remove rows with missing values in train data
test_data <- na.omit(test_data)    # Remove rows with missing values in test data

# Ensure the test data has the same columns as the train data
common_columns <- intersect(colnames(train_data), colnames(test_data))

# Subset both train and test data to include only the common columns
train_data <- train_data[, common_columns]
test_data <- test_data[, common_columns]

# Train the One-Class SVM model on the train data
model <- svm(train_data, type = "one-classification", nu = 0.1, kernel = "radial", decision.values = TRUE)

# Summary of the model
summary(model)

# Save the trained model
saveRDS(model, "one_class_svm_model.rds")

# Make predictions on the test data and get decision values
predictions <- predict(model, test_data, decision.values = TRUE)

# Extract decision values
decision_values <- attr(predictions, "decision.values")

# Set a threshold (e.g., 0.5, can be adjusted based on your needs)
threshold <- 0.5

# Classify as normal or anomaly based on threshold
predictions_classified <- ifelse(decision_values > threshold, 1, -1)

# Optional: Check the proportion of anomalies detected
table(predictions_classified)

# Print summary of predictions
summary(predictions_classified)

# Visualization 1: Distribution of decision values (Histogram)
ggplot(data.frame(decision_values), aes(x = decision_values)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Decision Values", x = "Decision Values", y = "Frequency") +
  theme_minimal()

# Visualization 2: Decision values vs. predicted labels
df <- data.frame(decision_values = decision_values, Prediction = factor(predictions_classified))
ggplot(df, aes(x = decision_values, fill = Prediction)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Decision Values by Prediction", x = "Decision Values", y = "Frequency") +
  scale_fill_manual(values = c("red", "green"), labels = c("Anomaly", "Normal")) +
  theme_minimal()

#------------------------------ Transfer Learning Evaluation------------------------------------

# Load the saved model
model <- readRDS("one_class_svm_model.rds")

# Load the evaluation data
evaluation_data <- read.csv("EvaluationDataFinal.csv")

# Handle missing values (optional)
evaluation_data <- na.omit(evaluation_data)  # Remove rows with missing values in evaluation data

# Ensure the evaluation data has the same columns as the model's training data
common_columns <- intersect(colnames(evaluation_data), colnames(model$SV))

# Subset the evaluation data to include only the common columns
evaluation_data <- evaluation_data[, common_columns]

# Make predictions on the evaluation data and get decision values
predictions <- predict(model, evaluation_data, decision.values = TRUE)

# Extract decision values
decision_values <- attr(predictions, "decision.values")

# Set a threshold (e.g., 0.5, can be adjusted based on your needs)
threshold <- 0.5

# Classify as normal or anomaly based on threshold
predictions_classified <- ifelse(decision_values > threshold, 1, -1)

# Optional: Check the proportion of anomalies detected
table(predictions_classified)

# Print summary of predictions
summary(predictions_classified)

# Performance Metric 1: Proportion of anomalies vs normal
cat("Proportion of Anomalies: ", sum(predictions_classified == -1) / length(predictions_classified), "\n")
cat("Proportion of Normal: ", sum(predictions_classified == 1) / length(predictions_classified), "\n")

# Performance Metric 2: Distribution of decision values (confidence level)
cat("Decision Value Statistics:\n")
summary(decision_values)

# Optional: Check how many data points have decision values near the threshold (uncertain cases)
near_threshold <- sum(abs(decision_values - threshold) < 0.1)  # threshold sensitivity
cat("Number of points near the threshold (uncertain cases): ", near_threshold, "\n")

# Visualization 1: Distribution of decision values (Histogram)
ggplot(data.frame(decision_values), aes(x = decision_values)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Decision Values on Evaluation Data", x = "Decision Values", y = "Frequency") +
  theme_minimal()

# Visualization 2: Decision values vs. predicted labels
df <- data.frame(decision_values = decision_values, Prediction = factor(predictions_classified))
ggplot(df, aes(x = decision_values, fill = Prediction)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Decision Values by Prediction (Evaluation Data)", x = "Decision Values", y = "Frequency") +
  scale_fill_manual(values = c("red", "green"), labels = c("Anomaly", "Normal")) +
  theme_minimal()

# Visualization 3: Scatter plot of features with anomalies highlighted (assuming 2D data)
if (ncol(evaluation_data) == 2) {
  df_plot <- data.frame(evaluation_data, Prediction = factor(predictions_classified))
  ggplot(df_plot, aes(x = evaluation_data[, 1], y = evaluation_data[, 2], color = Prediction)) +
    geom_point() +
    labs(title = "Anomaly Detection on Evaluation Data (2D)", x = colnames(evaluation_data)[1], y = colnames(evaluation_data)[2]) +
    scale_color_manual(values = c("red", "green"), labels = c("Anomaly", "Normal")) +
    theme_minimal()
}
