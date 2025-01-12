# -------------------- Load Libraries --------------------

library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)
library(lubridate)
library(GGally)
library(zoo)
library(heatmaply)

# -------------------- Helper Functions --------------------

# Function to handle missing values (impute with median)
impute_missing_values <- function(data) {
  data[is.na(data)] <- lapply(data, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
  return(data)
}

create_lag_features <- function(data, lag_steps = 1, selected_features) {
  for (sensor in selected_features) {
    new_col <- paste0(sensor, "_lag", lag_steps)  # Only create a single lag column
    if (!new_col %in% names(data)) {
      data[[new_col]] <- shift(data[[sensor]], n = lag_steps, fill = NA)  # Shift by only 1 lag step
    }
  }
  return(data)
}

# -------------------- Preprocessing Training Data with RUL --------------------

file_ids <- c("FD001", "FD002", "FD003", "FD004") # Dataset identifiers
for (file_id in file_ids) {
  train_file <- paste0("test_", file_id, ".txt")
  rul_file <- paste0("RUL_", file_id, ".txt")
  
  # Load data
  train_data <- fread(train_file, header = FALSE, sep = " ")
  rul_data <- fread(rul_file, header = FALSE)
  
  # Process last cycles
  test_last_cycle <- train_data %>%
    group_by(V1) %>%
    filter(V2 == max(V2)) %>%
    ungroup() %>%
    mutate(true_RUL = rul_data$V1)
  
  # Save processed data
  output_file <- paste0("processed_last_cycle_", file_id, ".csv")
  fwrite(test_last_cycle, output_file)
  print(paste("Processed and saved:", output_file))
}

# Merge datasets into a single data table
merged_data <- bind_rows(
  fread("processed_last_cycle_FD001.csv", header = TRUE),
  fread("processed_last_cycle_FD002.csv", header = TRUE),
  fread("processed_last_cycle_FD003.csv", header = TRUE),
  fread("processed_last_cycle_FD004.csv", header = TRUE)
)

# Rename columns
colnames(merged_data) <- c("unit_number", "time", "operational_setting_1", "operational_setting_2", 
                           "operational_setting_3", paste0("sensor_measurement_", 1:21), 
                           "RUL")

write.csv(merged_data, "TrainData.csv", row.names = FALSE)

# -------------------- Preprocessing Testing Data -----------------------

file_ids <- c("FD001", "FD002", "FD003", "FD004") # Dataset identifiers
all_test_data <- list()  # Create an empty list to store data

for (file_id in file_ids) {
  test_file <- paste0("train_", file_id, ".txt")
  
  # Load data
  test_data <- fread(test_file, header = FALSE, sep = " ")
  
  # Append data to the list
  all_test_data[[file_id]] <- test_data
  print(paste("Loaded data from:", test_file))
}

# Combine all test files into one data table
merged_test_data <- rbindlist(all_test_data)  # Combine the data tables

# Rename columns to match the expected structure
colnames(merged_test_data) <- c("unit_number", "time", "operational_setting_1", 
                                "operational_setting_2", "operational_setting_3", 
                                paste0("sensor_measurement_", 1:21))

# Save the merged data to a CSV file
write.csv(merged_test_data, "TestData.csv", row.names = FALSE)

# -------------------- Feature Importance with Random Forest --------------------

# Load training data
train_data <- fread("TrainData.csv")

# Extract only sensor columns and the RUL (response)
sensor_cols <- train_data %>% select(starts_with("sensor_measurement_"))
response_var <- train_data$RUL

# Random Forest Model for feature importance
library(randomForest)
set.seed(42)  # For reproducibility
rf_model <- randomForest(x = sensor_cols, y = response_var, importance = TRUE, ntree = 100)

# Extract feature importance
feature_importance <- importance(rf_model)  # Extract importance scores
feature_importance_df <- data.frame(Feature = rownames(feature_importance), Importance = feature_importance[, 1])

# Print feature importance scores
print("Feature Importance Scores:")
print(feature_importance_df)

# Sort and visualize the top features
feature_importance_df <- feature_importance_df %>%
  arrange(desc(Importance))

# Plot the feature importance for visualization
ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance Scores",
       x = "Features",
       y = "Importance") +
  theme_minimal()

# Select the top important features (e.g., top 10 features by importance)
num_features_to_select <- 10
selected_features <- feature_importance_df %>%
  arrange(desc(Importance)) %>%
  head(num_features_to_select) %>%
  pull(Feature)

print("Selected Important Features:")
print(selected_features)

# -------------------- Preprocessing Transfer Learning Evaluation Data --------------------

# Load test data
final_test_data <- read.table('final_test.txt', header = FALSE, sep = '', fill = TRUE)
num_cols <- ncol(final_test_data)

# Dynamically assign column names
column_names <- c("unit_number", "time", "operational_setting_1", "operational_setting_2", 
                  "operational_setting_3", 
                  paste0('sensor_measurement_', 1:(num_cols - 5)))
colnames(final_test_data) <- column_names

# Select features for lagging - same features selected from the training phase
selected_features <- c("sensor_measurement_13", "sensor_measurement_15", 
                       "sensor_measurement_11", "sensor_measurement_4", 
                       "sensor_measurement_14", "sensor_measurement_2", 
                       "sensor_measurement_3", "sensor_measurement_9", 
                       "sensor_measurement_8", "sensor_measurement_7") 

# Preprocess test data - handle missing values by imputing median
final_test_data <- impute_missing_values(final_test_data)

# Only retain the selected important sensor features and the operational settings
final_test_data <- final_test_data %>% select(
  unit_number, time, operational_setting_1, operational_setting_2, operational_setting_3,
  all_of(selected_features)
)

# Add lag features to the test data (only one lag step for each important sensor)
final_test_data <- create_lag_features(final_test_data, lag_steps = 1, selected_features)

# Handle missing values after lagging (remove rows with NA introduced by lagging)
final_test_data <- na.omit(final_test_data)

# Write the preprocessed test data to CSV
write.csv(final_test_data, "EvaluationDataFinal.csv", row.names = FALSE)

# -------------------- Preprocess Training Data with Selected Features ----------------------

# Normalize and handle missing data
train_data <- impute_missing_values(train_data)

# Only retain the selected important sensor features from train_data
train_data <- train_data %>% select(
  unit_number, time, operational_setting_1, operational_setting_2, operational_setting_3,
  all_of(selected_features), RUL
)

# Add only one lag feature for each important sensor
train_data <- create_lag_features(train_data, lag_steps = 1, selected_features)

# Handle missing values post lag feature creation
train_data <- na.omit(train_data)

write.csv(train_data, "TrainDataFinal.csv", row.names = FALSE)

# -------------------- Feature Selection & Lagging for Test Data -----------------------

# Selected sensor features for lagging, as identified previously
selected_features <- c("sensor_measurement_13", "sensor_measurement_15", 
                       "sensor_measurement_11", "sensor_measurement_4", 
                       "sensor_measurement_14", "sensor_measurement_2", 
                       "sensor_measurement_3", "sensor_measurement_9", 
                       "sensor_measurement_8", "sensor_measurement_7") 

# Preprocess Test Data
test_data <- fread("TestData.csv")  # Load the combined raw test data

# Handle missing values by imputing median values
test_data <- impute_missing_values(test_data)

# Select only the required operational settings and selected sensor features
test_data <- test_data %>% select(
  unit_number, time, operational_setting_1, operational_setting_2, operational_setting_3,
  all_of(selected_features)
)

# Add lag features for selected sensor columns
test_data <- create_lag_features(test_data, lag_steps = 1, selected_features)

# Handle missing values introduced by lagging
test_data <- na.omit(test_data)

# Save the preprocessed test data
write.csv(test_data, "TestDataFinal.csv", row.names = FALSE)

# -------------------- Final EDA --------------------

# Load preprocessed cleaned data
train_data <- fread("TrainDataFinal.csv")
test_data <- fread("TestDataFinal.csv")

# Visualization for lag features
ggplot(train_data, aes(x = time)) +
  geom_line(aes(y = sensor_measurement_13_lag1, color = "Sensor 13")) +
  geom_line(aes(y = sensor_measurement_7_lag1, color = "Sensor 7")) +
  labs(title = "Lag Features of Selected Sensors Over Time", x = "Time", y = "Values")
