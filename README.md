# Predictive Maintenance: Anomaly Detection and RUL Prediction

## Abstract
This project focuses on predictive maintenance for industrial equipment, combining anomaly detection and Remaining Useful Life (RUL) prediction to optimize maintenance schedules and reduce downtime. The model uses One-Class Support Vector Machine (SVM) for anomaly detection and XGBoost regression for RUL prediction, tested on the NASA Turbofan Engine Degradation Simulation dataset. The results show an F1-score of 0.89 for anomaly detection and an RMSE of 45.2 for RUL prediction.

## Problem Statement
Predictive maintenance has become essential for minimizing downtime and extending the lifespan of critical machinery. This study explores how integrating anomaly detection with RUL prediction can enhance the accuracy and effectiveness of maintenance strategies.

## Dataset
The primary dataset used is the NASA Turbofan Engine Degradation Simulation dataset, which contains time-series sensor data from turbofan engines. The dataset includes sensor readings (pressure, temperature, rotational speed) and a target variable, RUL, representing the remaining operational cycles before failure.

Key features:
- Sensor readings: Time-series data from various sensors.
- RUL: Remaining useful life of the engine.
- Anomaly flag: Binary indicator for detected anomalies.

## Preprocessing Steps
1. **Handling Missing Values:** Imputed missing values using the mean of each respective sensor reading.
2. **Feature Scaling:** Standardization applied to normalize sensor data.
3. **Lag Feature Creation:** Introduced lag features to capture temporal dependencies in time-series data.
4. **Dimensionality Reduction (PCA):** Applied PCA to reduce the dimensionality and retain the most important features.

## Methodology
### 1. Anomaly Detection with One-Class SVM
- One-Class SVM is used for unsupervised anomaly detection, learning the distribution of normal operating data and flagging deviations as anomalies.
- Achieved an F1-score of 0.89, indicating strong performance in identifying anomalies.

### 2. RUL Prediction with XGBoost
- XGBoost regression is employed for RUL prediction, using sensor readings and lag features to predict the remaining useful life.
- The model achieved an RMSE of 45.2, providing valuable estimates for proactive maintenance actions.

### 3. Transfer Learning
- Transfer learning was applied using the PHM2008 dataset for enhanced model generalization across different datasets.
- This approach helps the model adapt to new machinery or operating conditions without extensive retraining.

## Results
- **Anomaly Detection:** The One-Class SVM model achieved an F1-score of 0.89, successfully identifying anomalies with minimal false positives.
- **RUL Prediction:** The XGBoost model achieved an RMSE of 45.2, indicating reasonable accuracy in forecasting RUL.
- **Combined Approach:** The hybrid model integrating anomaly detection and RUL prediction outperformed individual models, providing a robust predictive maintenance solution.

## Future Work
- Explore advanced anomaly detection techniques like autoencoders or GANs for better handling of rare anomalies.
- Optimize the XGBoost model for real-time deployment in predictive maintenance systems.
- Experiment with deep learning models for both anomaly detection and RUL prediction.
