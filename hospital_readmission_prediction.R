# Load Libraries
library(caret)
library(xgboost)
library(dplyr)
library(pROC)
library(PRROC)

# Load Data
train_data <- read.csv("hm7-Train-2023.csv")
test_data <- read.csv("hm7-Test-2023.csv")
sample_submission <- read.csv("hm7-samplesubmission.csv")

# Data Preprocessing ----------------------------------------------------------

# Missing Value Imputation
clean_data <- function(data, is_test = FALSE) {
  data[data == ""] <- NA
  
  # Impute numeric columns
  numeric_cols <- sapply(data, is.numeric)
  for (col in names(data)[numeric_cols]) {
    data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
  }
  
  # Impute categorical columns
  categorical_cols <- sapply(data, function(x) is.character(x) || is.factor(x))
  for (col in names(data)[categorical_cols]) {
    mode_value <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
    data[[col]][is.na(data[[col]])] <- mode_value
  }
  
  # Drop unnecessary columns
  cols_to_remove <- c("indicator_2_level", "medical_specialty")
  if (!is_test) cols_to_remove <- c(cols_to_remove, "readmitted")
  data <- data[, !(colnames(data) %in% cols_to_remove)]
  return(data)
}

# Convert all character columns to factors, then numeric
encode_features <- function(data) {
  data <- data %>%
    mutate(across(where(is.character), as.factor)) %>%   # Convert characters to factors
    mutate(across(where(is.factor), as.numeric))         # Convert factors to numeric
  return(data)
}

# Apply Data Cleaning and Encoding
train_clean <- clean_data(train_data)
test_clean <- clean_data(test_data, is_test = TRUE)

train_x <- encode_features(train_clean[, !colnames(train_clean) %in% "patientID"])
test_x <- encode_features(test_clean[, !colnames(test_clean) %in% "patientID"])

# Encode Target Variable
train_y <- factor(train_data$readmitted, levels = c(0, 1), labels = c("Class0", "Class1"))

# Define Log Loss Function ----------------------------------------------------
log_loss <- function(y_true, y_pred_prob) {
  epsilon <- 1e-15
  y_pred_prob <- pmin(pmax(y_pred_prob, epsilon), 1 - epsilon)
  -mean(y_true * log(y_pred_prob) + (1 - y_true) * log(1 - y_pred_prob))
}

# Local Validation Split ------------------------------------------------------
set.seed(42)
train_index <- createDataPartition(train_y, p = 0.7, list = FALSE)

# Split into Training and Validation Sets
train_x_train <- train_x[train_index, ]
train_x_val <- train_x[-train_index, ]
train_y_train <- train_y[train_index]
train_y_val <- train_y[-train_index]

# Cross-Validation Setup
cv_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = mnLogLoss
)

# XGBoost Hyperparameter Grid
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 5,
  subsample = 0.75
)

# Train XGBoost on Training Data ----------------------------------------------
xgb_model <- train(
  x = train_x_train, y = train_y_train,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = xgb_grid,
  metric = "logLoss"
)

# Evaluate on Validation Set --------------------------------------------------
val_preds <- predict(xgb_model, train_x_val, type = "prob")[, "Class1"]
val_true <- as.numeric(train_y_val == "Class1")
val_logloss <- log_loss(val_true, val_preds)
cat("Validation Log Loss:", val_logloss, "\n")

# Final Model Training on Full Data -------------------------------------------
xgb_model_full <- train(
  x = train_x, y = train_y,
  method = "xgbTree",
  trControl = trainControl(method = "none"),  # No CV for full training
  tuneGrid = xgb_grid,
  metric = "logLoss"
)

# Predictions on Test Data ----------------------------------------------------
test_preds <- predict(xgb_model_full, test_x, type = "prob")[, "Class1"]

# Create Submission File ------------------------------------------------------
submission <- data.frame(
  patientId = test_data$patientID,
  predReadmit = test_preds
)
write.csv(submission, "submission.csv", row.names = FALSE)
cat("Submission file 'submission.csv' created successfully.\n")
