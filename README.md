# Hospital Readmission Prediction

## Project Overview
This project predicts 30-day hospital readmission using clinical and demographic data. 

## Tools and Libraries
- R
- XGBoost
- Caret
- dplyr

## Steps
1. Data Cleaning: Handle missing values, encode categorical variables.
2. Model Training: XGBoost with 5-fold cross-validation.
3. Evaluation: Achieved Validation Log Loss of 0.6309.

## How to Run
- Install required R libraries.
- Run `hospital_readmission_prediction.R` in RStudio.

## Expected Output
- The script outputs a **submission-ready CSV file** containing probabilistic predictions for 30-day readmissions.
- Log Loss metrics are printed to validate model performance.

## Project Files
- `hospital_readmission_prediction.R`: The main R script for data preprocessing, model training, and prediction.
- `hm7-samplesubmission.csv`: Example submission file with predictions.
