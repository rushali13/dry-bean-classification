# Dry Bean Classification Using Machine Learning

## 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms to classify dry bean varieties using physical characteristics of bean samples.

The objective is to evaluate different classification models using standard performance metrics and identify the model that performs best on the test dataset.

## 2. Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Original instances:** 13,611
- **Instances after removing exact duplicates:** 13,543
- **Number of features:** 16
- **Target variable:** Class
- **Number of classes:** 7

The dataset contains numerical measurements of dry beans, including area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity, convex area, and other shape-related characteristics.

The seven classes are:
BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Removed 68 exact duplicate records.
- Separated the input features and target variable.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve class proportions.
- Applied feature scaling where required by the algorithms.

The final dataset contains 13,543 records and 16 input features.

## 4. Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler was used in the pipelines for algorithms that benefit from feature scaling. The SVM model uses probability calibration for multiclass AUC evaluation.

## 5. Model Performance Comparison

The models were evaluated on the same held-out test dataset using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9947 | 0.9311 | 0.9303 | 0.9306 | 0.9028 |
| Decision Tree | 0.8955 | 0.9460 | 0.9094 | 0.9104 | 0.9098 | 0.8737 |
| KNN | 0.9155 | 0.9832 | 0.9290 | 0.9256 | 0.9270 | 0.8978 |
| Gaussian Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| SVM | **0.9210** | 0.9911 | **0.9344** | **0.9314** | **0.9328** | **0.9045** |

## 6. Observations and Results

- SVM achieved the highest accuracy, precision, recall, F1-score, and MCC among the five evaluated models.
- Logistic Regression achieved the highest AUC score.
- KNN also demonstrated competitive classification performance.
- Decision Tree and Gaussian Naive Bayes obtained comparatively lower accuracy and F1-scores.
- The results indicate that different algorithms perform differently on the same dataset, making evaluation across multiple metrics important.

## 7. Best Performing Model

Based on the evaluation results, **Support Vector Machine (SVM)** was the best overall-performing model.

It achieved an accuracy of 92.10% and an F1-score of 0.9328. Logistic Regression achieved the highest AUC of 0.9947.

## 8. Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Upload a CSV file containing test data.
- Select one of the five trained classification models.
- View model performance metrics.
- View the confusion matrix and classification report.
- Download the model comparison results.

The application loads previously saved models and does not retrain them when a user uploads a test CSV.

**Live Application:** [Add your deployed Streamlit URL here]

## 9. GitHub Repository

**
