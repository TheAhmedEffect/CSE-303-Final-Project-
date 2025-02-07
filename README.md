# Loan Default Prediction

This project analyzes and predicts loan defaults using machine learning techniques. The dataset is preprocessed, visualized, and used to train classification models such as Logistic Regression and Support Vector Machine (SVM). Performance is evaluated using classification metrics and ROC curves.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Machine Learning Models](#machine-learning-models)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
- [License](#license)

## Overview
This project provides an end-to-end analysis and modeling process for loan default prediction. It includes:
- Data preprocessing (handling missing values, scaling features)
- Exploratory Data Analysis (EDA)
- Machine Learning model training and evaluation
- Model performance visualization

## Dataset
The dataset used is `LoanData_Preprocessed_v1.2.csv`, which contains information about loan applicants and their financial status. The target variable `default` indicates whether a loan was defaulted (1) or not (0).

## Installation
To run this project, ensure you have Python installed. Install the required dependencies using:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
```
2. Run the script:
```bash
python loan_default_analysis.py
```

## Visualization
The script generates multiple visualizations to understand data distribution and correlations:
- Loan default class balance
- Income distribution
- Debt-to-income ratio distribution
- Correlation heatmap

## Machine Learning Models
The project implements two machine learning models:
1. **Logistic Regression** - A simple linear model for binary classification.
2. **Support Vector Machine (SVM)** - A non-linear model with kernel functions for classification.

## Performance Evaluation
Model performance is assessed using:
- **Classification Report** (Precision, Recall, F1-score)
- **ROC-AUC Score** (Ability to distinguish between classes)
- **ROC Curve** (Comparison of model performance visually)

## Results
The script prints classification reports and generates an ROC curve comparing Logistic Regression and SVM models.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
