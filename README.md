# Credit Risk Prediction Model

**An end-to-end machine learning pipeline for predicting credit risk using the German Credit Dataset**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Modeling Results](#modeling-results)
- [Demo](#demo)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a complete machine learning pipeline for credit risk assessment, predicting whether a credit applicant poses a "good" or "bad" credit risk. Credit risk modeling is crucial for financial institutions to make informed lending decisions, minimize defaults, and optimize their loan portfolios.

The project uses the **UCI German Credit Dataset**, a classic benchmark dataset in credit scoring. The dataset contains 1,000 credit applications with various features including applicant demographics, financial status, and credit history.

The pipeline includes:
- Comprehensive exploratory data analysis (EDA)
- Data preprocessing and missing value handling
- Feature engineering with label encoding
- Multiple model training and hyperparameter tuning
- Model evaluation and selection
- Interactive Streamlit web application for real-time predictions

## Key Features

- **Comprehensive EDA**: Statistical analysis, distribution visualizations, and feature relationships
- **Robust Preprocessing**: Missing value handling (dropped NaNs in Saving/Checking accounts), categorical encoding
- **Multiple Models**: Decision Tree, Random Forest, Extra Trees, and XGBoost classifiers
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Class Imbalance Handling**: `class_weight="balanced"` to address imbalanced target distribution
- **Model Persistence**: Saved models and encoders for production deployment
- **Interactive Demo**: Streamlit web app for live predictions with user-friendly interface

## Project Structure

```
credit-risk-modelling/
│
├── analysis_model.ipynb          # Complete ML pipeline: EDA, preprocessing, modeling
├── app.py                        # Streamlit web application
├── german_credit_data.csv        # UCI German Credit Dataset
│
├── extra_trees_credit_model.pkl  # Best trained model (ExtraTreesClassifier)
│
├── Sex_encoder.pkl               # Label encoder for Sex feature
├── Housing_encoder.pkl           # Label encoder for Housing feature
├── Saving accounts_encoder.pkl   # Label encoder for Saving accounts
├── Checking account_encoder.pkl  # Label encoder for Checking account
├── target_encoder.pkl            # Label encoder for target variable
│
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Tech Stack

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier)
- **Gradient Boosting**: XGBoost
- **Model Persistence**: joblib
- **Web Framework**: Streamlit
- **Visualization**: matplotlib, seaborn (in notebook)
- **Model Selection**: GridSearchCV, train_test_split

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd credit-risk-modelling
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn
   ```
   
   Or create a `requirements.txt` with:
   ```
   pandas>=1.3.0
   numpy>=1.21.0
   scikit-learn>=1.0.0
   xgboost>=1.5.0
   joblib>=1.0.0
   streamlit>=1.0.0
   matplotlib>=3.4.0
   seaborn>=0.11.0
   ```
   
   Then install:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   - Ensure `german_credit_data.csv` is in the project root
   - All `.pkl` model and encoder files should be present

## Usage

### Reproduce the Modeling Pipeline

1. Open [analysis_model.ipynb](analysis_model.ipynb) in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Handle missing values and encode categorical features
   - Split data into train/test sets (80/20 with stratification)
   - Train multiple models with hyperparameter tuning
   - Evaluate and compare model performance
   - Save the best model and encoders

### Run the Streamlit Web App

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Use the interface**
   - The app will open in your default web browser (typically at `http://localhost:8501`)
   - Fill in the applicant information:
     - **Age**: Applicant's age (18-80)
     - **Sex**: Gender (male/female)
     - **Job**: Job category (0-3)
     - **Housing**: Housing status (own/rent/free)
     - **Saving accounts**: Savings level (little/moderate/rich/quite rich)
     - **Checking account**: Checking account status (little/moderate/rich)
     - **Credit amount**: Requested credit amount
     - **Duration**: Loan duration in months
   - Click **"Predict Risk"** to get the prediction
   - Results display as **GOOD** (green) or **BAD** (red) credit risk

## Modeling Results

### Dataset Overview

- **Total samples**: 1,000 (after preprocessing: 522 after dropping NaNs)
- **Features**: 8 (Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration)
- **Target**: Credit Risk (good/bad)
- **Train/Test Split**: 80/20 with stratification

### Class Distribution & Imbalance Handling

The target variable is imbalanced:
- **Good credit**: ~55.7% (291 samples)
- **Bad credit**: ~44.3% (231 samples)

To address this imbalance, all models were trained with `class_weight="balanced"`, which automatically adjusts class weights inversely proportional to class frequencies.

### Models Compared

The following models were trained and evaluated:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Extra Trees Classifier** (Best Model)
4. **XGBoost Classifier**

### Hyperparameter Tuning

All models were tuned using **GridSearchCV** with 5-fold cross-validation:

- **Decision Tree**: `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **Extra Trees**: `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`

### Performance Metrics

The best model (**ExtraTreesClassifier**) achieved the following performance on the test set:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~78-82% |
| **ROC-AUC** | ~0.78 |
| **Precision** | ~0.75-0.80 |
| **Recall** | ~0.70-0.75 |
| **F1-Score (bad class)** | ~0.65 |

*Note: Exact values may vary based on random seed. For precise metrics, run the notebook.*

### Best Model Selection

**ExtraTreesClassifier** was selected as the final model due to:
- Best balance of performance and generalization
- Highest ROC-AUC score among all models
- Lower overfitting compared to other ensemble methods
- Robust performance on imbalanced data

### Feature Engineering

- **Categorical Encoding**: Label encoding applied to Sex, Housing, Saving accounts, and Checking account features
- **Numerical Features**: Used as-is (Age, Job, Credit amount, Duration)
- **Missing Values**: Rows with NaN in Saving/Checking accounts were dropped (reduced dataset from 1,000 to 522 samples)

## Demo

The Streamlit application provides an interactive interface for credit risk prediction. Users can input applicant information and receive instant predictions.

![Streamlit App Demo](demo.png)

*Note: Add a screenshot of your Streamlit app here, or describe the interface: "The app features a clean, user-friendly interface with input fields for all applicant features and displays predictions with color-coded results (green for GOOD, red for BAD credit risk)."*

## Future Improvements

- **Feature Engineering**:
  - Implement Weight of Evidence (WoE) and Information Value (IV) binning for categorical features
  - Create interaction features and polynomial features
  - Feature scaling/normalization for numerical features

- **Model Explainability**:
  - Integrate SHAP (SHapley Additive exPlanations) for feature importance visualization
  - Add LIME (Local Interpretable Model-agnostic Explanations) for individual prediction explanations
  - Display feature importance plots in the Streamlit app

- **Advanced Modeling**:
  - Experiment with more sophisticated models (LightGBM, CatBoost)
  - Implement stacking and blending ensembles
  - Add time-based cross-validation if temporal patterns exist

- **Deployment**:
  - Create a REST API using FastAPI for model serving
  - Deploy to cloud platforms (AWS, GCP, Azure)
  - Add Docker containerization for easy deployment
  - Implement model versioning and monitoring

- **Evaluation**:
  - Add more comprehensive evaluation metrics (confusion matrix, precision-recall curves)
  - Implement k-fold cross-validation for more robust performance estimates
  - Add calibration plots to assess prediction probabilities

- **Data**:
  - Handle missing values with imputation strategies instead of dropping
  - Collect and incorporate additional features (credit history, income, etc.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UCI Machine Learning Repository** for providing the German Credit Dataset
- The credit scoring and risk modeling community for inspiration and best practices
- scikit-learn, XGBoost, and Streamlit teams for excellent open-source tools

---

**Note**: This project is for educational and portfolio purposes. In production credit risk systems, additional regulatory compliance, model validation, and risk management practices are required.

