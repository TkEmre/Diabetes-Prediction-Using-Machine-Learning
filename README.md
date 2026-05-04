# Diabetes Prediction — Machine Learning Pipeline

An end-to-end machine learning pipeline for predicting diabetes risk based on clinical measurements from the **Pima Indians Diabetes Database**.

**Course:** SENG 352 — Data Analysis  
**Authors:** Emre Tek & Betül Özdemir  
**Date:** May 2026

## 📌 Project Overview
Diabetes affects over 537 million adults globally. Early detection is critical for effective intervention. This project implements a robust 10-stage ML pipeline to handle common clinical data issues—such as missing values, class imbalance, and outliers—while providing interpretable results through Explainable AI (XAI) techniques.

## 📊 Dataset Details
- **Source:** UCI Machine Learning Repository / Kaggle
- **Samples:** 768 female patients
- **Features:** 8 clinical features (Glucose, BMI, Age, Insulin, etc.) + 1 engineered feature.
- **Target:** `Outcome` (1: Diabetic, 0: Non-Diabetic)

## 🛠 The 10-Step ML Pipeline
The project follows a rigorous scientific workflow:
1.  **Data Quality Assessment:** Biologically impossible zeros (e.g., Glucose, BMI) were converted to `NaN`.
2.  **Missing Value Imputation:** Handled using class-stratified median imputation.
3.  **Outlier Management:** Capped extreme values using the **IQR (Interquartile Range)** method (Winsorizing).
4.  **Feature Engineering:** Categorized BMI values based on WHO standards.
5.  **Data Splitting:** 80/20 Stratified Train-Test split to maintain class distribution.
6.  **Addressing Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set.
7.  **Model Training:** Evaluated 7 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, Naive Bayes, KNN).
8.  **Hyperparameter Tuning:** Optimized using `RandomizedSearchCV`.
9.  **Performance Evaluation:** Focused on **F1-Score** and **ROC-AUC** to balance precision and recall.
10. **Model Explainability (XAI):** Visualized decision-making processes using **SHAP**, **LIME**, and **DiCE**.

## 📈 Performance Results
The **XGBoost** model demonstrated superior performance after tuning:

| Model | F1-Score | ROC-AUC | Accuracy |
| :--- | :--- | :--- | :--- |
| **XGBoost (Tuned)** | **0.840** | **0.955** | **0.876** |
| Random Forest | 0.827 | 0.950 | 0.857 |
| Decision Tree | 0.827 | 0.912 | 0.870 |

## 🔍 Explainable AI (XAI) Insights
- **SHAP Analysis:** Identified **Glucose**, **BMI**, and **Age** as the most significant predictors of diabetes.
- **LIME:** Provided local explanations for individual patient predictions, showing how specific clinical values shifted the risk score.
- **DiCE (Counterfactuals):** Generated "What-if" scenarios (e.g., "If this patient's Glucose level were 110 instead of 150, the prediction would flip to Non-Diabetic").

## 💻 Tech Stack
- **Languages:** Python 3.x
- **Libraries:** - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `imbalanced-learn`
  - Explainability: `shap`, `lime`, `dice-ml`
- **Environment:** Jupyter Notebook

## 🚀 Installation & Usage
To run the analysis locally:

```bash
# Clone the repository
git clone [https://github.com/username/diabetes-ml-pipeline.git](https://github.com/username/diabetes-ml-pipeline.git)

# Install dependencies
pip install xgboost imbalanced-learn shap lime dice-ml

# Run the notebook
jupyter notebook diabetes_ml_pipeline.ipynb
