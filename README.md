# Used Car Price Prediction Engine

## Project Overview
This repository contains an end-to-end implementation of the **Machine Learning Development Life-cycle (MLDL)** focused on predicting the market value of pre-owned vehicles. The primary objective was to move beyond "black-box" implementations and focus on **Mathematical Formulation**, rigorous **Regression Diagnostics**, and custom **Optimization** strategies.

## Technical Implementation & Highlights

### 1. Data Cleaning & Integrity
* **Deduplication**: Identified and removed 763 duplicate entries to prevent data leakage and maintain the statistical integrity of the regression analysis.
* **Outlier Mitigation**: Addressed non-representative data points, such as extreme mileage and vehicle age, which would otherwise skew the Ordinary Least Squares (OLS) estimation.

### 2. Feature Engineering & Statistical Transformation
* **Dimensionality Management**: Processed 1,491 unique car models to extract "Brand" as a primary feature, reducing model complexity and mitigating the risk of overfitting.
* **Logarithmic Transformation**: Applied a log transformation to the target variable (`selling_price`) to satisfy the **Normality of Residuals** and address potential heteroscedasticity—a critical Gauss-Markov assumption.
* **Feature Synthesis**: Engineered derived variables including `car_age` and `usage_intensity` (annual kilometers driven) to more accurately model vehicle depreciation.

### 3. Mathematical Foundations & Optimization
* **Manual Implementation**: Developed a Linear Regression model from first principles to demonstrate the mathematical intuition behind error minimization and matrix operations.
* **Gradient Descent Algorithms**: Implemented both **Batch** and **Stochastic Gradient Descent (SGD)** to optimize the Loss Function.
* **Analysis of Dynamics**: Evaluated the impact of **Learning Rates** and conducted a time-complexity comparison between different optimization solvers.

### 4. Regression Analysis & Model Diagnostics
* **Multicollinearity Detection**: Utilized **Variance Inflation Factor (VIF)** and correlation matrices to identify and eliminate redundant predictors.
* **Assumption Validation**: Conducted formal testing for **Linearity**, **Homoscedasticity**, and **Independence of Errors** (No Autocorrelation).
* **Feature Selection**: Employed a hybrid strategy using **Filter methods** (Correlation, ANOVA) and **Wrapper methods** (Sequential Backward Elimination) to derive the most parsimonious model.

## Evaluation Metrics
Model performance was benchmarked using standard regression metrics to ensure generalized predictive power:
* **MAE** (Mean Absolute Error)
* **MSE / RMSE** (Mean Squared Error / Root Mean Squared Error)
* **Adjusted R² Score**: Prioritized as the primary metric to account for model complexity and prevent over-fitting.

## Repository Structure
```text
├── data/               # Raw and processed datasets
├── notebooks/          # EDA, VIF Analysis, and Statistical Assumptions Testing
└── README.md           # Project documentation
