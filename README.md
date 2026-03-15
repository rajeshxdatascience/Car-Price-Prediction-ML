# 🚗 Used Car Price Prediction Engine

## 📌 Project Overview
Yeh project ek end-to-end **Machine Learning Development Life-cycle (MLDL)** implementation hai jo used car prices predict karta hai. Is project ka main focus "black-box" libraries ke bajaye **Mathematical Formulation**, rigorous **Regression Analysis**, aur custom **Optimization** techniques par hai.

## 🛠️ Technical Implementation & Highlights

### 1. Data Cleaning & Sanitization
* **Duplicate Management**: 763 duplicate entries ko identify aur remove kiya gaya taaki model bias se bacha ja sake aur **Regression Analysis** ki integrity bani rahe.
* **Handling Challenges**: **Non-representative data** aur potential **Outliers** ko address kiya gaya (jaise high `km_driven` aur `car_age`), jo **Simple Linear Regression** line ko skew kar sakte hain.

### 2. Advanced Feature Engineering
* **High-Cardinality Transformation**: 1,491 unique car names ko process karke "Brand" feature extract kiya gaya, jisse dimensionality kam hui aur **Overfitting** prevent hui.
* **Log Transformation**: Target variable (`selling_price`) par logarithmic scaling apply ki gayi taaki **Normality of Residuals** ki assumption satisfy ho sake.
* **Derived Features**: `car_age` aur `usage_intensity` (km per year) jaise **Relevant features** engineer kiye gaye model ki accuracy sudharne ke liye.

### 3. Mathematical Foundations & Optimization
* **Manual Implementation**: **Linear Regression model from scratch** build kiya gaya taaki error function ki **Intuition** aur **Mathematical Formulation** show ki ja sake.
* **Gradient Descent**: **Batch** aur **Stochastic Gradient Descent (SGD)** implement kiya gaya **Loss Function** ko minimize karne ke liye.
* **Learning Dynamics**: **Learning Rate** ka effect aur different optimization techniques ke beech **Time comparison** ko analyze kiya gaya.

### 4. Regression Analysis & Model Diagnostics
* **Multicollinearity Detection**: **Variance Inflation Factor (VIF)** aur **Correlation** matrices ka use karke redundant predictors ko remove kiya gaya.
* **Assumptions Validation**: **Linearity**, **Homoscedasticity**, aur **No Autocorrelation** ke liye formal checks kiye gaye.
* **Feature Selection**: **Filter methods** (Correlation, ANOVA) aur **Wrapper methods** (Sequential Backward Elimination) ka use karke optimal feature set select kiya gaya.

## 📊 Evaluation Metrics
Model performance ko in standard **Regression Metrics** par measure kiya gaya:
* **MAE** (Mean Absolute Error)
* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)
* **Adjusted R² Score**: Isse prioritize kiya gaya taaki unnecessary complexity ko penalize karke **Over-fitting** se bacha ja sake.

## 📂 Repository Structure
```text
├── data/               # Raw aur cleaned CSV files
├── notebooks/          # EDA, VIF Analysis, aur Assumptions Testing
├── src/                # Custom Gradient Descent aur Regression classes
└── README.md           # Project documentation
