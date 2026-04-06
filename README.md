# Financial Risk Prediction: End-to-End Feature Engineering Pipeline

## 📌 Project Overview
This project demonstrates a professional-grade, end-to-end data preprocessing and feature engineering pipeline designed for financial risk modeling. The objective was to take a raw, messy dataset of financial and demographic records and transform it into a highly optimized, multicollinearity-free, and mathematically independent matrix ready for machine learning deployment to predict loan defaults (`loan_default`).

## 🛠️ Technologies & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn`
* **Statistical Analysis:** `statsmodels` (VIF)
* **Data Visualization:** `matplotlib`, `seaborn`

## 🚀 Pipeline Architecture

### 1. Data Cleaning & Preprocessing
* **Imputation:** Handled missing continuous variables (`age`, `credit_score`) using median imputation to resist outlier skew, and categorical variables (`employment_type`) using statistical mode.
* **Outlier Handling:** Applied 99th-percentile Winsorization (capping) to extreme financial metrics (`income`, `monthly_spend`) to stabilize the dataset without losing valuable records.

### 2. Feature Transformation
* **Logarithmic Scaling:** Applied to severely right-skewed variables (`income`, `monthly_spend`) to normalize distributions.
* **Min-Max Scaling:** Scaled bounded variables (`credit_score`) strictly between 0.0 and 1.0.
* **Discretization (Binning):** Segmented continuous `age` into logical cohorts (Young, Middle-Aged, Senior) to capture non-linear generational trends.

### 3. Feature Engineering
Constructed high-signal, domain-specific features to provide deeper context to the predictive model:
* **Loan-to-Income Ratio:** Measures absolute debt burden and affordability.
* **Spend-to-Income Ratio:** Captures cash flow tightness and financial discipline.
* **Risk-Adjusted Capacity:** An interaction feature combining financial capacity and behavioral scores to mathematically amplify ideal borrower signals.

### 4. Categorical Encoding
* **Ordinal Encoding:** Preserved natural hierarchies for `education_level`, `risk_tolerance`, and `age_group`.
* **One-Hot Encoding:** Applied to nominal variables (`region`, `employment_type`) utilizing `drop_first=True` to explicitly prevent the dummy variable trap.

### 5. Feature Selection & Multicollinearity Management
* Validated feature predictive strength using **Mutual Information** and **Pearson Correlation** scores.
* Conducted a rigorous **Variance Inflation Factor (VIF)** analysis.
* Dropped strictly redundant features (e.g., `household_cashflow_score` with a 0.96 correlation). 
* Strategically retained highly predictive "parent" features despite high VIF scores, opting to resolve the multicollinearity mathematically downstream.

### 6. Dimensionality Reduction (PCA)
* Applied **Principal Component Analysis (PCA)** to the fully standardized feature space to resolve engineered multicollinearity and compress the dataset.
* **Result:** Successfully reduced the dataset from 20 engineered features down to **15 Principal Components**, reducing computational width by 25% while mathematically preserving **95% of the total dataset variance**.

## 📁 Files in this Repository
* `Feature_Engineering_Pipeline.ipynb`: The complete Jupyter Notebook containing all code, mathematical proofs, and visualizations.
* `Final_Engineered_Loan_Data_PCA.csv`: The final, compressed, machine-learning-ready output matrix.
* `Graded Assignment 1 Dataset.csv`: The original raw dataset (for reproduction).

## 🧠 Key Takeaways
This project highlights the critical balance between domain-driven feature construction and statistical constraints. By prioritizing predictive signals and utilizing PCA to handle the resulting multicollinearity, the final dataset provides an optimal foundation for predictive algorithms ranging from Logistic Regression to advanced ensemble methods.
