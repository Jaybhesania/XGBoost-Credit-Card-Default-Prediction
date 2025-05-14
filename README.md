# Credit Card Default Prediction with XGBoost & Amazon SageMaker

---

## Project Overview

This project focuses on building a **machine learning pipeline to classify credit card customers** based on their likelihood to default on their next payment. The workflow integrates **EDA**, **feature engineering**, **model building**, **hyperparameter tuning**, and **deployment on AWS SageMaker**, offering a full-stack ML solution in the financial domain.

---

## Project Structure

```
.
├── UCI_Credit_Card.csv               # Original dataset from UCI repository
├── AI_in_Finance_Sagemaker.ipynb     # Complete notebook: EDA, ML, SageMaker deployment
├── README.md                         # Project overview and documentation (this file)
```

---

## Dataset

* **Source**: UCI Machine Learning Repository
* **Size**: 30,000 records × 25 columns
* **Target Variable**: `default.payment.next.month` (1 = default, 0 = no default)
* **Attributes**: Demographics, payment history, bill amounts, and payment amounts

---

## Key Tasks

### 1. Data Exploration & Visualization

* Identified no missing values
* Detected **class imbalance**: \~22% defaults
* Visualized relationships across age, gender, education, and credit behavior
* Used correlation heatmaps, histograms, boxplots, and KDE plots

### 2. Preprocessing

* Dropped unnecessary columns (e.g., `ID`)
* Performed One-Hot Encoding on categorical features
* Applied **Min-Max Scaling** to numerical features

### 3. Model Building: XGBoost Classifier

* **Baseline Accuracy**: \~81.5%
* Evaluated using **confusion matrix**, **precision**, **recall**, and **F1-score**

### 4. Hyperparameter Tuning

* Used **GridSearchCV** to tune:

  * `gamma`
  * `subsample`
  * `colsample_bytree`
  * `max_depth`
* Best Parameters Achieved \~**81.57% Accuracy**

### 5. Model Deployment on AWS SageMaker

* Converted data for SageMaker format (target as first column)
* Uploaded training & validation data to S3
* Trained XGBoost model using built-in SageMaker container
* Deployed model as an endpoint
* Performed real-time inference
* **Final SageMaker Accuracy**: \~81.42%

---

## Final Evaluation Metrics (Deployed Model)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 81.42% |
| Precision | 74.91% |
| Recall    | 64.74% |

---

## Tech Stack

* **Languages**: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
* **ML Model**: XGBoost
* **Cloud Platform**: Amazon SageMaker (S3, IAM, built-in containers)
* **Environment**: AWS Jupyter Notebook

---

## How to Run

1. Upload the dataset `UCI_Credit_Card.csv`
2. Open `AI_in_Finance_Sagemaker.ipynb` in Jupyter or Google Colab
3. Run each cell step-by-step
4. (Optional) Set up AWS credentials for SageMaker deployment

---

## References

* UCI Dataset: [UCI Credit Card Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
* AWS SageMaker Docs: [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/index.html)
* XGBoost Docs: [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)

---

## Future Work

* Try other classifiers (e.g., LightGBM, CatBoost, Logistic Regression)
* Apply SMOTE or ensemble techniques to handle class imbalance
* Deploy a real-time dashboard with predictions
* Integrate model monitoring and retraining pipelines

---

## Credits

Inspired by real-world credit scoring systems and cloud-based ML workflows.

---
