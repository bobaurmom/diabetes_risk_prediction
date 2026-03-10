# Diabetes Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/Algorithm-Logistic%20Regression-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)

## 📝 Project Overview
This project implements a **Logistic Regression** machine learning model to predict the likelihood of diabetes in patients based on 16 clinical signs and symptoms.

Using the "Early Stage Diabetes Risk Prediction Dataset," the model identifies the strongest predictors—such as **Polyuria** (excessive urination) and **Polydipsia** (excessive thirst)—to classify individuals into "Positive" or "Negative" risk categories. This serves as a diagnostic support tool to help prioritize high-risk patients for further clinical testing.



## 🛠️ Key Features Analyzed
The model processes the following patient data points:

* **Demographics:** Age and Gender.
* **Primary Symptoms:** Polyuria, Polydipsia, Polyphagia, and Sudden Weight Loss.
* **Secondary Indicators:** Visual blurring, Itching, Irritability, Delayed healing, and Alopecia.
* **Physical Risk Factors:** Obesity, Muscle Stiffness, and Partial Paresis.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/diabetes-risk-prediction.git](https://github.com/your-username/diabetes-risk-prediction.git)
cd diabetes-risk-prediction
```

### 2. Install Dependencies
```Bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
### Steps to Run the Code
```Run the main script:
python main.py
```

## 📋 Attribute Information

| Attribute | Description |
| :--- | :--- |
| **Age** | Age of the patient (Range: 20–65+). |
| **Gender** | Biological sex (Male or Female). |
| **Polyuria** | Excessive or frequent urination (very common sign). |
| **Polydipsia** | Excessive thirst or increased water intake. |
| **Sudden weight loss** | Unexplained or rapid drop in body mass. |
| **Weakness** | Generalized fatigue, lethargy, or lack of energy. |
| **Polyphagia** | Excessive hunger or significantly increased appetite. |
| **Genital thrush** | Presence of a yeast infection. |
| **Visual blurring** | Blurred vision caused by fluid shifts in the eye lenses. |
| **Itching** | General skin irritation or localized itching. |
| **Irritability** | Mood swings or sudden changes in temperament. |
| **Delayed healing** | Wounds or sores that take an abnormally long time to recover. |
| **Partial paresis** | Significant muscle weakness or partial loss of voluntary movement. |
| **Muscle stiffness** | Tightness or inability to move muscles and joints easily. |
| **Alopecia** | Patchy or sudden hair loss. |
| **Obesity** | High Body Mass Index (BMI). |
| **Class** | **Target Variable:** Positive (1) or Negative (0). |
