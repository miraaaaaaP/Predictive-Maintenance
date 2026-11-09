# Predictive Maintenance Machine Learning Pipeline

An end-to-end Machine Learning pipeline designed for industrial predictive maintenance. This project focuses on classifying equipment failures using advanced data preprocessing, handling severe class imbalances, and evaluating high-performance predictive models.

## 🚀 Project Overview
In industrial IoT and manufacturing settings, equipment failures are rare events, creating extreme class imbalance datasets. This project builds a robust classification system to predict machinery downtime before it happens, minimizing costly operational disruptions.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning & Modeling:** Scikit-Learn, XGBoost, Random Forest, Support Vector Machines (SVM)
* **Imbalanced Data Handling:** Imbalanced-Learn (SMOTE)

## 📊 Pipeline Workflow
1. **Exploratory Data Analysis (EDA):** Statistical distributions, feature correlations, and boxplot analysis.
2. **Data Preprocessing & Cleaning:** Handling missing values, feature scaling, and encoding categorical variables.
3. **Addressing Class Imbalance:** Utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) to correct the severe imbalance in failure event observations.
4. **Model Training & Evaluation:** Training multiple classifiers (XGBoost, Random Forest, SVM) and evaluating them using metrics tailored for imbalanced data (**Recall, Precision, F1-Score, and ROC-AUC** rather than misleading accuracy).

## 📂 Repository Structure
```text
mounachML/
│
├── predictive_maintenance.ipynb    # Main Jupyter Notebook containing the full workflow
├── .gitignore                      # Git ignore rules (excluding raw data and checkpoints)
└── README.md                       # Project documentation
