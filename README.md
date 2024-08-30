# AI-ML-DataScience

## Credit Application Classification

### Project Overview

This project focuses on classifying credit applications as "Good" or "Bad" clients using machine learning techniques. The primary goal is to determine which model, Logistic Regression or Random Forest, performs better on a given credit dataset.

### Key Components

1. **Introduction and Background**
   - Importance of credit risk management for financial institutions.
   - Overview of credit scoring models, including parametric (Logistic Regression) and non-parametric (Random Forest) approaches.

2. **Algorithms Overview**
   - **Logistic Regression:** Detailed explanation of its use in credit scoring, including model assumptions and limitations.
   - **Random Forest:** Explanation of the ensemble method and its advantages over parametric models.

3. **Exploratory Data Analysis**
   - Data preprocessing, including handling missing values and outliers.
   - Data visualization and imputation strategies.
   - Addressing the issue of an imbalanced dataset.

4. **Model Implementation**
   - **Logistic Regression:** Model development, interaction terms, and evaluation using AUC-ROC.
   - **Random Forest:** Implementation, variable importance analysis, and comparison with Logistic Regression.

5. **Conclusion**
   - Comparative analysis of the models with a focus on their predictive performance and resource efficiency.

### Data and Tools
- **Dataset:** Kaggle’s "Give Me Some Credit" dataset.
- **Languages and Libraries:** R programming, `randomForest`, `ggplot2`, `caret`, and other essential data analysis libraries.

------------------------------------------------------------------------------------------------------------
# Dry Bean Classification Using Decision Tree

## Summary

This project focuses on classifying seven different types of dry beans—Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, and Sira—using a Decision Tree classifier. The classification is based on 16 features extracted from high-resolution images, which describe the form, shape, and structure of the beans.

### Model Details
- **Model**: Decision Tree Classifier
- **Libraries Used**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `graphviz`
- **Key Metrics**: Accuracy, Precision, Recall, F1-score

### Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset#)
- **Instances**: 13,611 dry bean samples
- **Features**: 16 (Area, Perimeter, Axis Lengths, Aspect Ratio, Eccentricity, Convex Area, etc.)
- **Classes**: 7 bean types

### Results
The Decision Tree model successfully classifies the different types of dry beans with notable accuracy. Detailed performance metrics such as precision, recall, and F1-score are provided, along with visualizations of the decision tree.

------------------------------------------------------------------------------------------------------------

