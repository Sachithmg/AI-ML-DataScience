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
# Classical Machine Learning with Scikit-Learn for SVM | RF | KNN

## Summary

This project showcases the implementation and performance comparison of three classical machine learning models: Support Vector Machine (SVM), Random Forest (RF), and K-Nearest Neighbour (KNN). The models are applied to multiple datasets, including:

1. **Early Stage Diabetes Risk Prediction Dataset**
2. **Breast Cancer Wisconsin (Diagnostic) Dataset**
3. **Wine Dataset**

### Model Details
- **Support Vector Machine (SVM)**: Used for high-dimensional spaces and effective in cases where the number of dimensions exceeds the number of samples.
- **Random Forest (RF)**: An ensemble method that operates by constructing a multitude of decision trees and outputting the mode of the classes.
- **K-Nearest Neighbour (KNN)**: A non-parametric method used for classification that predicts the class of a data point based on the majority class among its k nearest neighbors.

### Datasets
- **Early Stage Diabetes Risk Prediction Dataset**: A dataset consisting of medical records of 520 patients, used to predict diabetes risk.
- **Breast Cancer Wisconsin (Diagnostic) Dataset**: A well-known dataset containing features extracted from digitized images of breast masses, aimed at distinguishing between benign and malignant tumors.
- **Wine Dataset**: Comprises the results of a chemical analysis of wines grown in a specific region in Italy, used to predict the type of wine.

### Results
The models performed as follows:
- **SVM**: Generally provided high accuracy and precision, especially excelling in high-dimensional datasets.
- **RF**: Showed robustness and good performance across all datasets, particularly effective in handling noisy data.
- **KNN**: Performed well, especially in smaller datasets, though sensitive to noise and computationally expensive in larger datasets.

These models demonstrate the effectiveness of classical machine learning techniques in various classification tasks, with each model excelling under different conditions.

---------------------------------------------------------------------------------------------------------------

# CNN and Transfer Learning on Leaf Disease Dataset

## Overview

This project explores the application of Convolutional Neural Networks (CNN) and Transfer Learning to classify tea leaf diseases (White Spot, Algal Leaf, Brown Blight) using the following pre-trained networks:

- ResNet50V2
- ResNet101V2
- VGG16
- VGG19

## Dataset

The dataset consists of images representing three classes of tea leaf diseases. Initially, the dataset was used without augmentation, but later, image augmentation was applied to enhance the model's generalization ability. The final dataset distribution after augmentation is:

- **Algal Leaf:** 654 images
- **Brown Blight:** 661 images
- **White Spots:** 826 images

## Models

### Model 01
A CNN model built from scratch, optimized using Keras Tuner to determine the best architecture in terms of convolutional layers, pooling layers, fully connected layers, dropout rates, and learning rates.

- **Validation Accuracy:** 0.7570
- **Test Accuracy:** 0.7490

### Model 02
A CNN model leveraging Transfer Learning with pre-trained networks for feature extraction, followed by a redesigned fully connected layer optimized using Keras Tuner.

- **Best Performance with ResNet50V2:**
  - **Validation Accuracy:** 0.8845
  - **Test Accuracy:** 0.8327

## Key Insights

1. **Transfer Learning:** Significantly improves feature extraction and accelerates the development of models with higher accuracy.
2. **Dataset Size:** The amount of data is crucial for model performance, especially for generalization. Image augmentation is an effective strategy to enhance training data.

## Conclusion

This project demonstrates the effectiveness of Transfer Learning in computer vision tasks, particularly when dealing with limited datasets. The final model (Model 02 with ResNet50V2) achieved the highest accuracy, validating the importance of using advanced architectures for feature extraction in complex datasets.

----------------------------------------------------------------------------------------------------------------------------

# Comparison of Bi-LSTM with Index, Word2Vec, GloVe Embedding, and BERT

## Overview
This project explores and compares the performance of various natural language processing models for tweet classification, specifically focusing on health-related tweets. The models compared include Bi-LSTM with different embeddings (Index, Word2Vec, GloVe) and BERT.

## Dataset
The dataset comprises tweets labeled as personal health mentions (1) or non-personal health mentions (0). After removing duplicates and handling imbalances through upsampling, the dataset was split into training and testing sets.

## Objective
1. Develop a sequence/RNN model (LSTM or Bi-LSTM) for classifying tweets.
2. Perform a comprehensive comparison between different models and embeddings.

## Methodology
- **Bi-LSTM with Index Embedding:** A Bi-LSTM model was optimized using Keras tuner, which was then compared with models using other embeddings.
- **Word2Vec Embedding:** Evaluated using a similar Bi-LSTM model architecture.
- **GloVe Embedding:** Employed transfer learning with GloVe embeddings for tweet classification.
- **BERT:** Utilized transformers with BERT for superior performance in tweet classification.

## Results Summary
- **BERT:** Achieved the highest performance with a validation accuracy of 0.9169 and test accuracy of 0.8645, along with an AUC of 0.923.
- **GloVe Embedding:** Followed as the runner-up with a validation accuracy of 0.8934, test accuracy of 0.8240, and an AUC of 0.892.
- **Bi-LSTM with Index Embedding:** Showed a solid performance with a validation accuracy of 0.9040, test accuracy of 0.8010, and an AUC of 0.844.
- **Word2Vec Embedding:** Underperformed relative to expectations, with a validation accuracy of 0.8003, test accuracy of 0.6597, and an AUC of 0.779.

## Exploratory Data Analysis (EDA)
Detailed EDA was conducted, including data duplication checks, distribution analysis, cleanup, and visualization, followed by a training-validation data split.

## Conclusion
The BERT model outperforms the other models in this study, but the potential for improvement remains with the addition of more data and further model refinement.

------------------------------------------------------------------------------------------------------------------------------






