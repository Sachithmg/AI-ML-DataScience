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

# Critical Review of CycleGAN and ToDayGAN for Night-to-Day Image Translation

## Overview

This repository contains a critical review and analysis of the CycleGAN and ToDayGAN models, focusing on their application in night-to-day image translation for visual localization tasks. The research explores the challenges of visual localization across varying illumination conditions and the effectiveness of using Generative Adversarial Networks (GANs) to address these challenges.

## Contents

1. **Introduction**: 
   - Discusses the significance of visual localization, especially in applications like autonomous driving, where determining a vehicle's position and orientation in varying lighting conditions is crucial.
   - Introduces the concept of image-to-image translation using GANs, particularly in the context of night-to-day image translation without requiring paired datasets.

2. **Comparison with Related Work**: 
   - Provides an in-depth comparison between ToDayGAN and related models like CycleGAN and ComboGAN.
   - Highlights the challenges and limitations of traditional GAN models in handling unpaired datasets and domain shifts in image localization.

3. **Methodology**: 
   - Details the dataset used, including the Oxford RobotCar dataset, and the modifications made to the CycleGAN architecture to create ToDayGAN.
   - Discusses the role of discriminators in the ToDayGAN model and their specialization in different image characteristics (e.g., color, texture, gradients).

4. **Results and Analysis**: 
   - Presents the performance metrics of ToDayGAN, showcasing significant improvements in accuracy over previous state-of-the-art methods like DenseVLAD.
   - Includes an ablation study that highlights the effectiveness of different components within the ToDayGAN architecture.

5. **Conclusion and Future Works**: 
   - Summarizes the key findings and innovations of the research, including the partitioning of discriminators and the use of multi-scale outputs.
   - Suggests potential future research directions, such as exploring alternative losses, extending the model to other generative tasks, and applying the approach to different environmental conditions.

-------------------------------------------------------------------------------------------------------------------------

# AttnGAN - Fine-Grained Text to Image Generation: A Critical Review

## Overview
This repository contains a critical review of the paper "AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks" by Tao Xu et al. The review delves into the challenges of translating textual descriptions into high-fidelity visual representations, highlighting the AttnGAN model's advancements in word-level attention mechanisms for improving image generation quality.

## Contents
- **Introduction:** Discusses the fundamental problem of text-to-image synthesis and the limitations of existing models that rely on sentence-level vectors.
- **Comparison with Related Work:** Analyzes AttnGAN against other state-of-the-art models such as GAN-INT-CLS, GAWWN, StackGAN, StackGAN-v2, and PPGN, focusing on architectural innovations and performance.
- **Methodology:** Explores the unique components of AttnGAN, including the attentional generative network and the deep attentional multimodal similarity model (DAMSM), which enhance image-text alignment at a fine-grained level.
- **Results and Analysis:** Provides quantitative evaluations of AttnGAN's performance on datasets like CUB and COCO, demonstrating its superiority in generating high-resolution, contextually accurate images.
- **Conclusion and Future Work:** Summarizes the contributions of AttnGAN to text-to-image synthesis and suggests potential avenues for future research, including more complex attention models and leveraging cloud platforms like Google Colab for scalability.
- **Strengths and Weaknesses:** Discusses the novel contributions of AttnGAN, such as its ability to generate fine-grained details, while also acknowledging limitations like global coherence issues and computational complexity.

---------------------------------------------------------------------------------------------------------------

# Face Mask Recognition System using CNN

## Overview
This repository contains the code and resources for the Face Mask Recognition System using Convolutional Neural Networks (CNN). The project aims to develop a robust system capable of detecting whether individuals are wearing face masks in various real-world scenarios with different lighting conditions.

## Research Questions
The study addresses the following key research questions:
1. How do varying lighting conditions in real-world scenarios impact the accuracy of face mask detection models trained on synthetic data?
2. What preprocessing techniques can enhance the accuracy of face mask detection models in real-world scenarios?
3. Which transfer learning architectures yield the best face mask detection outcomes in diverse real-world settings?

## Methodology
The research methodology involves:
- Training CNN models on synthetic datasets.
- Evaluating the models on a test dataset representing real-world lighting conditions (e.g., sunny, cloudy, shadowy, indoor, outdoor artificial light).
- Experimenting with various preprocessing techniques such as sharpness enhancement and face elongation.
- Comparing the performance of different transfer learning architectures including VGG-16, VGG-19, ResNet, DenseNet, and MobileNet.

## Key Findings
- Models trained on synthetic data often struggle with real-world scenarios, especially under varying lighting conditions.
- Preprocessing techniques like sharpness enhancement can improve model accuracy in real-world scenarios.
- DenseNet and ResNet architectures demonstrated superior performance in feature extraction under challenging conditions.

## Recommendations
Future work should focus on:
- Exploring advanced preprocessing techniques.
- Implementing diverse data augmentation strategies.
- Optimizing transfer learning methods to better adapt models to real-world environments.
- Investigating ensemble methods to enhance overall accuracy.

-----------------------------------------------------------------------------------------------------------

# Deep Learning and Computer Vision for Image Text Detection and Recognition

## Overview
This project addresses a critical gap in the field of text detection and recognition by integrating advanced deep learning and computer vision techniques. The focus is on detecting and recognizing text within complex documents that include various structures like tables, lists, paragraphs, and headings. The project employs a combination of Optical Word Recognition (OWR) pipeline, machine learning models, and computer vision methods to enhance text detection and recognition accuracy, especially in documents distorted by skewness and tilt.

## Key Features
- **Text Detection and Recognition Pipeline:** A comprehensive pipeline covering preprocessing, segmentation, word recognition, and postprocessing.
- **Preprocessing Techniques:** Utilizes methods like Hough Transformation, Fast Fourier Transformation (FFT), and Convolutional Neural Networks (CNN) for correcting skewness and tilt.
- **Segmentation:** Identifies and processes tabular and non-tabular data structures, and further segments them into lines and words.
- **Optical Word Recognition (OWR):** Employs Convolutional Recurrent Neural Network (CRNN) for high-accuracy word recognition.
- **Data Management:** Integrates a normalized Entity-Relationship (ER) model stored in a MySQL database to manage course descriptor (CD) information.

## Methodology
The project adopts a Design Science Research Methodology, involving the development of four primary artifacts:
1. **Data Model:** A normalized ER schema to store and retrieve CD content.
2. **Ground Truth Generation Tool:** Extracts and processes text from PDF documents.
3. **Input Dataset Creation:** Generates a dataset with skewed and tilted images for model training.
4. **OWR Pipeline:** Implements the entire detection and recognition process.

## Evaluation
The pipeline was evaluated using metrics such as Mean Squared Error (MSE), confusion matrix, letter accuracy, and Connectionist Temporal Classification (CTC) loss for CRNN models. The OWR model showed promising results, with a sentence-matching ratio of 0.79 for undistorted images and 0.48 for distorted ones.

## Future Work
- **Dataset Expansion:** Expanding the training dataset to include more diverse document types.
- **OCR Model Integration:** Integrating OCR models to handle non-dictionary words and enhance overall recognition accuracy.

## Acknowledgements
This project was completed as part of the Master of Information Technology program at Whitireia & WelTec Education Institute, Wellington, New Zealand. Special thanks to the supervisors and mentors who provided guidance and support throughout the research.

## Keywords
Deep Learning, Computer Vision, Text Detection, Text Recognition, Image Processing, Optical Character Recognition (OCR), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Design Science Research, Entity-Relationship Model.

---------------------------------------------------------------------------------------------------------------






