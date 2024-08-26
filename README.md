# SPAM-OR-NOT: Email Spam Classification Project

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Machine Learning Techniques](#machine-learning-techniques)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Usage](#usage)
7. [Setup and Installation](#setup-and-installation)

## Introduction

Email spam classification is a common and critical application of machine learning and artificial intelligence. Spam emails, or unsolicited bulk emails, are not only annoying but can also pose significant security threats. The primary goal of this project is to develop a machine learning model that can accurately classify emails as spam or non-spam (ham). By leveraging various machine learning algorithms and techniques, this project aims to provide an effective solution to email spam detection.

## Project Structure

The project is structured to maintain clarity and modularity, making it easy to understand and extend. Here is an overview of the project structure:

SPAM-OR-NOT/ 
│ ├── main.py # Main script for training the model 
├── model.pkl # Saved machine learning model 
├── vectorizer.pkl # Saved TF-IDF vectorizer 
├── requirements.txt # Python dependencies for the project 
├── README.md # Project documentation 
└── SRC/ 
    ├── data_loader.py # Module for loading and processing data 
    ├── preprocessing.py # Module for data preprocessing 
    ├── model.py # Module for defining and training models 
    ├── evaluate.py # Module for evaluating the model 

## Dataset

The project uses the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase), a popular dataset for email classification. The dataset contains 4,601 email messages, each represented by 57 features that quantify various characteristics of the email content, such as the frequency of specific words or characters.

- **Features**: Numeric representations of various email content characteristics (e.g., word frequency).
- **Target Variable**: `is_spam` (1 if spam, 0 if not spam).

The dataset is ideal for demonstrating the application of machine learning algorithms to solve real-world problems in spam detection.

## Machine Learning Techniques

### Algorithms Used

1. **Random Forest Classifier**:
   - A popular ensemble learning method used for classification tasks. It operates by constructing multiple decision trees during training and outputs the mode of the classes for classification.
   - **Advantages**: Handles large datasets with higher dimensionality, prevents overfitting, and provides an estimate of feature importance.

2. **Support Vector Machine (SVM)**:
   - A supervised machine learning algorithm that finds the hyperplane that best separates different classes. SVMs are effective in high-dimensional spaces and work well with a clear margin of separation.
   - **Advantages**: Effective in high-dimensional spaces, robust to overfitting, especially in high-dimensional data.

3. **Logistic Regression**:
   - A linear model for binary classification that predicts the probability of an outcome falling into one of two classes. It uses a logistic function to model the probability.
   - **Advantages**: Simple to implement, interpretable, works well for binary classification with linearly separable data.

### Machine Learning Techniques and AI

- **Feature Engineering**: The dataset is preprocessed by scaling numerical features to ensure that all features contribute equally to the model.
- **Model Training**: Various models are trained on the dataset to identify the best-performing algorithm.
- **Hyperparameter Tuning**: Parameters such as `n_estimators` and `max_depth` in Random Forests are tuned to optimize the model's performance.
- **Cross-Validation**: The model's performance is evaluated using techniques like cross-validation to ensure it generalizes well to unseen data.

## Model Training and Evaluation

The project follows these steps for training and evaluating the model:

1. **Data Loading**: The dataset is loaded and split into training and test sets using `train_test_split` to ensure that the class distribution remains consistent across both sets.
2. **Feature Scaling**: Features are standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
3. **Model Training**: Various machine learning models are trained on the training data. For instance, `RandomForestClassifier` is trained with `class_weight='balanced'` to handle any class imbalance.
4. **Model Evaluation**: The trained model is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score. The confusion matrix provides insight into the types of errors made by the model.

### Evaluation Metrics

- **Accuracy**: Proportion of correctly predicted instances.
- **Precision**: Proportion of true positive predictions relative to the total positive predictions.
- **Recall**: Proportion of true positive predictions relative to the total actual positives.
- **F1-score**: Harmonic mean of precision and recall, providing a single metric that balances both.
- **ROC-AUC Score**: Area under the receiver operating characteristic curve, indicating the model's ability to discriminate between positive and negative classes.

## Usage

To use this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/touro3/SPAM-OR-NOT.git
   cd SPAM-OR-NOT
2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Train the Model**:
   ```bash
   python main.py

### Setup and Installation
- Follow these steps to set up the environment and install all necessary dependencies:

1. Clone the repository to your local machine.
2. Create a virtual environment and activate it.
3. Install dependencies using pip install -r requirements.txt.
4. Run the main.py script to train the model and save it as model.pkl.
