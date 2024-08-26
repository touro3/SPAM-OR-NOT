# src/model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_model(model_type='RandomForest'):
    """
    Returns a machine learning model based on the model_type specified.
    
    Parameters:
    - model_type (str): The type of model to return. Options: 'RandomForest', 'SVM', 'LogisticRegression'.
    
    Returns:
    - model: A machine learning model instance.
    """
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'SVM':
        model = SVC(random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_model(X_train, y_train, model_type='RandomForest'):
    """
    Trains a machine learning model based on the model_type specified.
    
    Parameters:
    - X_train (ndarray): The training data features.
    - y_train (ndarray): The training data labels.
    - model_type (str): The type of model to train. Options: 'RandomForest', 'SVM', 'LogisticRegression'.
    
    Returns:
    - model: A trained machine learning model instance.
    """
    model = get_model(model_type)
    model.fit(X_train, y_train)
    return model
