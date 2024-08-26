from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

def get_model(model_type='RandomForest'):
    """
    Returns a machine learning model based on the model_type specified.
    """
    if model_type == 'RandomForest':
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
    elif model_type == 'SVM':
        model = SVC(probability=True, class_weight='balanced', random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

def train_model(X_train, y_train, model_type='RandomForest'):
    """
    Trains a machine learning model based on the model_type specified.
    """
    model = get_model(model_type)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='model.pkl'):
    """
    Saves the trained model to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='model.pkl'):
    """
    Loads a trained model from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
