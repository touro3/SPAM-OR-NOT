from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

def train_individual_models(X_train, y_train):
    """
    Trains individual machine learning models.
    """
    models = {
        'RandomForest': get_model('RandomForest'),
        'SVM': get_model('SVM'),
        'LogisticRegression': get_model('LogisticRegression')
    }
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        models[model_name] = model
    
    return models

def create_voting_classifier(models):
    """
    Creates a VotingClassifier using the provided models.
    """
    voting_clf = VotingClassifier(estimators=[
        ('rf', models['RandomForest']),
        ('svc', models['SVM']),
        ('lr', models['LogisticRegression'])
    ], voting='soft')  # Use 'soft' voting for probability averaging
    
    return voting_clf

def train_ensemble_model(X_train, y_train):
    """
    Trains an ensemble model using VotingClassifier.
    """
    # Train individual models
    models = train_individual_models(X_train, y_train)
    
    # Create and train the VotingClassifier
    voting_clf = create_voting_classifier(models)
    print("Training VotingClassifier...")
    voting_clf.fit(X_train, y_train)
    
    return voting_clf

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

# Example usage
if __name__ == "__main__":
    # Assuming X_train, y_train are already defined
    voting_clf = train_ensemble_model(X_train, y_train)
    
    # Save the ensemble model
    save_model(voting_clf, 'voting_clf.pkl')
    
    print("Ensemble model saved successfully.")
