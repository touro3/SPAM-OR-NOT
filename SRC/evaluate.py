from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data and prints various performance metrics.

    Parameters:
    - model: The trained machine learning model.
    - X_test (ndarray): The testing data features.
    - y_test (ndarray): The true labels for the testing data.

    Returns:
    - accuracy (float): The accuracy of the model.
    - report (str): The classification report of the model.
    - cm (ndarray): The confusion matrix of the model.
    """
    y_pred = model.predict(X_test)
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, report, cm
