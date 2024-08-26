from SRC.data_loader import load_data
from SRC.preprocessing import preprocess_data
from SRC.model import train_model
from SRC.evaluate import evaluate_model

def main():
    # Load the data
    df = load_data('data/spambase.data')
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    
    # Train the model with the best hyperparameters
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    accuracy, report, cm = evaluate_model(model, X_test_scaled, y_test)
    
    # Optionally, you can save or log these metrics if needed
    # Example: Save metrics to a file
    with open('model_evaluation.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("Classification Report:\n")
        f.write(f"{report}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")

if __name__ == "__main__":
    main()
