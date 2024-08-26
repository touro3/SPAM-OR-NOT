from SRC.data_loader import load_data
from SRC.preprocessing import preprocess_data
from SRC.model import train_model, save_model, load_model
from SRC.evaluate import evaluate_model

# Load data
df = load_data('DATA/spambase.data')

# Preprocess data
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

# Train the model
model = train_model(X_train_scaled, y_train, model_type='RandomForest')

# Save the trained model
save_model(model, 'model.pkl')

print("Model trained and saved successfully.")

# Load and evaluate the model
loaded_model = load_model('model.pkl')
evaluate_model(loaded_model, X_test_scaled, y_test)