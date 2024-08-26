from SRC.model import train_ensemble_model, save_model, load_model
from SRC.data_loader import load_data
from SRC.preprocessing import preprocess_data
from SRC.evaluate import evaluate_model

# Load and preprocess data
df = load_data('DATA/spambase.data')
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

# Train the ensemble model
voting_clf = train_ensemble_model(X_train_scaled, y_train)

# Save the ensemble model
save_model(voting_clf, 'voting_clf.pkl')

# Load and evaluate the model
loaded_model = load_model('voting_clf.pkl')
evaluate_model(loaded_model, X_test_scaled, y_test)
