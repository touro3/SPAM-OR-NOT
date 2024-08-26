from flask import Flask, request, jsonify
from SRC.model import load_model
import numpy as np

# Load your trained model
model = load_model('model.pkl')

# Initialize Flask app
app = Flask(__name__)

def preprocess_input(email_text):
    """
    Preprocess the input email text to convert it into the format required by the model.
    """
    # Example: Dummy vectorization process
    vectorized_email = np.array([0.5] * 57).reshape(1, -1)  # Replace with actual preprocessing steps
    return vectorized_email

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if an email is spam or not.
    """
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Preprocess the input
        email_text = data['email']
        input_features = preprocess_input(email_text)
        
        # Predict using the loaded model
        prediction = model.predict_proba(input_features)[0, 1]  # Probability of being spam
        
        # Return the result as a JSON response
        return jsonify({'spam_probability': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
