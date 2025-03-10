from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(base_dir, "../models/vectorizer.pkl")
model_path = os.path.join(base_dir, "../models/model.pkl")

# Debugging: Print available files
print("Available files in models directory:", os.listdir(os.path.dirname(vectorizer_path)))

# Check if files exist
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    raise FileNotFoundError(f"Model or vectorizer file not found in {os.path.dirname(vectorizer_path)}")

# Load the vectorizer and model
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'review' not in data:
        return jsonify({'error': 'No review text provided'}), 400

    text = data['review']
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    
    return jsonify({'prediction': 'Fake Review' if prediction == 1 else 'Real Review'})

if __name__ == '__main__':
    app.run(debug=True)
