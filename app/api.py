from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
model = joblib.load("./models/svm_fake_review_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['review']
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return jsonify({'prediction': 'Fake Review' if prediction == 1 else 'Real Review'})

if __name__ == '__main__':
    app.run(debug=True)
