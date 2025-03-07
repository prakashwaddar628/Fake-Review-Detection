import joblib

vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
model = joblib.load("./models/svm_fake_review_model.pkl")

def predict_review(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Fake Review" if prediction == 1 else "Real Review"

while True:
    review = input("Enter a review (or 'exit' to quit): ")
    if review.lower() == 'exit':
        break
    print(f"Prediction: {predict_review(review)}")
