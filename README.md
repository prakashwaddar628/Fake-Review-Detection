# Fake-Review-Detection

# 1. Preprocessing (preprocess.ipynb)

    Loaded the dataset: Read and explored the dataset.
    Data Cleaning:
    Removed duplicates and missing values.
    Standardized text (lowercasing, removing special characters, etc.).
    Text Processing:
    Tokenization, stemming, and lemmatization.
    Stopword removal.
    Feature Engineering:
    Converted text into numerical vectors using TF-IDF or CountVectorizer.
    Saved the processed dataset for training.

# 2. Training (train.ipynb)

    Loaded the preprocessed dataset.
    Split into Train & Test sets.
    Trained multiple models:
    Logistic Regression
    Random Forest
    Naïve Bayes
    LSTM (if applicable)
    Evaluated performance:
    Compared accuracy and selected the best model.
    If LSTM performed best, saved it as lstm_model.h5.
    Otherwise, saved the best traditional model as model.pkl along with vectorizer.pkl.

# 3. Prediction (predict.ipynb)

    Loaded the best model and vectorizer.
    Loaded test data (X_test and y_test).
    Made predictions.
    Evaluated model performance:
    Accuracy, classification report, confusion matrix.

# A hybrid model combines multiple weaker models to create a stronger, more robust prediction system. we will implement an Ensemble Hybrid Model using Stacking and Weighted Voting, which will include:

# 1.Base Models (Weak Learners):

    Logistic Regression
    Random Forest
    Naïve Bayes
    LSTM
# 2. Meta-Learner (Strong Model):

    A final model (e.g., XGBoost or another strong classifier) that takes predictions from the base models and makes the final decision.

# to run api 
    cd app
    python api.py
    1. open postman
    2. POST request
    3.