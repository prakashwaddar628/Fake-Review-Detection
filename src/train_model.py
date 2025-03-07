import pandas as pd

# Load the dataset
df = pd.read_csv("./datasets/fake_reviews_dataset.csv")

# Inspect the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Encode labels ('OR' -> 0, 'CG' -> 1)
df['label'] = df['label'].map({'OR': 0, 'CG': 1})

# Drop missing values
df.dropna(subset=['text_'], inplace=True)

# Save cleaned dataset
df.to_csv("./datasets/cleaned_fake_reviews.csv", index=False)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("./datasets/cleaned_fake_reviews.csv")

# Split dataset
X = df['text_']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save the vectorizer for later use
import joblib
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train models
nb_model = MultinomialNB()
svm_model = SVC(kernel='linear', probability=True)
lr_model = LogisticRegression()

nb_model.fit(X_train_vec, y_train)
svm_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

# Evaluate models
models = {'Naive Bayes': nb_model, 'SVM': svm_model, 'Logistic Regression': lr_model}

for name, model in models.items():
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

# Save best model (SVM in this case)
joblib.dump(svm_model, "./models/svm_fake_review_model.pkl")


from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Train RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30)
rf_model.fit(X_train_vec, y_train)
joblib.dump(rf_model, "./models/random_forest_model.pkl")

# Train LSTM model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=100, input_length=100),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)
lstm_model.save("./models/lstm_model.h5")


from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

print("\nSupport Vector Machine:")
evaluate_model(svm_model, X_test_vec, y_test)

print("\nRandom Forest:")
evaluate_model(rf_model, X_test_vec, y_test)


