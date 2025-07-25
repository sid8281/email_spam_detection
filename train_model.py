import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("emails.csv")

# Optional: Convert labels to lowercase
df['label'] = df['label'].str.lower()

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1, 3))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("🎉 Model and vectorizer saved to /model/")
