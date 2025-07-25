

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset (replace with your full path if needed)
df = pd.read_csv('spam_dataset.csv')  # Ensure 'text' and 'label' columns exist

# Optional: check class balance
print(df['label'].value_counts())  # 1 = Spam, 0 = Not Spam

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization with n-grams
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_proba = model.predict_proba(X_test_tfidf)[:, 1]
threshold = 0.65  # Custom threshold for spam (adjust based on ROC analysis)
y_pred = (y_proba >= threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump((vectorizer, model), 'model/spam_model.pkl')
print("\nModel + vectorizer saved as 'spam_model.pkl'")
