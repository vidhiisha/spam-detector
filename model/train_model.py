import sys
import os

# ✅ Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from utils.preprocess import transform_text

print("🔥 Training started...")

# ✅ Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])

# ✅ Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ✅ Preprocess text
df['text'] = df['text'].apply(transform_text)

# ✅ Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['label']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ Models to compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True)
}

best_model = None
best_accuracy = 0

print("\n📊 Model Comparison:\n")

# ✅ Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    print(f"{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print("----------------------")

    # ✅ Select best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print(f"\n🏆 Best Model Selected: {type(best_model).__name__}")

# ✅ Ensure model folder exists
os.makedirs('model', exist_ok=True)

# ✅ Save best model
pickle.dump(best_model, open('model/model.pkl', 'wb'))
pickle.dump(tfidf, open('model/vectorizer.pkl', 'wb'))

print("✅ Model trained and saved successfully!")