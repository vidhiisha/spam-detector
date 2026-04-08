# 🚀 AI Spam Detection System

A Machine Learning-based web application that detects whether a message is **Spam or Not Spam** using Natural Language Processing (NLP) and classification models.

---

## 📌 Features

* 🔍 Detects spam messages in real-time
* 🤖 Uses Machine Learning (Naive Bayes / Logistic Regression / SVM)
* 📊 Model comparison with accuracy & precision
* 🌐 Flask-based web application
* 🔗 REST API for external usage
* 🎨 Modern UI with confidence score

---

## 🧠 Tech Stack

### 💻 Backend

* Python
* Flask

### 🤖 Machine Learning

* Scikit-learn
* TF-IDF Vectorizer
* Naive Bayes, Logistic Regression, SVM

### 🧹 NLP

* NLTK (Stopwords, Text Processing)

### 🎨 Frontend

* HTML
* CSS (Modern UI)

### ⚙️ Others

* Pickle (Model saving)
* REST API
* Git & GitHub

---

## 🔄 How It Works

1. User inputs a message
2. Text is preprocessed (lowercase, remove stopwords)
3. Converted into vectors using TF-IDF
4. Model predicts spam or not spam
5. Output shown with confidence score

---

## 🧪 API Usage

### Endpoint:

```
POST /api/predict
```

### Request:

```json
{
  "message": "You won a lottery!!!"
}
```

### Response:

```json
{
  "prediction": "spam",
  "confidence": 0.95
}
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/vidhiisha/spam-detector.git
cd spam-detector

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python model/train_model.py
python app.py
```

---

## 🚀 Deployment

Deployed using **Render**
https://spam-detector-w8nf.onrender.com/

---

## 💬 Interview Summary

> Built an end-to-end spam detection system using NLP and Machine Learning, integrated with a Flask web application and exposed as a REST API with confidence scoring.

---

## 📌 Future Improvements

* 🔥 Add Deep Learning model (LSTM/BERT)
* 🌐 Integrate React frontend
* ☁️ Deploy with Docker
* 📊 Add dashboard for analytics

---

## 👩‍💻 Author

**Vidhi Sahu**
GitHub: https://github.com/vidhiisha

---
