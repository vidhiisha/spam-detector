from flask import Flask, render_template, request, jsonify
import pickle
from utils.preprocess import transform_text

app = Flask(__name__)

# ✅ Load trained model
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# ✅ Web UI prediction
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Preprocess
    transformed = transform_text(message)
    vector = vectorizer.transform([transformed])

    # Prediction
    result = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]

    if result == 1:
        prediction = f"Spam ❌ ({round(prob*100,2)}% confidence)"
    else:
        prediction = f"Not Spam ✅ ({round((1-prob)*100,2)}% confidence)"

    return render_template('result.html', prediction=prediction)

# ✅ API (VERY IMPORTANT 🔥)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json['message']

    transformed = transform_text(data)
    vector = vectorizer.transform([transformed])

    result = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]

    return jsonify({
        "prediction": "spam" if result == 1 else "not spam",
        "confidence": float(prob)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)