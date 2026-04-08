import requests

url = "http://127.0.0.1:5000/api/predict"

data = {
    "message": "You won a lottery!!!"
}

response = requests.post(url, json=data)

print(response.json())