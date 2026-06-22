import requests

payload = {
    "make": "Ford",
    "model_name": "f-150",
    "year": 2023,
    "km_driven": 5000,
    "condition": 45
}

r = requests.post(
    "http://localhost:8000/predict/price",
    json=payload
)

print(r.status_code)
print(r.json())