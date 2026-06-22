import requests
import json

payload = {
    "make": "Ford",
    "model_name": "f-150",
    "year": 2018,
    "km_driven": 45000,
    "listed_price": 18000,
    "transmission": "automatic",
    "condition": 35.0,
    "sale_month": 6,
    "description": "Well maintained 2018 Ford F-150. Single owner. Full service history. No accidents. Test drive welcome."
}

r = requests.post(
    "http://localhost:8000/analyze/full",
    json=payload
)

print(json.dumps(r.json(), indent=2))