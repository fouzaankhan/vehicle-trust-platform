# test_nlp_api.py

import requests
import json

payload = {
    "description": "Well maintained 2018 Toyota Camry. Single owner. Full service history. No accidents. Test drive welcome."
}

r = requests.post(
    "http://localhost:8000/analyze/description",
    json=payload
)

print(json.dumps(r.json(), indent=2))