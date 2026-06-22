import requests
import json

payload = {
    "description": "URGENT SALE. I am posted abroad with army. God fearing seller. Escrow payment only. WhatsApp me immediately."
}

r = requests.post(
    "http://localhost:8000/analyze/description",
    json=payload
)

print("STATUS:", r.status_code)
print(json.dumps(r.json(), indent=2))