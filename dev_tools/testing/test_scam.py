import requests, json
payload = {
    'make': 'Ford', 'model_name': 'f-150', 'year': 2018, 'km_driven': 45000,
    'listed_price': 9000,
    'transmission': 'automatic', 'condition': 35.0, 'sale_month': 6,
    'description': 'URGENT SALE. Posted abroad with army. God fearing seller. Escrow payment only. Today only, call now!'
}
r = requests.post('http://localhost:8000/analyze/full', json=payload)
print(json.dumps(r.json(), indent=2))