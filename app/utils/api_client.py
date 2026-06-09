import requests

BASE_URL = "http://localhost:8000"

def predict_price(make, model_name, year, km_driven,
                  transmission, condition, sale_month):
    payload = {
        "make": make,
        "model_name": model_name,
        "year": year,
        "km_driven": km_driven,
        "transmission": transmission,
        "condition": condition,
        "sale_month": sale_month
    }
    try:
        r = requests.post(f"{BASE_URL}/predict/price", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API not running. Start uvicorn first."}
    except Exception as e:
        return {"error": str(e)}

def health_check():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable"}