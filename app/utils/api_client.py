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
    
def upload_image(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        r = requests.post(f"{BASE_URL}/analyze/image", files=files, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def analyze_full_listing(make, model_name, year, km_driven, listed_price,
                         transmission, condition, sale_month, description,
                         image_filename=None):
    payload = {
        "make": make,
        "model_name": model_name,
        "year": year,
        "km_driven": km_driven,
        "listed_price": listed_price,
        "transmission": transmission,
        "condition": condition,
        "sale_month": sale_month,
        "description": description,
        "image_filename": image_filename
    }
    try:
        r = requests.post(f"{BASE_URL}/analyze/full", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}