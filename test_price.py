from src.models.predict import PricePredictor

predictor = PricePredictor()

result = predictor.predict({
    "make": "Ford",
    "model": "f-150",
    "year": 2018,
    "km_driven": 45000,
    "transmission": "automatic",
    "condition": 35.0,
    "sale_month": 6
})

print(result)