from fastapi import UploadFile, File
from src.models.image_analyzer import VehicleImageAnalyzer
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional
import sys
import os
from src.models.nlp_fraud_detector import NLPFraudDetector

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.predict import PricePredictor
from src.models.nlp_fraud_detector import NLPFraudDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Vehicle Trust Intelligence API",
    description="Predicts fair market price and analyzes listing fraud risk.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------------------------
# Load models once
# --------------------------------------------------

predictor = PricePredictor()
nlp_detector = NLPFraudDetector()

image_analyzer = VehicleImageAnalyzer()
os.makedirs("data/uploads", exist_ok=True)

# --------------------------------------------------
# Input Schemas
# --------------------------------------------------

class ListingInput(BaseModel):
    make: str
    model_name: str
    year: int
    km_driven: int
    transmission: Optional[str] = "automatic"
    condition: Optional[float] = 25.0
    sale_month: Optional[int] = 6

    @validator("year")
    def valid_year(cls, v):
        if v < 1995 or v > 2024:
            raise ValueError("Year must be between 1995 and 2024")
        return v

    @validator("km_driven")
    def valid_km(cls, v):
        if v < 0 or v > 500000:
            raise ValueError("km_driven must be between 0 and 500000")
        return v

    @validator("condition")
    def valid_condition(cls, v):
        if v is not None and (v < 1 or v > 49):
            raise ValueError("Condition must be between 1 and 49")
        return v

class DescriptionInput(BaseModel):
    description: str


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_version": "v1"
    }


@app.post("/predict/price")
def predict_price(listing: ListingInput):
    try:
        input_dict = {
            "make": listing.make,
            "model": listing.model_name,
            "year": listing.year,
            "km_driven": listing.km_driven,
            "transmission": listing.transmission,
            "condition": listing.condition,
            "sale_month": listing.sale_month
        }

        result = predictor.predict(input_dict)

        return {
            "status": "success",
            "predicted_price_usd": result["predicted_price"],
            "input_received": input_dict
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )


@app.post("/analyze/description")
def analyze_description(input: DescriptionInput):
    try:
        fraud_result = nlp_detector.analyze(input.description)

        return {
            "status": "success",
            "fraud_analysis": fraud_result
        }

    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="NLP analysis failed."
        )
    
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        save_path = f"data/uploads/{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = image_analyzer.analyze(save_path)
        return {"status": "success", "image_analysis": result}
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed.")