import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.utils.logger import setup_logger
from backend.utils.config_loader import load_config

logger = setup_logger(__name__)


class TrustEngine:
    """
    Combines price anomaly, NLP fraud, and image quality signals
    into a unified 0-100 Trust Score with explanation.
    """

    def __init__(self):
        self.config = load_config()
        self.weights = self.config["trust_weights"]

    def compute_trust_score(self, predicted_price: float, listed_price: float,
                            nlp_result: dict, image_result: dict = None) -> dict:

        price_risk, price_detail = self._price_risk(predicted_price, listed_price)
        nlp_risk, nlp_detail = self._nlp_risk(nlp_result)
        image_risk, image_detail = self._image_risk(image_result)

        weighted_risk = (
            price_risk * self.weights["price_anomaly"] +
            nlp_risk * self.weights["nlp_fraud"] +
            image_risk * self.weights["image_quality"]
        )

        trust_score = round(100 - weighted_risk, 1)
        tier = self._classify_tier(trust_score)
        explanations = self._build_explanations(price_detail, nlp_detail, image_detail)

        return {
            "trust_score": trust_score,
            "tier": tier,
            "component_risks": {
                "price_anomaly_risk": round(price_risk, 1),
                "nlp_fraud_risk": round(nlp_risk, 1),
                "image_quality_risk": round(image_risk, 1)
            },
            "explanations": explanations,
            "weights_used": self.weights
        }

    # ------------------------------------------------------------------ #
    # Price Anomaly Risk
    # ------------------------------------------------------------------ #
    def _price_risk(self, predicted_price: float, listed_price: float):
        if predicted_price <= 0:
            return 0.0, {"deviation_pct": 0, "direction": "unknown"}

        deviation_pct = (listed_price - predicted_price) / predicted_price

        # Risk increases the further the listed price deviates from prediction
        # Both overpriced AND underpriced listings carry risk (underpriced = scam bait)
        abs_dev = abs(deviation_pct)

        if abs_dev <= 0.10:
            risk = 0
        elif abs_dev <= 0.25:
            risk = 30
        elif abs_dev <= 0.40:
            risk = 60
        else:
            risk = 90

        direction = "underpriced" if deviation_pct < 0 else "overpriced"

        return float(risk), {
            "deviation_pct": round(deviation_pct * 100, 1),
            "direction": direction,
            "predicted_price": round(predicted_price, 2),
            "listed_price": listed_price
        }

    # ------------------------------------------------------------------ #
    # NLP Fraud Risk — already 0-100 from the detector
    # ------------------------------------------------------------------ #
    def _nlp_risk(self, nlp_result: dict):
        if nlp_result is None:
            return 0.0, {"available": False}

        risk = nlp_result.get("rule_fraud_score", 0)
        return float(risk), {
            "available": True,
            "score": risk,
            "scam_phrases": nlp_result.get("scam_phrases_found", []),
            "urgency_phrases": nlp_result.get("urgency_phrases_found", [])
        }

    # ------------------------------------------------------------------ #
    # Image Quality Risk — inverted condition score
    # ------------------------------------------------------------------ #
    def _image_risk(self, image_result: dict):
        if image_result is None or "error" in (image_result or {}):
            return 0.0, {"available": False}

        condition_score = image_result.get("image_condition_score", 100)
        risk = 100 - condition_score
        return float(risk), {
            "available": True,
            "condition_score": condition_score,
            "verdict": image_result.get("verdict", "")
        }

    # ------------------------------------------------------------------ #
    # Risk Tier Classification
    # ------------------------------------------------------------------ #
    def _classify_tier(self, trust_score: float) -> str:
        if trust_score >= 75:
            return "TRUSTWORTHY"
        elif trust_score >= 50:
            return "CAUTION"
        elif trust_score >= 30:
            return "HIGH RISK"
        else:
            return "LIKELY SCAM"

    # ------------------------------------------------------------------ #
    # Plain-English Explanations
    # ------------------------------------------------------------------ #
    def _build_explanations(self, price_detail, nlp_detail, image_detail) -> list:
        reasons = []

        # Price explanation
        dev = price_detail.get("deviation_pct", 0)
        if abs(dev) > 10:
            direction = price_detail["direction"]
            severity = "HIGH" if abs(dev) > 25 else "MEDIUM"
            reasons.append({
                "flag": "PRICE_ANOMALY",
                "severity": severity,
                "message": f"Listed price is {abs(dev):.1f}% {direction} compared to "
                          f"the predicted fair value of ${price_detail['predicted_price']:,.0f}.",
                "suggestion": (
                    "Significantly underpriced listings can be a scam tactic to "
                    "attract quick interest. Verify with the seller why the price "
                    "is below market."
                ) if direction == "underpriced" else (
                    "This listing is priced above market average. Negotiate or "
                    "compare with similar listings before proceeding."
                )
            })
        else:
            reasons.append({
                "flag": "PRICE_ALIGNED",
                "severity": "NONE",
                "message": f"Listed price is within {abs(dev):.1f}% of predicted fair value — "
                          f"consistent with market expectations.",
                "suggestion": None
            })

        # NLP explanation
        if nlp_detail.get("available") and nlp_detail.get("score", 0) > 0:
            scam_phrases = nlp_detail.get("scam_phrases", [])
            urgency = nlp_detail.get("urgency_phrases", [])
            severity = "HIGH" if nlp_detail["score"] >= 60 else "MEDIUM"

            msg_parts = []
            if scam_phrases:
                msg_parts.append(f"contains known scam phrases ({', '.join(scam_phrases[:3])})")
            if urgency:
                msg_parts.append(f"uses urgency language ({', '.join(urgency[:2])})")

            reasons.append({
                "flag": "NLP_FRAUD_SIGNAL",
                "severity": severity,
                "message": f"Description {' and '.join(msg_parts)}.",
                "suggestion": "Be cautious of sellers using pressure tactics or "
                              "common scam phrasing. Avoid advance payments."
            })
        elif nlp_detail.get("available"):
            reasons.append({
                "flag": "NLP_CLEAN",
                "severity": "NONE",
                "message": "No fraud-indicating language detected in description.",
                "suggestion": None
            })

        # Image explanation
        if image_detail.get("available"):
            condition_score = image_detail.get("condition_score", 100)
            if condition_score < 50:
                reasons.append({
                    "flag": "IMAGE_QUALITY_LOW",
                    "severity": "MEDIUM",
                    "message": f"Listing images have quality issues "
                              f"({image_detail.get('verdict', '')}).",
                    "suggestion": "Request clearer, well-lit photos before proceeding."
                })

        return reasons


if __name__ == "__main__":
    import json

    engine = TrustEngine()

    # Simulated scenario: a scam-like listing
    nlp_sample = {
        "rule_fraud_score": 75,
        "scam_phrases_found": ["army", "god fearing", "escrow"],
        "urgency_phrases_found": ["urgent", "today only"]
    }

    image_sample = {"image_condition_score": 85, "verdict": "GOOD IMAGE QUALITY"}

    result = engine.compute_trust_score(
        predicted_price=18000,
        listed_price=9500,  # heavily underpriced — classic scam bait
        nlp_result=nlp_sample,
        image_result=image_sample
    )

    print(json.dumps(result, indent=2))
