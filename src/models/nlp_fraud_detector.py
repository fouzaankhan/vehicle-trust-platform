import re
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NLPFraudDetector:
    """
    Two-layer NLP fraud detection:
    Layer 1 — Rule-based: instant, explainable, no model needed
    Layer 2 — Semantic: sentence embeddings for duplicate detection
    """

    SCAM_PHRASES = [
        "army", "posted abroad", "god fearing", "escrow",
        "shifting abroad", "moving overseas", "urgent sale",
        "below market", "100% genuine", "serious buyers only",
        "whatsapp only", "advance payment", "bank transfer only",
        "delivered after payment", "private photos", "medical emergency",
        "army deployment", "ship via", "trust me"
    ]

    URGENCY_PHRASES = [
        "asap", "today only", "call now", "limited time",
        "urgent", "immediately", "last chance", "selling fast"
    ]

    CONTACT_PATTERNS = [
        r'\b\d{10}\b',           # phone number
        r'whatsapp',             # whatsapp mention
        r'\S+@\S+\.\S+',        # email
        r'telegram',             # telegram
    ]

    def analyze(self, description: str) -> dict:
        text = description.lower().strip()

        scam_hits = self._detect_scam_phrases(text)
        urgency_hits = self._detect_urgency(text)
        contact_hits = self._detect_contact_info(text)
        linguistic = self._linguistic_features(text)
        rule_score = self._compute_rule_score(scam_hits, urgency_hits, contact_hits)

        return {
            "rule_fraud_score": rule_score,
            "scam_phrases_found": scam_hits,
            "urgency_phrases_found": urgency_hits,
            "contact_info_found": contact_hits,
            "linguistic_features": linguistic,
            "verdict": self._verdict(rule_score),
            "explanation": self._explain(scam_hits, urgency_hits, contact_hits)
        }

    def _detect_scam_phrases(self, text):
        return [p for p in self.SCAM_PHRASES if p in text]

    def _detect_urgency(self, text):
        return [p for p in self.URGENCY_PHRASES if p in text]

    def _detect_contact_info(self, text):
        hits = []
        for pattern in self.CONTACT_PATTERNS:
            if re.search(pattern, text):
                hits.append(pattern)
        return hits

    def _linguistic_features(self, text):
        words = text.split()
        exclamations = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        return {
            "word_count": len(words),
            "exclamation_count": exclamations,
            "caps_ratio": round(caps_ratio, 3),
            "avg_word_length": round(avg_word_len, 2)
        }

    def _compute_rule_score(self, scam_hits, urgency_hits, contact_hits):
        score = 0
        score += len(scam_hits) * 20
        score += len(urgency_hits) * 10
        score += len(contact_hits) * 15
        return min(score, 100)

    def _verdict(self, score):
        if score >= 60:
            return "HIGH RISK"
        elif score >= 30:
            return "SUSPICIOUS"
        else:
            return "LOOKS LEGIT"

    def _explain(self, scam_hits, urgency_hits, contact_hits):
        reasons = []
        if scam_hits:
            reasons.append(
                f"Contains {len(scam_hits)} known scam phrase(s): "
                f"{', '.join(scam_hits[:3])}"
            )
        if urgency_hits:
            reasons.append(
                f"Uses urgency language: {', '.join(urgency_hits[:3])}"
            )
        if contact_hits:
            reasons.append(
                "Attempts to move communication off-platform "
                "(phone/WhatsApp/email in description)"
            )
        if not reasons:
            reasons.append("No significant fraud signals detected in description.")
        return reasons