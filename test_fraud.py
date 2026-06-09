from src.models.nlp_fraud_detector import NLPFraudDetector

detector = NLPFraudDetector()

text = """
URGENT SALE. I am posted abroad with army.
God fearing seller. Escrow payment only.
WhatsApp me immediately.
"""

result = detector.analyze(text)

print(result)