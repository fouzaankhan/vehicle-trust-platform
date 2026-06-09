import sys
print("Python:", sys.executable)

from src.models.duplicate_detector import DuplicateDetector
print("DuplicateDetector imported")

import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.duplicate_detector import DuplicateDetector

df = pd.read_csv("data/processed/listing_descriptions.csv")

detector = DuplicateDetector()
detector.build_index(
    descriptions=df["description"].tolist(),
    listing_ids=df["listing_id"].tolist()
)
detector.save_index()
print("Embedding index built and saved.")