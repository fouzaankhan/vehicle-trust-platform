import numpy as np
import joblib
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

class DuplicateDetector:
    MODEL_NAME = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.92

    def __init__(self, load_model=True):
        self.model = None

        if load_model:
            logger.info(f"Loading sentence transformer: {self.MODEL_NAME}")
            self.model = get_sentence_transformer()

        self.known_embeddings = None
        self.known_ids = []


    def build_index(self, descriptions: list, listing_ids: list):
        """Call once to index all known listings."""
        logger.info(f"Building embedding index for {len(descriptions)} listings...")
        self.known_embeddings = self.model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=True
        )
        self.known_ids = listing_ids
        logger.info("Index built.")

    def save_index(self, path="models/description_embeddings.npy",
                   ids_path="models/description_ids.joblib"):
        np.save(path, self.known_embeddings)
        joblib.dump(self.known_ids, ids_path)
        logger.info(f"Index saved to {path}")

    def load_index(self, path="models/description_embeddings.npy",
                   ids_path="models/description_ids.joblib"):
        self.known_embeddings = np.load(path)
        self.known_ids = joblib.load(ids_path)
        logger.info(f"Loaded index: {len(self.known_ids)} entries")

    def check_duplicate(self, description: str) -> dict:
        if self.known_embeddings is None:
            return {"is_duplicate": False, "reason": "No index loaded."}

        if self.model is None:
            self.model = get_sentence_transformer()

        new_emb = self.model.encode([description])

        sims = cosine_similarity(new_emb, self.known_embeddings)[0]
        max_sim = float(sims.max())
        max_idx = int(sims.argmax())

        is_dup = max_sim >= self.SIMILARITY_THRESHOLD

        return {
            "is_duplicate": is_dup,
            "max_similarity": round(max_sim, 4),
            "most_similar_listing_id": self.known_ids[max_idx] if is_dup else None,
            "duplicate_score": round(max_sim * 100, 1)
        }