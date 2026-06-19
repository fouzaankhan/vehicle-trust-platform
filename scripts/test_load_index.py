from src.models.duplicate_detector import DuplicateDetector

dup = DuplicateDetector(load_model=False)
dup.load_index()

print("IDs:", len(dup.known_ids))
print("Shape:", dup.known_embeddings.shape)