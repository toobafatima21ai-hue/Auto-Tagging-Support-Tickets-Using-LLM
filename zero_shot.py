# zero_shot.py
from transformers import pipeline
from config import CATEGORIES, ZERO_SHOT_MODEL

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    return _classifier


def tag_ticket_zero_shot(text: str, top_k: int = 3):
    """Returns top_k (label, score) tuples for a single ticket."""
    clf = get_classifier()
    result = clf(text, CATEGORIES, multi_label=True)
    pairs = list(zip(result["labels"], result["scores"]))
    return pairs[:top_k]


if __name__ == "__main__":
    sample = "My package still hasn't arrived and tracking hasn't updated in 6 days."
    for label, score in tag_ticket_zero_shot(sample):
        print(f"{label}: {score:.3f}")
