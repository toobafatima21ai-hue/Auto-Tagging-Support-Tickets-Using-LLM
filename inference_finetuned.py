# inference_finetuned.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH

_tokenizer = None
_model = None
_label_map = None


def load():
    global _tokenizer, _model, _label_map
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        with open(f"{MODEL_PATH}/label_map.json") as f:
            _label_map = json.load(f)
    return _tokenizer, _model, _label_map


def tag_ticket_finetuned(text: str, top_k: int = 3):
    tokenizer, model, label_map = load()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)
    return [(label_map[str(i.item())], p.item()) for p, i in zip(top_probs, top_idxs)]


if __name__ == "__main__":
    sample = "The dashboard keeps timing out when I try to load reports."
    for label, score in tag_ticket_finetuned(sample):
        print(f"{label}: {score:.3f}")
