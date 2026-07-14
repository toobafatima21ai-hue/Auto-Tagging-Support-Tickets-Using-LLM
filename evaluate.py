# evaluate.py
import pandas as pd
from tqdm import tqdm
from zero_shot import tag_ticket_zero_shot
from inference_finetuned import tag_ticket_finetuned


def evaluate(method_fn, df, name, sample_size=None):
    data = df if sample_size is None else df.sample(sample_size, random_state=42)
    top1_hits, top3_hits = 0, 0
    for _, row in tqdm(data.iterrows(), total=len(data), desc=name):
        preds = method_fn(row["text"], top_k=3)
        labels = [p[0] for p in preds]
        if labels and labels[0] == row["true_tag"]:
            top1_hits += 1
        if row["true_tag"] in labels:
            top3_hits += 1
    n = len(data)
    print(f"\n{name} — Top-1 accuracy: {top1_hits/n:.3f} | Top-3 hit rate: {top3_hits/n:.3f}")
    return top1_hits / n, top3_hits / n


if __name__ == "__main__":
    test_df = pd.read_csv("data/test_split.csv")

    # Zero-shot is slow (runs an NLI pass per label per ticket), so sample it
    evaluate(tag_ticket_zero_shot, test_df, "Zero-Shot (BART-MNLI)", sample_size=60)
    evaluate(tag_ticket_finetuned, test_df, "Fine-Tuned (DistilBERT)")
