# fine_tune.py
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from config import CATEGORIES, MODEL_PATH

# BASE_MODEL = "distilbert-base-uncased"
BASE_MODEL = "local_models/distilbert-base-uncased"


def main():
    df = pd.read_csv("data/tickets.csv")

    le = LabelEncoder()
    le.fit(CATEGORIES)
    df["label"] = le.transform(df["true_tag"])

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df.to_csv("data/train_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(CATEGORIES)
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # save label encoder mapping
    with open(f"{MODEL_PATH}/label_map.json", "w") as f:
        json.dump({str(i): c for i, c in enumerate(le.classes_)}, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
