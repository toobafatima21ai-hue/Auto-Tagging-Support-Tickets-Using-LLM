# few_shot.py
import json
import re
import ollama
from config import CATEGORIES, FEW_SHOT_EXAMPLES, OLLAMA_MODEL

SYSTEM_PROMPT = f"""You are a support ticket classifier. Given a ticket, output the
TOP 3 most relevant tags from this exact list (use these labels verbatim):
{", ".join(CATEGORIES)}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"tags": [{{"label": "...", "confidence": 0.0}}, {{"label": "...", "confidence": 0.0}}, {{"label": "...", "confidence": 0.0}}]}}

Confidences should sum to roughly 1.0 and reflect your certainty.
"""


def build_few_shot_prompt(text: str) -> str:
    examples_block = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_block += f'\nTicket: "{ex["text"]}"\nTags: {ex["tags"]}\n'
    return f"{examples_block}\nTicket: \"{text}\"\nTags:"


def tag_ticket_few_shot(text: str, top_k: int = 3):
    prompt = build_few_shot_prompt(text)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    content = response["message"]["content"]

    # Robust JSON extraction in case the model adds stray text
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return [("General Inquiry", 0.0)]  # fallback

    try:
        data = json.loads(match.group())
        pairs = [(t["label"], float(t["confidence"])) for t in data["tags"]]
        # keep only valid categories, in case model hallucinates a label
        pairs = [p for p in pairs if p[0] in CATEGORIES]
        return pairs[:top_k] if pairs else [("General Inquiry", 0.0)]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [("General Inquiry", 0.0)]


if __name__ == "__main__":
    sample = "Can I get a refund, I cancelled my plan but was still charged?"
    for label, score in tag_ticket_few_shot(sample):
        print(f"{label}: {score:.3f}")
