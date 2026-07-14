# 🎫 Support Ticket Auto-Tagger

Automatic multi-label classification of free-text support tickets using LLMs — comparing zero-shot, few-shot, and fine-tuned approaches, with a full Streamlit interface for interactive and batch tagging.

---

## Project Overview

This project automatically tags customer support tickets into relevant categories using Large Language Models. Instead of relying on a single modeling approach, it implements and compares three distinct LLM-based strategies — zero-shot classification, few-shot prompting, and supervised fine-tuning — to demonstrate a practical understanding of when and why each approach is appropriate. The system outputs the top 3 most probable tags per ticket, reflecting the reality that support tickets are often ambiguous and can reasonably belong to more than one category.

---

## Problem Statement

Support teams receive high volumes of free-text tickets daily — billing complaints, bug reports, feature requests, account issues, and more. Manually reading and routing each ticket to the correct team is slow, expensive, and inconsistent across agents. This creates delays in response time and increases operational overhead for growing support teams.

**Goal:** Build a system that automatically reads a ticket's raw text and predicts its most likely categories, enabling automatic routing and giving support agents an assistive first-pass suggestion instead of a blank slate.

---

## Solution Approach

The project tackles the classification problem from three angles, in increasing order of task-specific adaptation:

1. **Zero-Shot Classification** — uses a pretrained Natural Language Inference (NLI) model to classify tickets against category labels with no training or labeled examples.
2. **Few-Shot Prompting** — uses a local LLM guided by a small set of hand-picked labeled examples embedded directly in the prompt, requiring no model training.
3. **Fine-Tuning** — trains a transformer classifier directly on labeled ticket data, learning task-specific patterns unavailable to the other two approaches.

All three methods are evaluated on the same held-out test set so their trade-offs (accuracy vs. setup cost vs. data requirements) can be directly compared.

---

## Key Features

- 🏷️ Top-3 tag prediction per ticket with confidence scores
- ⚖️ Side-by-side comparison of zero-shot, few-shot, and fine-tuned outputs
- 🧠 Prompt-engineered few-shot classification with structured JSON output parsing
- 📊 Quantitative evaluation: Top-1 accuracy and Top-3 hit rate
- 🖥️ Interactive Streamlit interface with single-ticket and batch CSV tagging
- 📈 Built-in analytics dashboard (tag distribution, confidence histograms)
- ⬇️ Downloadable tagged results as CSV
- 🔌 Fully open-source stack — no paid APIs required
  ---

## Tech Stack

| Layer | Tools |
|---|---|
| Zero-shot modeling | HuggingFace Transformers, `facebook/bart-large-mnli` |
| Few-shot prompting | Ollama, `phi3:mini` (local LLM) |
| Fine-tuning | HuggingFace Transformers, `distilbert-base-uncased`, PyTorch, Datasets |
| Evaluation | scikit-learn, pandas |
| Interface | Streamlit, Plotly |
| Data handling | pandas, NumPy |

---

## Project Workflow

1. **Generate/collect data** — build or import a labeled ticket dataset (`text`, `true_tag`)
2. **Zero-shot tagging** — score tickets against category labels via NLI, no training
3. **Few-shot tagging** — prompt a local LLM with curated examples, parse structured output
4. **Fine-tuning** — split data 80/20, tokenize, train DistilBERT as a 10-class classifier
5. **Evaluation** — compare zero-shot vs. fine-tuned on the same held-out test set
6. **Deployment** — serve all three methods through a Streamlit interface for interactive and batch use

---

## Dataset Information

This project uses a **synthetically generated dataset** (`generate_dataset.py`), built from category-specific templates with randomized placeholders to produce realistic, varied ticket phrasing. This was a deliberate choice over a downloaded dataset:

1. Publicly available free-text ticket datasets (e.g. on Kaggle) have inconsistent licensing and aren't reliably scriptable to fetch automatically.
2. A generator makes the project fully reproducible — anyone cloning the repo gets the same data with one command, no manual downloads or credentials needed.

**Dataset stats:** 600 tickets across 10 categories (60 per category), columns `ticket_id`, `text`, `true_tag`.

**Categories:** Billing & Payments · Technical Issue · Account Access · Feature Request · Bug Report · Shipping & Delivery · Refund & Cancellation · General Inquiry · Complaint · Subscription Management

**Using a real dataset instead:** swap `data/tickets.csv` for any CSV with `text` and `true_tag` columns — no other code changes needed.

---

## Model Development

- **Zero-shot:** `facebook/bart-large-mnli` reframes classification as entailment — checking whether the ticket text "entails" each candidate label, ranked by entailment strength. No training data used.
- **Few-shot:** `phi3:mini` (via Ollama) is prompted with 5 hand-labeled example tickets embedded in the system prompt, then asked to classify new tickets in the same structured JSON format, which is parsed and validated against the known category list.
- **Fine-tuned:** `distilbert-base-uncased` is trained as a 10-class sequence classifier on 80% of the dataset (4 epochs, learning rate 2e-5, batch size 16), with the remaining 20% held out for evaluation.

---

## Results & Performance

> Run `python evaluate.py` after training to populate this table with your own numbers — no benchmark figures are hardcoded in this project.

| Method | Top-1 Accuracy | Top-3 Hit Rate |
|---|---|---|
| Zero-Shot (BART-MNLI) | _run to populate_ | _run to populate_ |
| Fine-Tuned (DistilBERT) | _run to populate_ | _run to populate_ |

**Expected pattern:** zero-shot needs no labeled data and adapts instantly to new/renamed categories, but is slower (one inference pass per candidate label per ticket) and generally less accurate. Fine-tuning requires labeled data and a training step up front, but is faster at inference and typically more accurate once trained on domain-specific tickets.

---
---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Zero-shot model requires no training data but is slow (one NLI pass per candidate label per ticket) | Sampled a subset of the test set for zero-shot evaluation to keep benchmarking practical |
| Few-shot LLM output isn't naturally structured | Constrained the prompt to require strict JSON output and added regex-based extraction with fallback handling for malformed responses |
| Publicly available ticket datasets have inconsistent licensing and aren't scriptable to fetch directly | Built a template-based synthetic dataset generator for full reproducibility, with the option to swap in a real dataset later |
| Hugging Face Hub unreachable on some networks (DNS/ISP-level blocking) | Added automatic fallback to the `hf-mirror.com` endpoint, plus support for loading models from a manually downloaded local folder |
| Large model downloads (e.g. `model.safetensors`) failing mid-transfer on unstable connections | Used resumable downloads (`curl -C -`) to continue partial downloads instead of restarting from scratch |

---

## Future Enhancements

- Benchmark against a real-world labeled dataset (e.g. Kaggle Customer Support Ticket Dataset) alongside the synthetic one
- Extend to true multi-label ground truth (currently single true label per ticket, with top-3 predicted for ranking)
- Experiment with larger open-source LLMs (Llama 3, Mistral) for few-shot comparison
- Add an active learning loop: route low-confidence predictions to human review, feed corrections back into fine-tuning data
- Deploy the Streamlit app to Streamlit Community Cloud or Hugging Face Spaces for a live public demo

---

## Author

**Tooba Fatima**
AI Engineering Undergraduate, Superior University, Faisalabad, Pakistan

---

## Contact Information

- **GitHub:** [github.com/toobafatima21ai-hue](https://github.com/toobafatima21ai-hue)
