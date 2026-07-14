# config.py

CATEGORIES = [
    "Billing & Payments",
    "Technical Issue",
    "Account Access",
    "Feature Request",
    "Bug Report",
    "Shipping & Delivery",
    "Refund & Cancellation",
    "General Inquiry",
    "Complaint",
    "Subscription Management",
]

# Hand-picked, diverse few-shot examples used to prompt the LLM
FEW_SHOT_EXAMPLES = [
    {
        "text": "I was charged twice for my monthly subscription this month, please help.",
        "tags": ["Billing & Payments", "Subscription Management", "Refund & Cancellation"],
    },
    {
        "text": "The app crashes every time I try to upload a photo from my gallery.",
        "tags": ["Bug Report", "Technical Issue", "Complaint"],
    },
    {
        "text": "I can't log into my account even after resetting my password twice.",
        "tags": ["Account Access", "Technical Issue", "General Inquiry"],
    },
    {
        "text": "Would it be possible to add dark mode to the mobile app?",
        "tags": ["Feature Request", "General Inquiry", "Technical Issue"],
    },
    {
        "text": "My order was supposed to arrive 5 days ago and there's still no tracking update.",
        "tags": ["Shipping & Delivery", "Complaint", "General Inquiry"],
    },
]

MODEL_PATH = "models/finetuned_ticket_classifier"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
OLLAMA_MODEL = "phi3:mini"
