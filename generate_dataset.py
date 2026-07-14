# generate_dataset.py
import random
import csv
import os
from config import CATEGORIES

random.seed(42)

PRODUCTS = ["mobile app", "desktop software", "web dashboard", "subscription plan",
            "premium account", "API service", "checkout page", "notification settings"]
NAMES = ["John", "Ayesha", "Carlos", "Mei", "Fatima", "David", "Priya", "Omar"]

TEMPLATES = {
    "Billing & Payments": [
        "I was charged {amount} but my plan should only cost {amount2}, please explain.",
        "My credit card was billed twice for the same {product} order this month.",
        "There's an unrecognized charge of {amount} on my statement from your company.",
        "Can you clarify why my invoice this month is higher than usual?",
    ],
    "Technical Issue": [
        "The {product} keeps freezing whenever I try to save my work.",
        "I'm getting a 500 error every time I open the {product}.",
        "Nothing loads on the {product} since the latest update.",
        "The {product} is extremely slow and unresponsive today.",
    ],
    "Account Access": [
        "I can't log into my account even after resetting my password.",
        "It says my account is locked, but I don't know why.",
        "Two-factor authentication isn't sending me a code to sign in.",
        "I forgot my username and the recovery email isn't reaching me.",
    ],
    "Feature Request": [
        "Would it be possible to add dark mode to the {product}?",
        "It would be great if the {product} supported exporting to Excel.",
        "Can you add a bulk-edit option to the {product}?",
        "Please consider adding multi-language support to the {product}.",
    ],
    "Bug Report": [
        "The {product} crashes every time I try to upload a file.",
        "Dates are displaying incorrectly on the {product}.",
        "I found a bug where deleting an item duplicates it instead.",
        "The search bar on the {product} returns no results even for valid queries.",
    ],
    "Shipping & Delivery": [
        "My order was supposed to arrive {days} days ago and there's no tracking update.",
        "The package I received was damaged during shipping.",
        "I received the wrong item in my order, I ordered a different {product}.",
        "Delivery status hasn't changed from 'processing' for {days} days.",
    ],
    "Refund & Cancellation": [
        "I want to cancel my {product} and get a refund for this month.",
        "Please process a refund for the order I returned last week.",
        "How do I cancel my subscription before the next billing cycle?",
        "I was charged after I already cancelled my {product}.",
    ],
    "General Inquiry": [
        "What are the working hours for your customer support team?",
        "Do you offer student discounts on the {product}?",
        "How do I upgrade from the free plan to premium?",
        "Where can I find documentation for the {product}?",
    ],
    "Complaint": [
        "I'm really unhappy with the quality of support I've received so far.",
        "This is the third time I'm contacting you about the same {product} issue.",
        "Your customer service hung up on me twice today, this is unacceptable.",
        "I've been a loyal customer for years and this experience has been disappointing.",
    ],
    "Subscription Management": [
        "How do I switch from the monthly to the annual {product} plan?",
        "Can I pause my subscription for a couple of months?",
        "I want to add a second user seat to my current plan.",
        "How do I downgrade my {product} subscription?",
    ],
}


def fill(template):
    return template.format(
        product=random.choice(PRODUCTS),
        amount=f"${random.randint(10, 200)}",
        amount2=f"${random.randint(5, 100)}",
        days=random.randint(2, 10),
        name=random.choice(NAMES),
    )


def generate(n_per_category=60):
    rows = []
    ticket_id = 1
    for category, templates in TEMPLATES.items():
        for _ in range(n_per_category):
            text = fill(random.choice(templates))
            rows.append({"ticket_id": ticket_id, "text": text, "true_tag": category})
            ticket_id += 1
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    rows = generate()
    with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "text", "true_tag"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} tickets across {len(CATEGORIES)} categories -> data/tickets.csv")
