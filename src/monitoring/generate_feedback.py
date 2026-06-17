import random
from src.monitoring.feedback import log_prediction


# -------------------------
# Templates
# -------------------------
templates = [
    "this movie was {}",
    "i {} this film",
    "it was a {} experience",
    "a {} movie overall",
    "absolutely {} performance",
    "one of the most {} films i have seen",
    "the story was {}",
    "acting was {} and engaging"
]

positive_words = [
    "amazing", "great", "fantastic", "excellent",
    "wonderful", "brilliant", "impressive", "outstanding"
]

negative_words = [
    "bad", "terrible", "boring", "awful",
    "poor", "mediocre", "disappointing", "weak"
]

verbs_positive = ["loved", "enjoyed", "liked", "admired"]
verbs_negative = ["hated", "disliked", "regretted"]


# -------------------------
# Text generator
# -------------------------
def generate_text(label: str):

    template = random.choice(templates)

    if label == "positive":
        word = random.choice(positive_words)
        verb = random.choice(verbs_positive)
    else:
        word = random.choice(negative_words)
        verb = random.choice(verbs_negative)

    text = template.format(word)

    if random.random() > 0.5:
        text = f"i {verb} it - {text}"

    return text


# -------------------------
# Main generator
# -------------------------
def generate_feedback(n=300, noise=0.15):

    print(f"Generating {n} feedback samples...")

    for _ in range(n):

        label = random.choice(["positive", "negative"])

        text = generate_text(label)

        # simulate noise
        if random.random() < noise:
            prediction = "negative" if label == "positive" else "positive"
        else:
            prediction = label

        confidence = round(random.uniform(0.55, 0.99), 2)

        log_prediction(
            text=text,
            prediction=prediction,
            true_label=label,
            confidence=confidence
        )

    print("Feedback generation completed.")


if __name__ == "__main__":
    generate_feedback()