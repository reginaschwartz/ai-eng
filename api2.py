import json
import numpy as np
from collections import Counter
from datasets import load_dataset

# 1. Load the pinned dataset from Hugging Face
ds = load_dataset("tuetschek/atis")

# 2. Extract texts and intents across train and test splits
train_texts = ds["train"]["text"]
train_intents = ds["train"]["intent"]

test_texts = ds["test"]["text"]
test_intents = ds["test"]["intent"]

# Combine both splits
all_texts = list(train_texts) + list(test_texts)
all_intents = list(train_intents) + list(test_intents)

# 3. Calculate exact row counts
train_count = len(train_texts)
test_count = len(test_texts)
total_count = len(all_texts)

# 4. Collect complete combined intent vocabulary (including multi-intents/compound intents as exact strings)
# and compute total combined class distribution across all rows
class_counts = Counter(all_intents)
unique_classes = sorted(class_counts.keys())

# 5. Recompute the exact average word count using text.split() across all 5,871 utterances
word_counts = [len(text.split()) for text in all_texts]
avg_word_count = round(float(np.mean(word_counts)), 2)

# 6. Build the corrected summary object
eda_summary = {
    "dataset": "ATIS",
    "source": "https://huggingface.co/datasets/tuetschek/atis",
    "row_count": total_count,
    "splits": [
        {
            "name": "train",
            "row_count": train_count
        },
        {
            "name": "test",
            "row_count": test_count
        }
    ],
    "classes": unique_classes,
    "class_count": len(unique_classes),
    "class_distribution": dict(class_counts),
    "average_message_word_count": avg_word_count
}

# Write out to eda_summary.json
with open("eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2)