import json
from collections import Counter
from datasets import load_dataset

# 1. Load the ATIS dataset
ds = load_dataset("tuetschek/atis")

# 2. Extract intent strings and texts across both splits
train_intents = ds["train"]["intent"]
test_intents = ds["test"]["intent"]
all_intents = list(train_intents) + list(test_intents)

train_texts = ds["train"]["text"]
test_texts = ds["test"]["text"]
all_texts = list(train_texts) + list(test_texts)

# 3. Collect full intent counts (including compound labels as single classes)
class_distribution = Counter(all_intents)
sorted_classes = sorted(class_distribution.keys())
total_unique_classes = len(sorted_classes)

# 4. Compute average utterance word count using text.split() across all utterances
total_words = sum(len(text.split()) for text in all_texts)
average_word_count = round(total_words / len(all_texts), 2) if all_texts else 0.0

# 5. Update JSON structure
try:
    with open("eda_summary.json", "r") as f:
        eda = json.load(f)
except FileNotFoundError:
    eda = {}

eda["dataset"] = "ATIS"
eda["source"] = "https://huggingface.co/datasets/tuetschek/atis"
eda["row_count"] = len(all_texts)
eda["splits"] = [
    {"name": "train", "row_count": len(train_texts)},
    {"name": "test", "row_count": len(test_texts)}
]
eda["classes"] = sorted_classes
eda["class_count"] = total_unique_classes
eda["class_distribution"] = dict(class_distribution)
eda["average_message_word_count"] = average_word_count

# Save updated JSON
with open("eda_summary.json", "w") as f:
    json.dump(eda, f, indent=2)