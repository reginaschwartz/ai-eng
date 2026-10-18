import json
from collections import Counter
from datasets import load_dataset

# 1. Load the ATIS dataset
ds = load_dataset("tuetschek/atis")

# 2. Extract intent strings across both splits
train_intents = ds["train"]["intent"]
test_intents = ds["test"]["intent"]
all_intents = list(train_intents) + list(test_intents)

# 3. Keep every stored label whole, including multi-intent labels such as "flight+airfare".
label_counts = Counter(label.strip() for label in all_intents)
sorted_classes = sorted(label_counts)
total_unique_classes = len(sorted_classes)

# 4. Update JSON structure
try:
    with open("eda_summary.json", "r") as f:
        eda = json.load(f)
except FileNotFoundError:
    eda = {}

eda["classes"] = sorted_classes
eda["class_count"] = total_unique_classes
eda["class_distribution"] = dict(sorted(label_counts.items()))

# Save updated JSON
with open("eda_summary.json", "w") as f:
    json.dump(eda, f, indent=2)