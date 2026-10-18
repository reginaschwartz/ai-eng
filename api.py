import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from collections import Counter
from statistics import mean


def build_openai_client() -> OpenAI:
    load_dotenv(override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    return OpenAI()

dataset = load_dataset(
    "tuetschek/atis",
    revision="d44649dde31b4c9d2104aac42a35ed83956e7dec",
)
train = dataset["train"].to_pandas()
test = dataset["test"].to_pandas()
all_rows = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)


def word_count(text: pd.Series) -> pd.Series:
    return text.fillna("").str.split().str.len()


print("Rows")
print(f"  total: {len(all_rows)}")
for split, frame in (("train", train), ("test", test)):
    print(f"  {split}: {len(frame)} ({len(frame) / len(all_rows):.1%})")

intents = sorted(all_rows["intent"].unique())
print(f"\nIntent classes ({len(intents)})")
for intent in intents:
    print(f"  {intent}")

counts = all_rows["intent"].value_counts()
shares = counts / counts.sum()
print("\nExamples per intent")
for intent, count in counts.items():
    print(f"  {intent}: {count} ({shares[intent]:.1%})")

majority = counts.iloc[0]
minority = counts.iloc[-1]
print(
    f"\nImbalance: largest class is {shares.iloc[0]:.1%} of rows; "
    f"largest/smallest count is {majority / minority:.1f}x "
    f"({majority} vs {minority})."
)

lengths = word_count(all_rows["text"])
print("\nAverage utterance length (words)")
print(f"  overall: {lengths.mean():.2f}")
for split, frame in (("train", train), ("test", test)):
    print(f"  {split}: {word_count(frame['text']).mean():.2f}")


def normalize_intent(intent: str) -> str:
    return intent.strip().lower()


rows = [row for split in dataset.values() for row in split]
classes = sorted({normalize_intent(row["intent"]) for row in rows})
class_count = len(classes)
class_distribution = dict(
    sorted(Counter(normalize_intent(row["intent"]) for row in rows).items())
)
average_message_word_count = mean(len(row["text"].split()) for row in rows)

print("classes", classes)
print("class_count", class_count)
print("class_distribution", class_distribution)
print("average_message_word_count", average_message_word_count)



