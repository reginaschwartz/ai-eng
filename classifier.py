import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from metrics import exact_match


load_dotenv(override=False)
client = OpenAI()
MODEL = "gpt-4.1-nano"


def load_single_labels(path: Path) -> list[str]:
    summary: dict[str, Any] = json.loads(path.read_text())
    return [label for label in summary["classes"] if "+" not in label]


INTENT_DESCRIPTIONS: dict[str, str] = {
    "abbreviation": "the meaning of a code or abbreviation, such as an airline or airport code",
    "aircraft": "the type of airplane used for a flight, not the airline that operates it",
    "airfare": "the price of a flight ticket when the speaker is not asking only for the cheapest fare",
    "airline": "which airline operates a flight, not what an airline code means",
    "airport": "which airport serves a city or flight, not which city an airport is in",
    "capacity": "how many seats or passengers an aircraft holds",
    "cheapest": "the lowest available fare, rather than a general price question",
    "city": "which city an airport serves, not which airport a city uses",
    "day_name": "the day of the week for a date",
    "distance": "how far it is between two places",
    "flight": "finding flights by origin, destination, date, or time constraints",
    "flight_no": "the flight number of a specific flight",
    "flight_time": "departure time, arrival time, or how long a flight takes",
    "ground_fare": "the cost of ground transportation such as a taxi, limousine, or rental car",
    "ground_service": "which ground transportation is available, not what it costs",
    "meal": "what meals are served on a flight",
    "quantity": "how many flights or choices are available, not how many seats a plane has",
    "restriction": "rules or limits on a fare, such as advance purchase or a minimum stay",
}


def build_system_prompt(descriptions: dict[str, str]) -> str:
    return (
        "Classify each airline-travel utterance into exactly one intent:\n"
        + "\n".join(
            f"- {label}: {description}" for label, description in descriptions.items()
        )
        + "\n\nReturn the intent, a confidence from 0 to 1, and 1 to 3 lowercase evidence phrases copied from the utterance."
    )


MANUAL_V1_DESCRIPTIONS: dict[str, str] = {
    **INTENT_DESCRIPTIONS,
    "cheapest": "a request for the cheapest fare in the database when no cities or flights are named",
    "airfare": "the price of a flight ticket, including a request for the cheapest airfare or fare between cities",
}

SYSTEM_PROMPT = build_system_prompt(INTENT_DESCRIPTIONS)
PROMPT_VERSIONS = {
    "baseline": SYSTEM_PROMPT,
    "manual-v1": build_system_prompt(MANUAL_V1_DESCRIPTIONS),
}
PROMPTS_DIR = Path("prompts")
PROMPTS_DIR.mkdir(exist_ok=True)
(PROMPTS_DIR / "baseline.txt").write_text(PROMPT_VERSIONS["baseline"])
(PROMPTS_DIR / "manual-v1.txt").write_text(PROMPT_VERSIONS["manual-v1"])

single_labels = set(load_single_labels(Path("eda_summary.json")))
expected_labels = set(INTENT_DESCRIPTIONS)
missing = sorted(expected_labels - single_labels)
unexpected = sorted(single_labels - expected_labels)
if missing or unexpected:
    raise ValueError(
        f"eda_summary.json disagrees with the expected labels. "
        f"Missing: {missing}. Unexpected: {unexpected}."
    )

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "atis_intent",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": list(INTENT_DESCRIPTIONS),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "multipleOf": 0.01,
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[a-z ]+$"},
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["intent", "confidence", "evidence"],
            "additionalProperties": False,
        },
    },
}


def classify_messages(messages: list[dict[str, str]]) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format=response_format,
    )

    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"The model refused classification: {message.refusal}")
    if message.content is None:
        raise RuntimeError("The model returned no classification")

    payload: dict[str, Any] = json.loads(message.content)
    print(payload)
    intent = payload["intent"]
    if intent not in INTENT_DESCRIPTIONS:
        raise RuntimeError(f"The model returned an unsupported intent: {intent}")
    return intent


def classify_intent(utterance: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    cleaned_utterance = utterance.strip()
    if not cleaned_utterance:
        raise ValueError("utterance must not be empty")
    return classify_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cleaned_utterance},
        ]
    )


DATASET_REVISION = "d44649dde31b4c9d2104aac42a35ed83956e7dec"
BASELINE_SAMPLE_SIZE = 10
BASELINE_RESULTS_PATH = Path("baseline_results.json")


def select_baseline_rows(sample_size: int = BASELINE_SAMPLE_SIZE) -> list[dict[str, str]]:
    dataset = load_dataset("tuetschek/atis", revision=DATASET_REVISION)
    selected: list[dict[str, str]] = []
    for split_name in ("train", "test"):
        for row in dataset[split_name]:
            gold_intent = str(row["intent"]).strip()
            if gold_intent not in INTENT_DESCRIPTIONS:
                continue
            selected.append(
                {
                    "id": f"{split_name}-{row['id']}",
                    "text": str(row["text"]),
                    "gold_intent": gold_intent,
                }
            )
            if len(selected) == sample_size:
                return selected
    raise RuntimeError(f"Found {len(selected)} labeled rows, need {sample_size}")


def evaluate_baseline(
    sample_size: int = BASELINE_SAMPLE_SIZE,
    system_prompt: str = SYSTEM_PROMPT,
    results_path: Path = BASELINE_RESULTS_PATH,
) -> dict[str, Any]:
    rows = select_baseline_rows(sample_size)
    predictions: list[dict[str, Any]] = []
    for row in rows:
        predicted_intent = classify_intent(row["text"], system_prompt)
        predictions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": predicted_intent,
                "correct": bool(exact_match(predicted_intent, row["gold_intent"])),
            }
        )

    correct = sum(row["correct"] for row in predictions)
    accuracy = correct / len(predictions)
    results = {
        "sample_size": len(predictions),
        "correct": correct,
        "accuracy": accuracy,
        "model": MODEL,
        "notes": "No few-shot examples.",
        "predictions": predictions,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def classify_utterances(utterances: list[str]) -> list[str]:
    predictions: list[str] = []
    for utterance in utterances:
        predicted_intent = classify_intent(utterance)
        print(f"intent: {predicted_intent}")
        predictions.append(predicted_intent)
    return predictions


if __name__ == "__main__":
    baseline = evaluate_baseline()
    print(
        f"sample_size={baseline['sample_size']} "
        f"correct={baseline['correct']} accuracy={baseline['accuracy']}"
    )
