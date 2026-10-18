import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


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


SYSTEM_PROMPT = (
    "Classify each airline-travel utterance into exactly one intent:\n"
    + "\n".join(
        f"- {label}: {description}" for label, description in INTENT_DESCRIPTIONS.items()
    )
    + "\n\nReturn the intent, a confidence from 0 to 1, and 1 to 3 lowercase evidence phrases copied from the utterance."
)

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


def classify_intent(utterance: str) -> str:
    cleaned_utterance = utterance.strip()
    if not cleaned_utterance:
        raise ValueError("utterance must not be empty")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned_utterance},
        ],
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


def classify_utterances(utterances: list[str]) -> list[str]:
    predictions: list[str] = []
    for utterance in utterances:
        predicted_intent = classify_intent(utterance)
        print(f"intent: {predicted_intent}")
        predictions.append(predicted_intent)
    return predictions


if __name__ == "__main__":
    classify_utterances(
        [
            "i want to fly from boston to denver",
            "how much does a flight from dallas to baltimore cost",
            "what is the cheapest fare from boston to dallas",
            "which airline flies between denver and philadelphia",
             "are meals served on the flight"
        ]
    )
