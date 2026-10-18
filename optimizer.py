import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Literal, TypedDict

import mlflow
from datasets import load_dataset
from dotenv import load_dotenv

from classifier import (
    DATASET_REVISION,
    MODEL,
    SYSTEM_PROMPT,
    classify_intent,
    classify_messages,
)
from metrics import exact_match


SPLIT_SEED = 42
OPTIMIZATION_INTENTS = (
    "flight",
    "airfare",
    "airline",
    "abbreviation",
    "ground_service",
)
CANDIDATE_SEEDS = [7, 21]
SHOT_COUNT = 4
RESULTS_PATH = Path("optimization_results.json")


def select_rows(rows, take: int, skip: int = 0):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["intent"])].append(row)

    selected = []
    rng = Random(SPLIT_SEED)
    for intent in OPTIMIZATION_INTENTS:
        candidates = sorted(groups[intent], key=lambda row: str(row["id"]))
        rng.shuffle(candidates)
        selected.extend(candidates[skip : skip + take])
    return selected


class FewShotExample(TypedDict):
    example_id: str
    text: str
    intent: str


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def build_messages(
    system_prompt: str,
    examples: list[FewShotExample],
    utterance: str,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt}
    ]
    for example in examples:
        messages.extend([
            {"role": "user", "content": example["text"]},
            {
                "role": "assistant",
                "content": json.dumps({"intent": example["intent"]}),
            },
        ])
    messages.append({"role": "user", "content": utterance})
    return messages


def sample_examples(
    demo_pool: list[FewShotExample],
    shot_count: int,
    seed: int,
) -> list[FewShotExample]:
    if not 1 <= shot_count <= len(demo_pool):
        raise ValueError("shot_count is outside the demonstration pool")
    return Random(seed).sample(demo_pool, shot_count)


@dataclass(frozen=True)
class CandidateResult:
    seed: int
    example_ids: tuple[str, ...]
    dev_accuracy: float


ScoreCandidate = Callable[[list[FewShotExample]], float]


def optimize_few_shot(
    demo_pool: list[FewShotExample],
    seeds: list[int],
    shot_count: int,
    score_candidate: ScoreCandidate,
) -> tuple[CandidateResult, list[CandidateResult]]:
    results = []
    for seed in seeds:
        examples = sample_examples(demo_pool, shot_count, seed)
        results.append(CandidateResult(
            seed=seed,
            example_ids=tuple(item["example_id"] for item in examples),
            dev_accuracy=score_candidate(examples),
        ))

    best = max(results, key=lambda result: (result.dev_accuracy, -result.seed))
    return best, results


def load_intent_rows(split_name: str) -> list[dict[str, str]]:
    dataset = load_dataset("tuetschek/atis", revision=DATASET_REVISION)
    rows: list[dict[str, str]] = []
    for row in dataset[split_name]:
        intent = str(row["intent"]).strip()
        if intent not in OPTIMIZATION_INTENTS:
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "intent": intent,
                "text": str(row["text"]),
                "split": split_name,
            }
        )
    return rows


def as_example(row: dict[str, str]) -> FewShotExample:
    return {
        "example_id": f"{row['split']}-{row['id']}",
        "text": row["text"],
        "intent": row["intent"],
    }


def build_splits() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows = load_intent_rows("train")
    test_rows = load_intent_rows("test")
    demonstration = select_rows(train_rows, take=5, skip=0)
    development = select_rows(train_rows, take=2, skip=5)
    test = select_rows(test_rows, take=4, skip=0)
    validate_splits(demonstration, development, test)
    return demonstration, development, test


def validate_splits(
    demonstration: list[dict[str, str]],
    development: list[dict[str, str]],
    test: list[dict[str, str]],
) -> None:
    expected = {
        "demonstration": (demonstration, 5),
        "development": (development, 2),
        "test": (test, 4),
    }
    for name, (rows, per_intent) in expected.items():
        if len(rows) != per_intent * len(OPTIMIZATION_INTENTS):
            raise RuntimeError(f"{name} has {len(rows)} rows")
        for intent in OPTIMIZATION_INTENTS:
            count = sum(row["intent"] == intent for row in rows)
            if count != per_intent:
                raise RuntimeError(f"{name} has {count} {intent} rows")

    demonstration_ids = {as_example(row)["example_id"] for row in demonstration}
    development_ids = {as_example(row)["example_id"] for row in development}
    if demonstration_ids & development_ids:
        raise RuntimeError("demonstration and development rows overlap")

    again = select_rows(load_intent_rows("train"), take=5, skip=0)
    if [row["id"] for row in again] != [row["id"] for row in demonstration]:
        raise RuntimeError("demonstration split is not reproducible")


def accuracy_of(predictions: list[tuple[str, str]]) -> float:
    return sum(exact_match(predicted, gold) for predicted, gold in predictions) / len(
        predictions
    )


def score_zero_shot(rows: list[dict[str, str]]) -> float:
    predictions = [
        (classify_intent(row["text"]), row["intent"])
        for row in rows
    ]
    return accuracy_of(predictions)


def score_examples(examples: list[FewShotExample], rows: list[dict[str, str]]) -> float:
    predictions = []
    for row in rows:
        messages = build_messages(SYSTEM_PROMPT, examples, row["text"])
        predictions.append((classify_messages(messages), row["intent"]))
    return accuracy_of(predictions)


def configure_mlflow() -> None:
    load_dotenv(Path(".env"), override=False)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("atis-intent-classification")


def log_candidate_run(seed: int, example_ids: tuple[str, ...], dev_accuracy: float) -> str:
    with mlflow.start_run(run_name=f"optimize-dev-seed-{seed}") as run:
        mlflow.log_params(
            {
                "model": MODEL,
                "prompt_version": "baseline",
                "dataset": f"tuetschek/atis@{DATASET_REVISION}",
                "sample_id": "optimization-development",
                "split_seed": str(SPLIT_SEED),
                "seed": str(seed),
                "shot_count": str(SHOT_COUNT),
                "example_ids": ",".join(example_ids),
            }
        )
        mlflow.log_metric("dev_accuracy", dev_accuracy)
        return run.info.run_id


def log_test_run(
    seed: int,
    example_ids: tuple[str, ...],
    test_accuracy: float,
    results_path: Path,
) -> str:
    with mlflow.start_run(run_name=f"optimize-test-seed-{seed}") as run:
        mlflow.log_params(
            {
                "model": MODEL,
                "prompt_version": "baseline",
                "dataset": f"tuetschek/atis@{DATASET_REVISION}",
                "sample_id": "optimization-test",
                "split_seed": str(SPLIT_SEED),
                "seed": str(seed),
                "shot_count": str(SHOT_COUNT),
                "example_ids": ",".join(example_ids),
            }
        )
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_artifact(str(results_path), artifact_path="evaluation")
        return run.info.run_id


def run_optimization() -> dict[str, object]:
    configure_mlflow()
    demonstration, development, test = build_splits()
    demo_pool = [as_example(row) for row in demonstration]
    development_baseline = score_zero_shot(development)

    def score_candidate(examples: list[FewShotExample]) -> float:
        return score_examples(examples, development)

    best, results = optimize_few_shot(
        demo_pool,
        CANDIDATE_SEEDS,
        SHOT_COUNT,
        score_candidate,
    )
    for result in results:
        log_candidate_run(result.seed, result.example_ids, result.dev_accuracy)

    winning_examples = sample_examples(demo_pool, SHOT_COUNT, best.seed)
    winning_ids = tuple(example["example_id"] for example in winning_examples)
    if winning_ids != best.example_ids:
        raise RuntimeError("winning examples do not match the selected candidate")
    test_accuracy = score_examples(winning_examples, test)

    payload = {
        "split_seed": SPLIT_SEED,
        "optimization_intents": list(OPTIMIZATION_INTENTS),
        "demonstration_ids": [example["example_id"] for example in demo_pool],
        "development_ids": [as_example(row)["example_id"] for row in development],
        "test_ids": [as_example(row)["example_id"] for row in test],
        "development_baseline": development_baseline,
        "shot_count": SHOT_COUNT,
        "candidates": [
            {
                "seed": result.seed,
                "example_ids": list(result.example_ids),
                "dev_accuracy": result.dev_accuracy,
            }
            for result in results
        ],
        "selected_seed": best.seed,
        "test_accuracy": test_accuracy,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    test_run_id = log_test_run(best.seed, best.example_ids, test_accuracy, RESULTS_PATH)
    print(json.dumps(payload, indent=2))
    print("test_run", test_run_id)
    return payload


if __name__ == "__main__":
    run_optimization()
