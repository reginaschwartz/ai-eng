import json
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Callable, Literal, TypedDict


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
