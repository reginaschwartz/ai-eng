import json
from collections import Counter
from pathlib import Path


TRACES_PATH = Path(__file__).with_name("tau2_full.jsonl")
TASKS = (1, 3, 5, 16)
PROFILE_PATH = Path("agent_trace_profile.json")


def load_traces() -> list[dict]:
    """Read the local tau2 trace file."""
    return [json.loads(line) for line in TRACES_PATH.open(encoding="utf-8")]


def outcomes_for(traces: list[dict], query_index: int) -> list[int]:
    attempts = sorted(
        (trace for trace in traces if trace["query_index"] == query_index),
        key=lambda trace: trace["sample_index"],
    )
    return [int(trace["final_label"]) for trace in attempts]


def pass_at_k(outcomes: list[int]) -> int:
    return int(any(label == 1 for label in outcomes))


def pass_pow_k(outcomes: list[int]) -> int:
    return int(all(label == 1 for label in outcomes))


def build_profile(traces: list[dict]) -> dict:
    working = [trace for trace in traces if trace["query_index"] in TASKS]
    tasks = [
        {"query_index": query_index, "outcomes": outcomes_for(working, query_index)}
        for query_index in TASKS
    ]
    counts = Counter(int(trace["final_label"]) for trace in working)
    task_1 = tasks[0]["outcomes"]
    run_count = len(working)
    majority = max(counts.values())
    return {
        "tasks": tasks,
        "run_count": run_count,
        "outcome_counts": {str(label): counts[label] for label in (1, -1, 0)},
        "constant_baseline_accuracy": majority / run_count,
        "task_1_pass_rate": sum(label == 1 for label in task_1) / len(task_1),
        "task_1_pass_at_5": pass_at_k(task_1),
        "task_1_pass_pow_5": pass_pow_k(task_1),
    }


def main() -> None:
    profile = build_profile(load_traces())
    PROFILE_PATH.write_text(json.dumps(profile, indent=2) + "\n")
    print(PROFILE_PATH.read_text())


if __name__ == "__main__":
    main()
