import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from task_judge import MODEL, render_messages


load_dotenv(Path(__file__).with_name(".env"), override=False)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
TRACES_PATH = Path(__file__).with_name("tau2_full.jsonl")
TASK_RESULTS_PATH = Path("task_judge_results.json")
RESULTS_PATH = Path("step_judge_results.json")
TASKS = (1, 3, 5, 16)
WINDOW = 8
PRIMARY_STRATEGY = "window-8"
NEUTRAL_HANDLING = "kept"
VERDICT_TO_LABEL = {"good": 1, "bad": -1, "neutral": 0}
CLASS_NAMES = ((1, "good"), (-1, "bad"), (0, "neutral"))

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["good", "bad", "neutral"]},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}

STEP_JUDGE_PROMPT = """You judge one decision in a customer-service agent's recorded run.

You see the customer request, only the messages that preceded this decision, and the decision itself. Later messages are not available and must not be assumed.

Return "good" when this decision was a reasonable move given the request and any policy visible in the prefix.
Return "bad" when this decision violates the policy, calls the wrong tool, passes a wrong argument, or promises an action the policy forbids.
Return "neutral" when the decision neither helps nor harms.

Judge this step alone. A confident message does not make the decision good."""


def load_working_set() -> list[dict]:
    traces = [json.loads(line) for line in TRACES_PATH.open(encoding="utf-8")]
    working = []
    for query_index in TASKS:
        attempts = sorted(
            (trace for trace in traces if trace["query_index"] == query_index),
            key=lambda trace: trace["sample_index"],
        )
        working.extend(attempts)
    return working


def step_contexts(trace: dict, strategy: str) -> list[tuple[int, str, str, int]]:
    contexts = []
    for key in sorted(trace["step_labels"], key=int):
        index = int(key)
        prior = trace["messages"][:index]
        if strategy == "window-8":
            window = prior[-WINDOW:]
            prefix = render_messages(
                window,
                start_index=index - len(window),
                include_system=True,
            )
        elif strategy == "full":
            prefix = render_messages(prior, include_system=True)
        else:
            raise ValueError(f"Unknown prefix strategy: {strategy}")
        step = render_messages(
            trace["messages"][index : index + 1],
            start_index=index,
            include_system=True,
        )
        contexts.append((index, prefix, step, int(trace["step_labels"][key])))
    return contexts


def rendered_step(trace: dict, prefix: str, step: str) -> str:
    return "\n".join(
        [
            f"# Customer request\n{trace['question']}\n",
            f"# Preceding messages\n{prefix}\n",
            f"# Decision\n{step}",
        ]
    )


def assert_no_future_messages(prefix: str, step_index: int) -> None:
    indices = [int(value) for value in re.findall(r"\[(\d+)\]", prefix)]
    if any(index >= step_index for index in indices):
        raise RuntimeError(f"Prefix for step {step_index} includes a later message")


def judge_step(rendered: str) -> dict:
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": STEP_JUDGE_PROMPT},
                    {"role": "user", "content": rendered},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "step_verdict",
                        "schema": STEP_SCHEMA,
                        "strict": True,
                    },
                },
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("The step judge returned no content")
            verdict = json.loads(content)
            if verdict.get("verdict") not in VERDICT_TO_LABEL:
                raise RuntimeError(f"Out-of-vocabulary verdict: {verdict.get('verdict')!r}")
            return verdict
        except Exception as error:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"retry after {error}")
    raise RuntimeError("The step judge did not return a verdict")


def judge_strategy(working: list[dict], strategy: str) -> list[dict]:
    records = []
    for trace in working:
        run_id = [trace["query_index"], trace["sample_index"]]
        for index, prefix, step, expected in step_contexts(trace, strategy):
            assert_no_future_messages(prefix, index)
            verdict = judge_step(rendered_step(trace, prefix, step))
            record = {
                "run_id": run_id,
                "step_index": index,
                "predicted": VERDICT_TO_LABEL[verdict["verdict"]],
                "expected": expected,
                "reason": verdict["reason"],
            }
            records.append(record)
            print(
                f"{strategy} run={run_id} step={index} "
                f"predicted={record['predicted']} expected={expected}",
                flush=True,
            )
    return records


def per_class(records: list[dict]) -> dict[str, dict[str, float]]:
    report = {}
    for label, name in CLASS_NAMES:
        true_positive = sum(
            1 for record in records if record["predicted"] == label and record["expected"] == label
        )
        predicted = sum(1 for record in records if record["predicted"] == label)
        actual = sum(1 for record in records if record["expected"] == label)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {
            "support": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def predicted_has_bad_step(run_id: list[int], step_records: list[dict]) -> bool:
    return any(
        record["predicted"] == -1
        for record in step_records
        if record["run_id"] == run_id
    )


def judge_comparison(step_records: list[dict]) -> str:
    task_results = json.loads(TASK_RESULTS_PATH.read_text())
    step_only = []
    task_only = []
    for record in task_results["records"]:
        run_id = [record["query_index"], record["sample_index"]]
        flagged = predicted_has_bad_step(run_id, step_records)
        task_failed = record["predicted"] == -1
        if flagged and not task_failed:
            step_only.append(f"{run_id[0]}-{run_id[1]}")
        elif task_failed and not flagged:
            task_only.append(f"{run_id[0]}-{run_id[1]}")
    both_flagged = []
    for record in task_results["records"]:
        run_id = [record["query_index"], record["sample_index"]]
        if predicted_has_bad_step(run_id, step_records) and record["predicted"] == -1 and record["expected"] == 1:
            both_flagged.append(f"{run_id[0]}-{run_id[1]}")
    return (
        "A trace is broken for the step judge when any judged step is bad, and broken for the "
        "task judge when its run verdict is failed. "
        f"The step judge flags bad steps while the task judge does not on: {', '.join(step_only) or 'none'}. "
        "A bad step can still leave the run successful, so those disagreements can be recoveries. "
        f"Both judges flag these human passes as broken: {', '.join(both_flagged) or 'none'}. "
        "On 3-0 and 5-4 the human label says the run recovered after bad steps, so the step judge is right about the steps and the task judge is wrong about the outcome. "
        f"The task judge calls a failure while the step judge finds no bad step on: {', '.join(task_only) or 'none'}. "
        "Those point to a failure that is not localized to one decision, or to a window that dropped the policy in the opening system message."
    )


def strategy_note(primary: list[dict], other: list[dict]) -> str:
    primary_report = per_class(primary)
    other_report = per_class(other)
    primary_accuracy = sum(record["predicted"] == record["expected"] for record in primary) / len(primary)
    other_accuracy = sum(record["predicted"] == record["expected"] for record in other) / len(other)
    better = "window-8" if primary_report["bad"]["recall"] >= other_report["bad"]["recall"] else "full"
    return (
        f"window-8 accuracy {primary_accuracy:.2f}, bad recall {primary_report['bad']['recall']:.2f}. "
        f"full prefix accuracy {other_accuracy:.2f}, bad recall {other_report['bad']['recall']:.2f}. "
        f"On bad-step recall, {better} scored better."
    )


def build_payload(records: list[dict], comparison: str, strategies: str) -> dict:
    labels = Counter(record["expected"] for record in records)
    accuracy = sum(record["predicted"] == record["expected"] for record in records) / len(records)
    payload = {
        "model": MODEL,
        "split": "working",
        "prefix_strategy": PRIMARY_STRATEGY,
        "neutral_handling": NEUTRAL_HANDLING,
        "step_count": len(records),
        "majority_class_accuracy": max(labels.values()) / len(records),
        "accuracy": accuracy,
        "per_class": per_class(records),
        "strategy_comparison": strategies,
        "records": records,
        "judge_comparison": comparison,
    }
    if payload["step_count"] != len(payload["records"]):
        raise RuntimeError("step_count does not match the record count")
    return payload


def main() -> None:
    working = load_working_set()
    labeled = sum(len(trace["step_labels"]) for trace in working)
    print(f"labeled steps {labeled}", flush=True)
    window_records = judge_strategy(working, PRIMARY_STRATEGY)
    full_records = judge_strategy(working, "full")
    payload = build_payload(
        window_records,
        judge_comparison(window_records),
        strategy_note(window_records, full_records),
    )
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
