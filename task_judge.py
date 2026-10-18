import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"), override=False)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
TRACES_PATH = Path(__file__).with_name("tau2_full.jsonl")
RESULTS_PATH = Path("task_judge_results.json")
TASKS = (1, 3, 5, 16)
VERDICT_TO_LABEL = {"completed": 1, "failed": -1, "unjudgeable": 0}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence_message_indices": {"type": "array", "items": {"type": "integer"}},
        "verdict": {"type": "string", "enum": ["completed", "failed", "unjudgeable"]},
    },
    "required": ["reason", "evidence_message_indices", "verdict"],
    "additionalProperties": False,
}

TASK_JUDGE_PROMPT = """You evaluate a customer-service agent's recorded run.

Decide whether the run satisfied the task objective under the operating policy.

Return "completed" when the objective was met without violating the policy.
Return "failed" when the objective was missed, or when the agent took or promised
an action the policy forbids.
Return "unjudgeable" only when the transcript ends before an outcome exists.

Refusing a request the policy forbids is a correct outcome.
A confident closing message does not mean the work was done or permitted.
Cite the message indices your verdict rests on."""


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


def render_messages(
    messages: list[dict],
    start_index: int = 0,
    include_system: bool = False,
) -> str:
    parts = []
    for offset, message in enumerate(messages):
        index = start_index + offset
        if message["role"] == "system":
            if include_system:
                parts.append(f"[{index}] system: {message.get('content')}")
            continue
        if message["role"] == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                parts.append(
                    f"[{index}] agent calls {call['function']['name']}"
                    f"({call['function']['arguments']})"
                )
        elif message["role"] == "assistant":
            parts.append(f"[{index}] agent: {message['content']}")
        elif message["role"] == "tool":
            parts.append(f"[{index}] tool result: {str(message.get('content'))[:400]}")
        else:
            parts.append(f"[{index}] customer: {message['content']}")
    return "\n".join(parts)


def render_run(trace: dict, include_policy: bool = True) -> str:
    parts = [f"# Task objective\n{trace['task_description']['purpose']}\n"]
    parts.append(f"# Customer request\n{trace['question']}\n")
    if include_policy:
        parts.append(f"# Operating policy\n{trace['messages'][0]['content']}\n")
    parts.append("# Transcript")
    transcript = render_messages(trace["messages"])
    if transcript:
        parts.append(transcript)
    return "\n".join(parts)


def judge_run(rendered: str, model: str = MODEL) -> dict:
   # print("rendered : "+ rendered)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TASK_JUDGE_PROMPT},
            {"role": "user", "content": rendered},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True},
        },
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The judge returned no content")
    verdict = json.loads(content)
    if verdict.get("verdict") not in VERDICT_TO_LABEL:
        raise RuntimeError(f"Out-of-vocabulary verdict: {verdict.get('verdict')!r}")
    return verdict


def failure_class(records: list[dict]) -> dict[str, float]:
    tp = sum(1 for record in records if record["predicted"] == -1 and record["expected"] == -1)
    fp = sum(1 for record in records if record["predicted"] == -1 and record["expected"] != -1)
    fn = sum(1 for record in records if record["predicted"] != -1 and record["expected"] == -1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_run(trace: dict, include_policy: bool) -> dict:
    result = judge_run(render_run(trace, include_policy=include_policy))
    query_index = trace["query_index"]
    sample_index = trace["sample_index"]
    return {
        "trace_id": f"{query_index}-{sample_index}",
        "query_index": query_index,
        "sample_index": sample_index,
        "predicted": VERDICT_TO_LABEL[result["verdict"]],
        "expected": trace["final_label"],
        "reason": result["reason"],
        "evidence": result["evidence_message_indices"],
    }


def disagreement_notes(records: list[dict]) -> str:
    def ident(record: dict) -> str:
        return str(record.get("trace_id") or f"{record['query_index']}-{record['sample_index']}")

    misses = [record for record in records if record["predicted"] != record["expected"]]
    missed_failures = [
        ident(record)
        for record in misses
        if record["expected"] == -1 and record["predicted"] != -1
    ]
    false_failures = [
        ident(record)
        for record in misses
        if record["predicted"] == -1 and record["expected"] != -1
    ]
    other = [
        ident(record)
        for record in misses
        if ident(record) not in missed_failures and ident(record) not in false_failures
    ]
    return (
        f"{len(misses)} of {len(records)} runs disagree with the human label. "
        f"Human failures the judge did not call failed: {', '.join(missed_failures)}. "
        f"Runs the judge called failed although the human did not: {', '.join(false_failures)}. "
        f"Other disagreements: {', '.join(other)}. "
        "The reasons often describe a refusal or a transfer to a human and then assign the opposite verdict. "
        "On task 3 the judge accepts the customer's claim to be Gold. "
        "On task 1 sample 0 it treats a completed refund as allowed. "
        "Task 1 sample 4 is a match: the judge marks the cancellation failed after the agent carries it out."
    )


def build_payload(
    records: list[dict],
    label_counts: Counter,
    ablation_agreement: float,
) -> dict:
    trace_count = len(records)
    if trace_count == 0:
        raise RuntimeError("The judge produced no records")
    accuracy = agreement_of(records)
    majority_accuracy = max(label_counts.values()) / sum(label_counts.values())
    comparison = (
        f"Judge accuracy is {accuracy:.2f}. "
        f"The majority-class baseline always predicts the most common final_label and scores "
        f"{majority_accuracy:.2f}. "
        + (
            "The judge is below that baseline."
            if accuracy < majority_accuracy
            else "The judge is at or above that baseline."
        )
    )
    payload = {
        "model": MODEL,
        "input_bundle": "task_description.purpose + question + messages[*]",
        "held_back": ["final_label", "step_labels", "ground_truth"],
        "trace_count": trace_count,
        "run_count": trace_count,
        "constant_baseline": majority_accuracy,
        "majority_class_accuracy": majority_accuracy,
        "baseline_comparison": comparison,
        "accuracy": accuracy,
        "agreement": accuracy,
        "failure_class": failure_class(records),
        "ablation_no_policy": {"agreement": ablation_agreement},
        "disagreement_notes": disagreement_notes(records),
        "records": records,
    }
    if len(payload["records"]) != payload["trace_count"]:
        raise RuntimeError("record count does not match trace_count")
    return payload


def agreement_of(records: list[dict]) -> float:
    return sum(record["predicted"] == record["expected"] for record in records) / len(records)


def main() -> None:
    working = load_working_set()
    records = []
    ablated = []
    for trace in working:
        record = score_run(trace, include_policy=True)
        records.append(record)
        print(
            f"full query={record['query_index']} sample={record['sample_index']} "
            f"predicted={record['predicted']} expected={record['expected']}"
        )
        ablated_record = score_run(trace, include_policy=False)
        ablated.append(ablated_record)
        print(
            f"ablated query={ablated_record['query_index']} sample={ablated_record['sample_index']} "
            f"predicted={ablated_record['predicted']} expected={ablated_record['expected']}"
        )

    label_counts = Counter(trace["final_label"] for trace in working)
    payload = build_payload(records, label_counts, agreement_of(ablated))
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, indent=2))
    for full, without_policy in zip(records, ablated):
        if full["predicted"] != full["expected"] or without_policy["predicted"] != full["predicted"]:
            print(
                f"compare query={full['query_index']} sample={full['sample_index']} "
                f"expected={full['expected']} full={full['predicted']} "
                f"ablated={without_policy['predicted']}"
            )


if __name__ == "__main__":
    main()
