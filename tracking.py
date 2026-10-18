import os
from typing import TypedDict
from pathlib import Path
import mlflow


class PredictionRow(TypedDict):
    id: str
    text: str
    gold_intent: str
    predicted_intent: str
    correct: bool


def count_error_type(
    analyzed_rows: list[dict[str, str]],
    target_error_type: str,
) -> int:
    return sum(
        row["error_type"] == target_error_type
        for row in analyzed_rows
    )


def track_evaluation(
    predictions: list[PredictionRow],
    accuracy: float,
    model: str,
    prompt_version: str,
    dataset: str,
    sample_id: str,
    results_path: Path,
    hypothesis: str = "",
    analyzed_rows: list[dict[str, str]] | None = None,
    prompt_path: Path | None = None,
    outcome: str = "",
) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("atis-intent-classification")

    correct_count = sum(bool(row["correct"]) for row in predictions)
    error_count = len(predictions) - correct_count
    errors = [row for row in predictions if not row["correct"]]

    with mlflow.start_run(run_name=f"baseline-{prompt_version}") as run:
        params = {
            "model": model,
            "prompt_version": prompt_version,
            "dataset": dataset,
            "sample_id": sample_id,
        }
        if hypothesis:
            params["hypothesis"] = hypothesis
        mlflow.log_params(params)
        metrics = {
            "accuracy": accuracy,
            "sample_size": len(predictions),
            "correct_count": correct_count,
            "error_count": error_count,
        }
        if analyzed_rows is not None:
            metrics["label_boundary_error_count"] = count_error_type(
                analyzed_rows, "label_boundary"
            )
        mlflow.log_metrics(metrics)
        mlflow.log_table(
            data={
                "example_id": [row["id"] for row in predictions],
                "utterance": [row["text"] for row in predictions],
                "expected_intent": [row["gold_intent"] for row in predictions],
                "predicted_intent": [row["predicted_intent"] for row in predictions],
                "correct": [row["correct"] for row in predictions],
            },
            artifact_file="evaluation/predictions.json",
        )
        error_types = {
            row["id"]: row["error_type"] for row in (analyzed_rows or [])
        }
        mlflow.log_table(
            data={
                "example_id": [row["id"] for row in errors],
                "utterance": [row["text"] for row in errors],
                "expected_intent": [row["gold_intent"] for row in errors],
                "predicted_intent": [row["predicted_intent"] for row in errors],
                "error_type": [error_types.get(str(row["id"]), "") for row in errors],
            },
            artifact_file="evaluation/errors.json",
        )
        if prompt_path is not None:
            mlflow.log_artifact(str(prompt_path), artifact_path="prompt")
        if outcome:
            mlflow.set_tag("outcome", outcome)
        mlflow.log_artifact(str(results_path), artifact_path="evaluation")
        return run.info.run_id
def attach_dataset_ids(
    predictions: list[dict[str, object]],
) -> list[dict[str, object]]:
    from datasets import load_dataset

    from classifier import DATASET_REVISION, INTENT_DESCRIPTIONS

    dataset = load_dataset("tuetschek/atis", revision=DATASET_REVISION)
    ids_by_text: dict[str, object] = {}
    for split_name in ("train", "test"):
        for row in dataset[split_name]:
            if str(row["intent"]).strip() not in INTENT_DESCRIPTIONS:
                continue
            ids_by_text.setdefault(str(row["text"]), row["id"])

    identified: list[dict[str, object]] = []
    for prediction in predictions:
        row = dict(prediction)
        row["id"] = ids_by_text[str(prediction["text"])]
        identified.append(row)
    return identified
def record_baseline_evaluation() -> str:
    import importlib.util

    from classifier import BASELINE_RESULTS_PATH, DATASET_REVISION, evaluate_baseline

    module_path = Path(__file__).with_name("ml-flow-test.py")
    spec = importlib.util.spec_from_file_location("ml_flow_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    ml_flow_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_flow_test)
    ml_flow_test.configure_mlflow()

    results = evaluate_baseline()
    predictions: list[PredictionRow] = [
        {
            "id": int(row["id"]),
            "text": str(row["text"]),
            "gold_intent": str(row["gold_intent"]),
            "predicted_intent": str(row["predicted_intent"]),
            "correct": bool(row["correct"]),
        }
        for row in attach_dataset_ids(results["predictions"])
    ]
    run_id = track_evaluation(
        predictions=predictions,
        accuracy=float(results["accuracy"]),
        model=str(results["model"]),
        prompt_version="no-few-shot",
        dataset=f"tuetschek/atis@{DATASET_REVISION}",
        sample_id="first-10-single-label-rows",
        results_path=BASELINE_RESULTS_PATH,
    )
    print(run_id)
    return run_id


if __name__ == "__main__":
    record_baseline_evaluation()