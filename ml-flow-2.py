from pathlib import Path

import mlflow


def log_evaluation_run(
    accuracy: float,
    predictions: list[dict[str, object]],
    model: str,
    prompt_version: str,
    dataset: str,
    sample_id: str,
    max_workers: int,
    results_path: Path,
) -> str:
    correct_count = sum(bool(row["correct"]) for row in predictions)
    error_count = len(predictions) - correct_count

    with mlflow.start_run(run_name=f"baseline-{prompt_version}") as run:
        mlflow.log_params(
            {
                "model": model,
                "prompt_version": prompt_version,
                "dataset": dataset,
                "sample_id": sample_id,
                "max_workers": max_workers,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "sample_size": len(predictions),
                "correct_count": correct_count,
                "error_count": error_count,
            }
        )
        mlflow.log_artifact(str(results_path), artifact_path="evaluation")
        log_error_table(select_errors(predictions))
        return run.info.run_id


def select_errors(
    predictions: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [row for row in predictions if not bool(row["correct"])]


def log_error_table(errors: list[dict[str, object]]) -> None:
    table: dict[str, list[object]] = {
        "example_id": [row["id"] for row in errors],
        "utterance": [row["text"] for row in errors],
        "expected_intent": [row["gold_intent"] for row in errors],
        "predicted_intent": [row["predicted_intent"] for row in errors],
    }
    mlflow.log_table(data=table, artifact_file="evaluation/errors.json")


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
    predictions = attach_dataset_ids(results["predictions"])
    run_id = log_evaluation_run(
        accuracy=float(results["accuracy"]),
        predictions=predictions,
        model=str(results["model"]),
        prompt_version="no-few-shot",
        dataset=f"tuetschek/atis@{DATASET_REVISION}",
        sample_id="first-10-single-label-rows",
        max_workers=1,
        results_path=BASELINE_RESULTS_PATH,
    )
    print(run_id)
    return run_id


if __name__ == "__main__":
    record_baseline_evaluation()