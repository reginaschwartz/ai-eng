import os

import mlflow
from dotenv import load_dotenv


def configure_mlflow() -> None:
    load_dotenv(override=False)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("atis-intent-classification")


configure_mlflow()
print(mlflow.get_tracking_uri())


if __name__ == "__main__":
    configure_mlflow()
    print("MLflow experiment configured")