import os
import mlflow
from pathlib import Path

def log_experiment(experiment_name, run_name, params, metrics, artifacts=None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI",
        "http://localhost:5000",)

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifacts:
            for artifact in artifacts:
                artifact_path = Path(artifact)

                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Artifact not found: {artifact_path}"
                    )

                mlflow.log_artifact(str(artifact_path))