import os
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime


API_CONTAINER = os.getenv("API_CONTAINER_NAME", "BrainScanAI-v2-api")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def check_datasets():
    required_paths = [
        Path("/opt/airflow/datasets/classification_clean/Test"),
        Path("/opt/airflow/datasets/detection_yolo/data.yaml"),
        Path("/opt/airflow/datasets/detection_yolo/images/test"),
        Path("/opt/airflow/models/classification/class_names.json"),
        Path("/opt/airflow/models/classification/inceptionv3_best.keras"),
        Path("/opt/airflow/models/classification/inceptionv3_finetuned_best.keras"),
        Path("/opt/airflow/models/detection/best.pt"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required dataset or model paths are missing: " + ", ".join(missing)
        )


def run_in_api_container(module_name):
    command = [
        "docker",
        "exec",
        "-w",
        "/app",
        "-e",
        f"MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}",
        API_CONTAINER,
        "python",
        "-m",
        module_name,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{module_name} failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )


def evaluate_classification():
    run_in_api_container("src.classification.evaluate")


def evaluate_detection():
    run_in_api_container("src.detection.evaluate")


def log_classification_results():
    run_in_api_container("src.classification.log_mlflow")


def log_detection_results():
    run_in_api_container("src.detection.log_mlflow")


def register_classification_model():
    run_in_api_container("src.classification.register_model")


def register_detection_model():
    run_in_api_container("src.detection.register_model")


with DAG(
    dag_id="brainscanai_model_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["brainscanai", "mlflow"],
) as dag:
    check = PythonOperator(
        task_id="check_datasets",
        python_callable=check_datasets,
    )
    classification_evaluation = PythonOperator(
        task_id="evaluate_classification",
        python_callable=evaluate_classification,
    )
    detection_evaluation = PythonOperator(
        task_id="evaluate_detection",
        python_callable=evaluate_detection,
    )
    classification_logging = PythonOperator(
        task_id="log_classification_results",
        python_callable=log_classification_results,
    )
    detection_logging = PythonOperator(
        task_id="log_detection_results",
        python_callable=log_detection_results,
    )
    classification_registration = PythonOperator(
        task_id="register_classification_model",
        python_callable=register_classification_model,
    )
    detection_registration = PythonOperator(
        task_id="register_detection_model",
        python_callable=register_detection_model,
    )

    check >> [classification_evaluation, detection_evaluation]
    classification_evaluation >> classification_logging >> classification_registration
    detection_evaluation >> detection_logging >> detection_registration
