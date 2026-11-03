import csv
from pathlib import Path

from src.tracking.mlflow_tracker import log_experiment


def log_detection_results():
    root_dir = Path(__file__).resolve().parents[2]
    metrics_path = root_dir / "reports" / "detection" / "yolov8_test_metrics.csv"
    model_path = root_dir / "models" / "detection" / "best.pt"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    with metrics_path.open(newline="") as file:
        metrics = next(csv.DictReader(file))

    log_experiment(
        experiment_name="BrainScanAI-Detection",
        run_name="YOLOv8n-test-evaluation",
        params={
            "model_path": str(model_path),
            "dataset_split": "test",
        },
        metrics={
            "precision": float(metrics["Precision"]),
            "recall": float(metrics["Recall"]),
            "mAP50": float(metrics["mAP50"]),
            "mAP50_95": float(metrics["mAP50-95"]),
        },
        artifacts=[model_path, metrics_path],
    )


if __name__ == "__main__":
    log_detection_results()
