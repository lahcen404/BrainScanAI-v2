from pathlib import Path

from ultralytics import YOLO

from src.tracking.mlflow_tracker import log_experiment


def train_yolo(
    model_name="yolov8n.pt",
    epochs=30,
    image_size=640,
    batch_size=16,
    patience=8,
    experiment_run_name="YOLOv8n-training",
    training_name="brainscan_yolov8n",
):
    # 1. Check dataset configuration
    dataset_yaml = Path("datasets/detection_yolo/data_local.yaml")

    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}"
        )

    print(f"Using dataset: {dataset_yaml}")

    # 2. Load pretrained YOLO model
    model = YOLO(model_name)

    # 3. Train the model
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=patience,
        name=training_name,
    )

    # 4. Retrieve best checkpoint
    best_model_path = Path(model.trainer.best)

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {best_model_path}"
        )

    # 5. Retrieve training metrics
    metrics = model.trainer.metrics

    # 6. Prepare hyperparameters
    params = {
        "model": model_name,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "patience": patience,
        "dataset": str(dataset_yaml),
    }

    # 7. Prepare metrics for MLflow
    mlflow_metrics = {
        "precision": float(metrics["metrics/precision(B)"]),
        "recall": float(metrics["metrics/recall(B)"]),
        "mAP50": float(metrics["metrics/mAP50(B)"]),
        "mAP50_95": float(metrics["metrics/mAP50-95(B)"]),
    }

    # 8. Log experiment in MLflow
    log_experiment(
        experiment_name="BrainScanAI-Detection",
        run_name=experiment_run_name,
        params=params,
        metrics=mlflow_metrics,
        artifacts=[str(best_model_path)],
    )

    print("\nYOLOv8 training completed!")
    print(f"Best model: {best_model_path}")
    print(f"Metrics: {mlflow_metrics}")

    return best_model_path, mlflow_metrics


if __name__ == "__main__":
    train_yolo()