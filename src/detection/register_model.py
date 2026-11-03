import csv
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from ultralytics import YOLO


class YOLODetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        predictions = []
        for image_path in model_input["image_path"]:
            detections = []
            for result in self.model.predict(source=image_path, verbose=False):
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detections.append({
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    })
            predictions.append({
                "image_path": str(image_path),
                "detections": detections,
            })
        return pd.DataFrame(predictions)


def register_yolo_model(model_path=None):
    root_dir = Path(__file__).resolve().parents[2]
    model_path = Path(model_path or root_dir / "models/detection/best.pt")
    metrics_path = root_dir / "reports/detection/yolov8_test_metrics.csv"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    with metrics_path.open(newline="") as file:
        metrics = next(csv.DictReader(file))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("BrainScanAI-Detection")

    with mlflow.start_run(run_name="YOLOv8n-model-registration"):
        mlflow.log_params({
            "source_weights": str(model_path),
            "model_type": "YOLOv8",
            "evaluation_split": "Test",
        })
        mlflow.log_metrics({
            "test_precision": float(metrics["Precision"]),
            "test_recall": float(metrics["Recall"]),
            "test_mAP50": float(metrics["mAP50"]),
            "test_mAP50_95": float(metrics["mAP50-95"]),
        })
        mlflow.log_artifact(str(metrics_path))
        model_info = mlflow.pyfunc.log_model(
            name="yolo_detector",
            python_model=YOLODetector(),
            artifacts={"weights": str(model_path)},
            registered_model_name="BrainScanAI-YOLOv8",
            pip_requirements=[
                "mlflow==3.17.2",
                "ultralytics==8.4.183",
                "pandas",
            ],
        )
    return model_info.model_uri


if __name__ == "__main__":
    print(register_yolo_model())
