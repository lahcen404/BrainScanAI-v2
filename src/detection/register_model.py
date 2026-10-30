
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from ultralytics import YOLO


# 1. Wrap the Ultralytics model so MLflow can load and use it.
class YOLODetector(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        weights_path = context.artifacts["weights"]
        self.model = YOLO(weights_path)

    def predict(self, context, model_input, params=None):
        predictions = []

        for image_path in model_input["image_path"]:
            results = self.model.predict(
                source=image_path,
                verbose=False,
            )

            for result in results:
                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0])

                    detections.append({
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [
                            float(value)
                            for value in box.xyxy[0].tolist()
                        ],
                    })

                predictions.append({
                    "image_path": str(image_path),
                    "detections": detections,
                })

        return pd.DataFrame(predictions)


# 2. Package and register the existing YOLO weights.
def register_yolo_model(
    model_path="models/detection/best.pt",
):
    model_path = Path(model_path).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {model_path}"
        )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("BrainScanAI-Detection")

    registered_model_name = "BrainScanAI-YOLOv8"

    with mlflow.start_run(
        run_name="YOLOv8n-model-registration"
    ) as run:

        # Package the YOLO weights and prediction wrapper.
        model_info = mlflow.pyfunc.log_model(
            name="yolo_detector",
            python_model=YOLODetector(),
            artifacts={
                "weights": str(model_path),
            },
            pip_requirements=[
                "mlflow==3.17.2",
                "ultralytics==8.4.183",
                "pandas",
            ],
        )

        # Register the packaged model.
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_model_name,
        )

        # Preserve the source evaluation and its existing metrics.
        mlflow.log_params({
            "source_weights": str(model_path),
            "model_type": "YOLOv8",
            "source_evaluation_run_id":
                "4c51263bc6904fce861ec1d2564d41a0",
        })

        mlflow.log_metrics({
            "test_precision": 0.9259724547737114,
            "test_recall": 0.9269046971709985,
            "test_mAP50": 0.9547261665130946,
            "test_mAP50_95": 0.7186111870686309,
        })

        print("\nModel registration completed!")
        print(f"Run ID: {run.info.run_id}")
        print(f"Registered name: {registered_model.name}")
        print(f"Version: {registered_model.version}")
        print(
            f"Model URI: models:/{registered_model.name}/"
            f"{registered_model.version}"
        )

    return registered_model


if __name__ == "__main__":
    register_yolo_model()
