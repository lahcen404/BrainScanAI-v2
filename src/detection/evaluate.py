
import csv
from pathlib import Path

from ultralytics import YOLO

def evaluate_yolo(
    model_path=None,
    dataset_yaml=None,
):
    root_dir = Path(__file__).resolve().parents[2]
    model_path = Path(model_path or root_dir / "models/detection/best.pt")
    dataset_yaml = Path(
        dataset_yaml or root_dir / "datasets/detection_yolo/data.yaml"
    )
    # 1. Validate paths
    model_path = Path(model_path)
    dataset_yaml = Path(dataset_yaml)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    # 2. Load the trained YOLO model
    model = YOLO(str(model_path))

    # 3. Evaluate using the test dataset
    results = model.val(
        data=str(dataset_yaml),
        split="test",
    )

    # 4. Extract test metrics
    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    # 5. Save test metrics to CSV
    results_dir = root_dir / "reports" / "detection"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "yolov8_test_metrics.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Model", "Dataset", "Precision", "Recall",
                        "mAP50", "mAP50-95"],
        )

        writer.writeheader()
        writer.writerow({
            "Model": model_path.name,
            "Dataset": "Test",
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "mAP50": metrics["mAP50"],
            "mAP50-95": metrics["mAP50_95"],
        })

    print("YOLOv8 test evaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Test metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    evaluate_yolo()
