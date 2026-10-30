
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_DIR / "src"))

from tracking.mlflow_tracker import log_experiment


MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"

EXPERIMENT_NAME = "BrainScanAI-Classification"


# --------------------------------------------------
# 2. Load evaluation results
# --------------------------------------------------

metrics_path = REPORT_DIR / "classification_model_metrics.csv"

if not metrics_path.exists():
    raise FileNotFoundError(metrics_path)

metrics_df = pd.read_csv(metrics_path)

print("Loaded evaluation results:")
print(metrics_df.to_string(index=False))


# --------------------------------------------------
# 3. Define experiments
# --------------------------------------------------

experiments = [
    {
        "model_name": "Feature Extraction",
        "run_name": "InceptionV3-Feature-Extraction",
        "model_file": "inceptionv3_best.keras",
        "report_file": "feature_extraction_report.csv",
        "confusion_file": "feature_extraction_confusion_matrix.csv",
    },
    {
        "model_name": "Fine-Tuning",
        "run_name": "InceptionV3-Fine-Tuning",
        "model_file": "inceptionv3_finetuned_best.keras",
        "report_file": "fine-tuning_report.csv",
        "confusion_file": "fine-tuning_confusion_matrix.csv",
    },
]


# --------------------------------------------------
# 4. Log each experiment
# --------------------------------------------------

for experiment in experiments:

    model_name = experiment["model_name"]

    # Find the corresponding row in the evaluation CSV
    row = metrics_df.loc[
        metrics_df["Model"] == model_name
    ].iloc[0]

    model_path = MODEL_DIR / experiment["model_file"]
    report_path = REPORT_DIR / experiment["report_file"]
    confusion_path = REPORT_DIR / experiment["confusion_file"]
    class_names_path = MODEL_DIR / "class_names.json"

    artifacts = [
        model_path,
        report_path,
        confusion_path,
        class_names_path,
        metrics_path,
    ]

    # Verify all artifacts exist before logging
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Artifact not found: {artifact}"
            )

    # Experiment parameters
    params = {
        "architecture": "InceptionV3",
        "model_phase": model_name,
        "image_size": "299x299",
        "batch_size": 32,
        "test_images": int(row["Images"]),
        "number_of_classes": 4,
        "loss_function": "sparse_categorical_crossentropy",
        "model_format": "Keras",
        "evaluation_split": "Test",
    }

    # Verified test metrics
    metrics = {
        "test_loss": float(row["Loss"]),
        "test_accuracy": float(row["Accuracy"]),
        "test_precision_macro": float(row["Precision (macro)"]),
        "test_recall_macro": float(row["Recall (macro)"]),
        "test_f1_macro": float(row["F1-score (macro)"]),
    }

    print("\n" + "=" * 55)
    print(f"Logging: {experiment['run_name']}")
    print("=" * 55)

    log_experiment(
        experiment_name=EXPERIMENT_NAME,
        run_name=experiment["run_name"],
        params=params,
        metrics=metrics,
        artifacts=artifacts,
    )

    print(f"Successfully logged: {experiment['run_name']}")


print("\nBoth classification experiments have been logged!")
print(f"Experiment: {EXPERIMENT_NAME}")
print("Open MLflow UI: http://localhost:5000")
