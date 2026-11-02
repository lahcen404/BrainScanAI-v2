import pandas as pd

from src.tracking.mlflow_tracker import log_experiment


def log_classification_results():
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[2]
    model_dir = root_dir / "models" / "classification"
    report_dir = root_dir / "reports" / "classification"
    metrics_path = report_dir / "classification_model_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    metrics_df = pd.read_csv(metrics_path)
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

    for experiment in experiments:
        row = metrics_df.loc[
            metrics_df["Model"] == experiment["model_name"]
        ].iloc[0]
        artifacts = [
            model_dir / experiment["model_file"],
            report_dir / experiment["report_file"],
            report_dir / experiment["confusion_file"],
            model_dir / "class_names.json",
            metrics_path,
        ]
        for artifact in artifacts:
            if not artifact.is_file():
                raise FileNotFoundError(artifact)

        log_experiment(
            experiment_name="BrainScanAI-Classification",
            run_name=experiment["run_name"],
            params={
                "architecture": "InceptionV3",
                "model_phase": experiment["model_name"],
                "image_size": "299x299",
                "batch_size": 32,
                "test_images": int(row["Images"]),
                "number_of_classes": 4,
                "evaluation_split": "Test",
            },
            metrics={
                "test_loss": float(row["Loss"]),
                "test_accuracy": float(row["Accuracy"]),
                "test_precision_macro": float(row["Precision (macro)"]),
                "test_recall_macro": float(row["Recall (macro)"]),
                "test_f1_macro": float(row["F1-score (macro)"]),
            },
            artifacts=artifacts,
        )


if __name__ == "__main__":
    log_classification_results()
