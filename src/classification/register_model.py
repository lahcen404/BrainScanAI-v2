import json
import os
from pathlib import Path

import mlflow
import mlflow.keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"


def random_blur(images):
    apply_blur = tf.random.uniform(()) < 0.3

    def blur():
        channels = tf.shape(images)[-1]
        kernel = tf.ones((3, 3, channels, 1), dtype=images.dtype) / 9.0
        return tf.nn.depthwise_conv2d(
            images, kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

    return tf.cond(apply_blur, blur, lambda: images)


def random_noise(images):
    noise = tf.random.normal(
        shape=tf.shape(images), mean=0.0, stddev=5.0, dtype=images.dtype
    )
    return tf.clip_by_value(images + noise, 0.0, 255.0)


def register_classification_model():
    metrics_path = REPORT_DIR / "classification_model_metrics.csv"
    class_names_path = MODEL_DIR / "class_names.json"
    model_path = MODEL_DIR / "inceptionv3_finetuned_best.keras"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not class_names_path.is_file():
        raise FileNotFoundError(class_names_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    with class_names_path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    metrics_df = pd.read_csv(metrics_path)
    row = metrics_df.loc[metrics_df["Model"] == "Fine-Tuning"].iloc[0]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("BrainScanAI-Classification")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": preprocess_input,
            "random_blur": random_blur,
            "random_noise": random_noise,
        },
        compile=False,
        safe_mode=False,
    )
    with mlflow.start_run(run_name="InceptionV3-Fine-Tuning-Registration"):
        mlflow.log_params({
            "architecture": "InceptionV3",
            "model_phase": "Fine-Tuning",
            "image_size": "299x299",
            "number_of_classes": len(class_names),
            "model_format": "Keras",
            "evaluation_split": "Test",
        })
        mlflow.log_metrics({
            "test_accuracy": float(row["Accuracy"]),
            "test_loss": float(row["Loss"]),
            "test_precision_macro": float(row["Precision (macro)"]),
            "test_recall_macro": float(row["Recall (macro)"]),
            "test_f1_macro": float(row["F1-score (macro)"]),
        })
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(class_names_path))
        model_info = mlflow.keras.log_model(
            model=model,
            name="inceptionv3_classifier",
            registered_model_name="BrainScanAI-InceptionV3",
            await_registration_for=300,
            pip_requirements=[
                f"tensorflow=={tf.__version__}",
                f"mlflow=={mlflow.__version__}",
                "numpy<2.2",
            ],
        )

    return model_info.model_uri


if __name__ == "__main__":
    print(register_classification_model())
