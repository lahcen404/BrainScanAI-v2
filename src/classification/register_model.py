
import json
import sys
from pathlib import Path

import mlflow
import mlflow.keras
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import preprocess_input


# ==================================================
# 1. Custom functions used by the saved model
# ==================================================

def random_blur(images):
    apply_blur = tf.random.uniform(()) < 0.3

    def blur():
        channels = tf.shape(images)[-1]

        kernel = tf.ones(
            (3, 3, channels, 1),
            dtype=images.dtype,
        ) / 9.0

        return tf.nn.depthwise_conv2d(
            images,
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

    return tf.cond(apply_blur, blur, lambda: images)


def random_noise(images):
    noise = tf.random.normal(
        shape=tf.shape(images),
        mean=0.0,
        stddev=5.0,
        dtype=images.dtype,
    )

    return tf.clip_by_value(
        images + noise,
        0.0,
        255.0,
    )


# ==================================================
# 2. Project configuration
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "classification"
    / "inceptionv3_finetuned_best.keras"
)

CLASS_NAMES_PATH = (
    ROOT_DIR
    / "models"
    / "classification"
    / "class_names.json"
)

TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "BrainScanAI-Classification"
REGISTERED_MODEL_NAME = "BrainScanAI-InceptionV3"

RUN_NAME = "InceptionV3-Fine-Tuning-Registration"


# ==================================================
# 3. Validate files
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(CLASS_NAMES_PATH)

with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
    class_names = json.load(file)

print("Classes:", class_names)
print("Model:", MODEL_PATH)


# ==================================================
# 4. Configure MLflow
# ==================================================

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ==================================================
# 5. Load the fine-tuned model
# ==================================================

print("\nLoading fine-tuned InceptionV3 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input,
        "random_blur": random_blur,
        "random_noise": random_noise,
    },
    compile=False,
    safe_mode=False,
)

print("Model loaded successfully!")


# ==================================================
# 6. Register the model in MLflow
# ==================================================

with mlflow.start_run(run_name=RUN_NAME):

    mlflow.log_param("architecture", "InceptionV3")
    mlflow.log_param("training_phase", "Fine-Tuning")
    mlflow.log_param("image_size", "299x299")
    mlflow.log_param("number_of_classes", len(class_names))
    mlflow.log_param("model_format", "Keras")
    mlflow.log_param("evaluation_accuracy", 0.9302)

    mlflow.log_metric("test_accuracy", 0.9302)
    mlflow.log_metric("test_loss", 0.1916)
    mlflow.log_metric("test_precision_macro", 0.9288)
    mlflow.log_metric("test_recall_macro", 0.9302)
    mlflow.log_metric("test_f1_macro", 0.9289)

    mlflow.log_artifact(str(CLASS_NAMES_PATH))

    print("\nLogging and registering model...")

    model_info = mlflow.keras.log_model(
        model=model,
        name="inceptionv3_classifier",
        registered_model_name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
        pip_requirements=[
            f"tensorflow=={tf.__version__}",
            f"mlflow=={mlflow.__version__}",
            "numpy<2.2",
        ],
    )

    print("\nModel registration completed!")
    print("Model URI:", model_info.model_uri)
    print("Model ID:", model_info.model_id)


# ==================================================
# 7. Check registry versions
# ==================================================

client = mlflow.tracking.MlflowClient(
    tracking_uri=TRACKING_URI,
)

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

print("\nRegistered versions:")

for version in versions:
    print(
        f"Version: {version.version} | "
        f"Status: {version.status} | "
        f"Run ID: {version.run_id}"
    )

print("\nRegistration script completed!")
