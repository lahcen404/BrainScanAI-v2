
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.applications.inception_v3 import preprocess_input


# ==================================================
# 1. Custom functions from the training notebook
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
# 2. Project paths and configuration
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "data" / "classification_clean"
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the class names
with open(
    MODEL_DIR / "class_names.json",
    encoding="utf-8",
) as file:
    class_names = json.load(file)

print("Classes:", class_names)


# ==================================================
# 3. Load evaluation dataset
# ==================================================

def load_dataset(split_name):

    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split_name,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


test_ds = load_dataset("Test")


# ==================================================
# 4. Models to evaluate
# ==================================================

models_to_evaluate = {
    "Feature Extraction": "inceptionv3_best.keras",
    "Fine-Tuning": "inceptionv3_finetuned_best.keras",
}


# ==================================================
# 5. Evaluate both saved models
# ==================================================

results = []

for model_name, filename in models_to_evaluate.items():

    print("\n" + "=" * 55)
    print(f"Evaluating: {model_name}")
    print("=" * 55)

    model_path = MODEL_DIR / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load the saved model without compiling it
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

    # Compile for evaluation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Evaluate loss and accuracy
    loss, accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    # Collect actual labels
    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    # Predict probabilities
    probabilities = model.predict(
        test_ds,
        verbose=1,
    )

    # Convert probabilities into class predictions
    y_pred = np.argmax(probabilities, axis=1)

    # Calculate detailed classification metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Store metrics for comparison
    results.append({
        "Model": model_name,
        "Model File": filename,
        "Images": len(y_true),
        "Loss": loss,
        "Accuracy": accuracy,
        "Precision (macro)": report["macro avg"]["precision"],
        "Recall (macro)": report["macro avg"]["recall"],
        "F1-score (macro)": report["macro avg"]["f1-score"],
    })

    # Save classification report
    report_df = pd.DataFrame(report).transpose()

    report_path = (
        REPORT_DIR
        / f"{model_name.lower().replace(' ', '_')}_report.csv"
    )

    report_df.to_csv(report_path)

    # Calculate confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names)),
    )

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    cm_path = (
        REPORT_DIR
        / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # Display results
    print(f"\nLoss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(
        f"Precision (macro): "
        f"{report['macro avg']['precision']:.2%}"
    )
    print(
        f"Recall (macro): "
        f"{report['macro avg']['recall']:.2%}"
    )
    print(
        f"F1-score (macro): "
        f"{report['macro avg']['f1-score']:.2%}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    print(f"Report saved: {report_path}")
    print(f"Confusion matrix saved: {cm_path}")

    # Release model resources before loading the next model
    del model
    tf.keras.backend.clear_session()


# ==================================================
# 6. Save final comparison
# ==================================================

results_df = pd.DataFrame(results)

results_path = REPORT_DIR / "classification_model_metrics.csv"

results_df.to_csv(
    results_path,
    index=False,
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Loss": "{:.4f}".format,
            "Accuracy": "{:.2%}".format,
            "Precision (macro)": "{:.2%}".format,
            "Recall (macro)": "{:.2%}".format,
            "F1-score (macro)": "{:.2%}".format,
        },
    )
)

print(f"\nComparison saved to: {results_path}")
print("\nEvaluation completed successfully!")
