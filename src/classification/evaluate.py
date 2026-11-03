import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.inception_v3 import preprocess_input


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "datasets" / "classification_clean"
MODEL_DIR = ROOT_DIR / "models" / "classification"
REPORT_DIR = ROOT_DIR / "reports" / "classification"
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32


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


def evaluate_classification():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with (MODEL_DIR / "class_names.json").open(encoding="utf-8") as file:
        class_names = json.load(file)

    test_dir = DATASET_DIR / "Test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Classification test dataset not found: {test_dir}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        class_names=class_names,
    ).prefetch(tf.data.AUTOTUNE)

    models_to_evaluate = {
        "Feature Extraction": "inceptionv3_best.keras",
        "Fine-Tuning": "inceptionv3_finetuned_best.keras",
    }
    results = []

    for model_name, filename in models_to_evaluate.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        loss, accuracy = model.evaluate(test_ds, verbose=1)
        y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
        probabilities = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(probabilities, axis=1)
        report = classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

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

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            REPORT_DIR / f"{model_name.lower().replace(' ', '_')}_report.csv"
        )
        pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=range(len(class_names))),
            index=class_names,
            columns=class_names,
        ).to_csv(
            REPORT_DIR
            / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
        )

        del model
        tf.keras.backend.clear_session()

    results_path = REPORT_DIR / "classification_model_metrics.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    return results_path


if __name__ == "__main__":
    evaluate_classification()
