
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.tracking.mlflow_tracker import log_experiment


# ==================================================
# 1. Project configuration
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "datasets" / "classification_clean"
MODEL_DIR = ROOT_DIR / "models" / "classification"

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 299
BATCH_SIZE = 16

FEATURE_EXTRACTION_EPOCHS = 20
FINE_TUNING_EPOCHS = 15

FINE_TUNE_LAYERS = 30

SEED = 42


# ==================================================
# 2. Load and validate class names
# ==================================================

def load_class_names():

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Invalid class_names.json")

    return class_names


# ==================================================
# 3. Load datasets
# ==================================================

def load_datasets(class_names):

    train_dir = DATASET_DIR / "Train"
    val_dir = DATASET_DIR / "Val"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=autotune
    )

    val_dataset = val_dataset.prefetch(
        buffer_size=autotune
    )

    return train_dataset, val_dataset


# ==================================================
# 4. Build InceptionV3 model
# ==================================================

def build_model(number_of_classes):

    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="input_image",
    )

    x = data_augmentation(inputs)

    # InceptionV3 preprocessing: scale pixels from
    # [0, 255] to [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
        name="inception_preprocessing",
    )(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainScanAI-InceptionV3",
    )

    return model, base_model


# ==================================================
# 5. Compile model
# ==================================================

def compile_model(model, learning_rate):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ==================================================
# 6. Create callbacks
# ==================================================

def create_callbacks(checkpoint_path):

    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ==================================================
# 7. Log experiment in MLflow
# ==================================================

def log_training_phase(
    run_name,
    phase,
    epochs,
    learning_rate,
    history,
    model_path,
):

    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = max(history.history["val_accuracy"])

    metrics = {
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
    }

    params = {
        "architecture": "InceptionV3",
        "phase": phase,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "fine_tune_layers": FINE_TUNE_LAYERS,
    }

    log_experiment(
        experiment_name="BrainScanAI-Classification",
        run_name=run_name,
        params=params,
        metrics=metrics,
        artifacts=[str(model_path)],
    )

    return metrics


# ==================================================
# 8. Train classification model
# ==================================================

def train_classification():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    feature_checkpoint = (
        MODEL_DIR / "inceptionv3_feature_extraction_best.keras"
    )

    finetuned_checkpoint = (
        MODEL_DIR / "inceptionv3_finetuned_v2_best.keras"
    )

    class_names = load_class_names()

    print(f"Classes: {class_names}")
    print(f"Dataset: {DATASET_DIR}")

    train_dataset, val_dataset = load_datasets(class_names)

    model, base_model = build_model(
        number_of_classes=len(class_names)
    )

    # --------------------------------------------------
    # Phase 1: Feature extraction
    # --------------------------------------------------

    print("\nStarting Phase 1: Feature Extraction")

    feature_learning_rate = 1e-3

    compile_model(
        model,
        learning_rate=feature_learning_rate,
    )

    feature_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FEATURE_EXTRACTION_EPOCHS,
        callbacks=create_callbacks(feature_checkpoint),
    )

    feature_metrics = log_training_phase(
        run_name="InceptionV3-Feature-Extraction",
        phase="feature_extraction",
        epochs=FEATURE_EXTRACTION_EPOCHS,
        learning_rate=feature_learning_rate,
        history=feature_history,
        model_path=feature_checkpoint,
    )

    print("\nFeature extraction completed.")
    print(feature_metrics)

    # --------------------------------------------------
    # Phase 2: Fine-tuning
    # --------------------------------------------------

    print("\nStarting Phase 2: Fine-Tuning")

    # Load the best feature extraction checkpoint.
    model = tf.keras.models.load_model(
        feature_checkpoint
    )

    base_model = model.get_layer("inception_v3")

    base_model.trainable = True

    # Freeze all layers first.
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze only the last selected layers.
    for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = True

    fine_tuning_learning_rate = 1e-5

    compile_model(
        model,
        learning_rate=fine_tuning_learning_rate,
    )

    fine_tuning_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNING_EPOCHS,
        callbacks=create_callbacks(finetuned_checkpoint),
    )

    finetuned_metrics = log_training_phase(
        run_name="InceptionV3-Fine-Tuning",
        phase="fine_tuning",
        epochs=FINE_TUNING_EPOCHS,
        learning_rate=fine_tuning_learning_rate,
        history=fine_tuning_history,
        model_path=finetuned_checkpoint,
    )

    print("\nFine-tuning completed.")
    print(finetuned_metrics)

    print("\nClassification training completed!")
    print(f"Feature extraction model: {feature_checkpoint}")
    print(f"Fine-tuned model: {finetuned_checkpoint}")

    return {
        "feature_extraction_model": feature_checkpoint,
        "finetuned_model": finetuned_checkpoint,
        "feature_extraction_metrics": feature_metrics,
        "finetuned_metrics": finetuned_metrics,
    }


# ==================================================
# 9. Script entry point
# ==================================================

if __name__ == "__main__":
    train_classification()
