
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock

import mlflow
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.inception_v3 import preprocess_input


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

CLASSIFICATION_MODEL_URI = "models:/BrainScanAI-InceptionV3/1"
DETECTION_MODEL_URI = "models:/BrainScanAI-YOLOv8/1"

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
CLASS_NAMES_PATH = MODEL_DIR / "classification" / "class_names.json"

IMAGE_SIZE = (299, 299)
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}

classification_model = None
detection_model = None
class_names = []

model_lock = Lock()


def random_blur(images):
    apply_blur = tf.random.uniform(()) < 0.3

    def blur():
        channels = tf.shape(images)[-1]
        kernel = tf.ones(
            (3, 3, channels, 1),
            dtype=images.dtype
        ) / 9.0

        return tf.nn.depthwise_conv2d(
            images,
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

    return tf.cond(apply_blur, blur, lambda: images)


def random_noise(images):
    noise = tf.random.normal(
        shape=tf.shape(images),
        mean=0.0,
        stddev=5.0,
        dtype=images.dtype
    )

    return tf.clip_by_value(
        images + noise,
        0.0,
        255.0
    )


CUSTOM_OBJECTS = {
    "preprocess_input": preprocess_input,
    "random_blur": random_blur,
    "random_noise": random_noise
}


def load_models():
    global classification_model
    global detection_model
    global class_names

    with model_lock:
        if (
            classification_model is not None
            and detection_model is not None
        ):
            return

        logger.info("Connecting to MLflow...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info("Loading class names...")

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            class_names = json.load(file)

        logger.info("Loading InceptionV3 model...")

        classification_model = mlflow.keras.load_model(
            CLASSIFICATION_MODEL_URI,
            custom_objects=CUSTOM_OBJECTS
        )

        logger.info("Checking YOLOv8 detection worker...")

        detection_check = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.detection.worker",
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        if detection_check.returncode != 0:
            logger.error(
                "Detection worker check failed with exit code %s: %s",
                detection_check.returncode,
                detection_check.stderr.strip(),
            )
            raise RuntimeError("Detection worker is not available.")

        detection_model = True
        logger.info("Both models loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BrainScanAI API...")
    load_models()
    yield
    logger.info("Shutting down BrainScanAI API...")


app = FastAPI(
    title="BrainScanAI API",
    description="Brain MRI classification and tumor detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


async def validate_image(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are accepted."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    try:
        image = Image.open(BytesIO(contents))

        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format."
            )

        image.verify()

        image = Image.open(BytesIO(contents)).convert("RGB")

        return image

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is invalid or corrupted."
        )


def prepare_classification_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def load_detection_font(font_size: int):
    """Load a bold font for readable detection labels."""

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(
                    font_path,
                    font_size
                )
            except OSError:
                continue

    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list
) -> Image.Image:
    """
    Draw high-quality bounding boxes and readable labels
    without changing the original image dimensions.
    """

    image = image.copy().convert("RGB")

    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    # Adapt annotation size to the image resolution.
    scale = min(image_width, image_height)

    line_width = max(3, round(scale / 250))
    font_size = max(20, round(scale / 25))
    padding = max(6, round(font_size / 5))

    font = load_detection_font(font_size)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        x1 = max(0, min(image_width - 1, round(x1)))
        y1 = max(0, min(image_height - 1, round(y1)))
        x2 = max(0, min(image_width - 1, round(x2)))
        y2 = max(0, min(image_height - 1, round(y2)))

        class_name = str(detection["class_name"])
        confidence = float(detection["confidence"])

        label = f"{class_name}: {confidence:.1%}"

        # Draw the bounding box.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="#FF2020",
            width=line_width
        )

        # Calculate label dimensions.
        text_box = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_width = text_width + (padding * 2)
        label_height = text_height + (padding * 2)

        # Place the label above the box when possible.
        label_y = y1 - label_height

        # If there isn't enough room, place it inside the box.
        if label_y < 0:
            label_y = y1

        # Prevent the label from going outside the image.
        label_x = min(
            x1,
            max(0, image_width - label_width)
        )

        # Draw a solid red label background.
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height
            ),
            fill="#FF2020"
        )

        # Draw bold white text.
        draw.text(
            (
                label_x + padding,
                label_y + padding - text_box[1]
            ),
            label,
            fill="white",
            font=font
        )

    return image


@app.get("/")
def root():
    return {
        "message": "Welcome to BrainScanAI API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "classification_model_loaded": classification_model is not None,
        "detection_model_loaded": detection_model is not None
    }


@app.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...)
):
    if classification_model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model is not available."
        )

    image = await validate_image(file)
    image_array = prepare_classification_image(image)

    try:
        predictions = classification_model.predict(
            image_array,
            verbose=0
        )

        predictions = np.asarray(predictions)[0]

        if (
            len(predictions) != len(class_names)
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError("Invalid model prediction output.")

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]

        confidence = float(predictions[predicted_index]) * 100

        probabilities = {
            class_names[index]: round(
                float(probability) * 100,
                2
            )
            for index, probability in enumerate(predictions)
        }

        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Classification prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Classification prediction failed."
        )


@app.post("/predict/detection")
async def predict_detection(
    file: UploadFile = File(...)
):
    if detection_model is None:
        raise HTTPException(
            status_code=503,
            detail="Detection model is not available."
        )

    image = await validate_image(file)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, format="JPEG")

        detection_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.detection.worker",
                temp_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        if detection_process.returncode != 0:
            logger.error(
                "Detection worker failed with exit code %s: %s",
                detection_process.returncode,
                detection_process.stderr.strip(),
            )
            raise RuntimeError("Detection worker failed.")

        detections = json.loads(detection_process.stdout)

        # Generate high-quality annotations.
        annotated_image = draw_detections(
            image,
            detections
        )

        image_buffer = BytesIO()

        annotated_image.save(
            image_buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True
        )

        encoded_image = base64.b64encode(
            image_buffer.getvalue()
        ).decode("utf-8")

        return {
            "success": True,
            "filename": file.filename,
            "number_of_detections": len(detections),
            "detections": detections,
            "annotated_image": encoded_image,
            "image_format": "jpeg"
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Detection prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Detection prediction failed."
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
