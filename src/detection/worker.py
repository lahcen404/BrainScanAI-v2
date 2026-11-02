import contextlib
import io
import json
import os
import sys

import mlflow
import pandas as pd


DETECTION_MODEL_URI = "models:/BrainScanAI-YOLOv8/1"


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        image_path = None
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        raise ValueError("Expected exactly one image path.")

    if image_path is not None and not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model = mlflow.pyfunc.load_model(DETECTION_MODEL_URI)
        if image_path is not None:
            results = model.predict(
                pd.DataFrame({"image_path": [image_path]})
            )

    if image_path is None:
        print("ok")
    else:
        detections = results.iloc[0]["detections"]
        print(json.dumps(detections))


if __name__ == "__main__":
    main()
