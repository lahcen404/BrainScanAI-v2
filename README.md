# BrainScanAI-V2

BrainScanAI-V2 is an end-to-end brain MRI machine-learning project for brain
tumor classification and object detection.

The project includes:

- InceptionV3 image classification with TensorFlow/Keras.
- YOLOv8 object detection with Ultralytics.
- A FastAPI inference API.
- A Streamlit user interface.
- MLflow experiment tracking and model registry.
- Apache Airflow orchestration for evaluation and model registration.
- Docker Compose services for local development.

The project reuses existing trained models. The Airflow workflow evaluates and
registers existing checkpoints; it does not train, fine-tune, overwrite, or
retrain models.

## Technology stack

### Application

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- Python Multipart
- Pillow
- OpenCV

### Machine learning

- TensorFlow 2.21.0
- Keras
- InceptionV3
- PyTorch
- Ultralytics YOLO 8.4.183
- NumPy
- Pandas
- scikit-learn

### MLOps and orchestration

- MLflow 3.17.2
- Apache Airflow 2.9.1
- Airflow `PythonOperator`
- Airflow `LocalExecutor`

### Infrastructure

- Docker
- Docker Compose
- PostgreSQL 16
- SQLite for the local MLflow backend store
- pgAdmin

### User interfaces

- Streamlit
- FastAPI Swagger/OpenAPI

## Project structure

```text
BrainScanAI-V2/
├── api/
│   └── main.py
├── airflow/
│   ├── dags/
│   │   └── brainscan_pipeline.py
│   └── Dockerfile
├── app/
│   └── streamlit_app.py
├── datasets/
│   ├── classification_clean/
│   └── detection_yolo/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   └── Dockerfile.streamlit
├── models/
│   ├── classification/
│   └── detection/
├── reports/
├── src/
│   ├── classification/
│   ├── detection/
│   ├── data/
│   └── tracking/
├── requirements/
├── compose.yaml
└── README.md
```

## Existing models

### Classification

Classification models are stored in:

```text
models/classification/
```

The evaluation workflow uses:

```text
models/classification/inceptionv3_best.keras
models/classification/inceptionv3_finetuned_best.keras
```

The available class names are stored in:

```text
models/classification/class_names.json
```

### Object detection

The YOLOv8 model is:

```text
models/detection/best.pt
```

The Docker-compatible YOLO dataset configuration is:

```text
datasets/detection_yolo/data.yaml
```

Do not use a dataset YAML containing host-specific paths such as
`/home/<user>/...` inside Docker.

## Prerequisites

Install:

- Docker Engine
- Docker Compose
- Git
- At least 8 GB of available RAM is recommended.

The API image installs TensorFlow, PyTorch, and Ultralytics. The initial image
build may take several minutes and requires significant disk space.

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd BrainScanAI-V2
```

### Configure environment variables

Create the environment file:

```bash
cp .env.example .env
```

Update `.env` with local credentials and configuration. Never commit `.env` or
share its secret values.

Important service URLs:

```dotenv
MLFLOW_TRACKING_URI="http://mlflow:5000"
API_URL="http://api:8000"
MODEL_DIR="/app/models"
DATASET_DIR="/app/datasets"
```

Inside Docker, use service names such as `mlflow`, `api`, and `postgres`.
Do not use `localhost` for communication between containers.

### Configure Airflow Docker access

Airflow runs the machine-learning evaluation and registration modules inside
the existing API container. It therefore needs access to the Docker socket.

Find the Docker socket group ID:

```bash
stat -c '%g' /var/run/docker.sock
```

Set it before starting Compose:

```bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
```

You may also add it to `.env`:

```dotenv
DOCKER_GID=966
```

Replace `966` if the Docker socket has a different group ID.

## Build and start

Build all images:

```bash
docker compose build
```

Start all services:

```bash
docker compose up -d
```

Check service status:

```bash
docker compose ps
```

Follow all logs:

```bash
docker compose logs -f
```

Rebuild only API and Airflow:

```bash
docker compose build api airflow
docker compose up -d --force-recreate api airflow
```

## Services

| Service | URL | Purpose |
|---|---|---|
| FastAPI | http://localhost:8000 | REST API |
| Swagger UI | http://localhost:8000/docs | Interactive API documentation |
| OpenAPI schema | http://localhost:8000/openapi.json | API schema |
| Streamlit | http://localhost:8501 | User interface |
| MLflow | http://localhost:5000 | Experiments and model registry |
| Airflow | http://localhost:8080 | DAG management |
| pgAdmin | http://localhost:5050 | PostgreSQL administration |

The local Airflow Compose setup creates:

```text
Username: admin
Password: admin
```

Change these credentials for any non-local deployment.

## FastAPI usage

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "classification_model_loaded": true,
  "detection_model_loaded": true
}
```

### Classification prediction

```bash
curl -X POST \
  http://localhost:8000/predict/classification \
  -H "accept: application/json" \
  -F "file=@path/to/brain-image.jpg"
```

### Detection prediction

```bash
curl -X POST \
  http://localhost:8000/predict/detection \
  -H "accept: application/json" \
  -F "file=@path/to/brain-image.jpg"
```

The detection response includes detection classes, confidence values, bounding
boxes, and a Base64-encoded annotated JPEG image.

The API executes YOLO inference in a separate worker process. This avoids
native TensorFlow and PyTorch/LLVM conflicts in the same process.

## Airflow workflow

The DAG is located at:

```text
airflow/dags/brainscan_pipeline.py
```

DAG ID:

```text
brainscanai_model_pipeline
```

The DAG is manually triggered and has no automatic schedule by default.

### Task flow

```text
check_datasets
    ├── evaluate_classification
    │       └── log_classification_results
    │               └── register_classification_model
    │
    └── evaluate_detection
            └── log_detection_results
                    └── register_detection_model
```

### Tasks

#### `check_datasets`

Verifies the required datasets, model checkpoints, class names, and YOLO
configuration.

#### `evaluate_classification`

Evaluates the existing InceptionV3 checkpoints and generates:

```text
reports/classification/classification_model_metrics.csv
reports/classification/feature_extraction_report.csv
reports/classification/feature_extraction_confusion_matrix.csv
reports/classification/fine-tuning_report.csv
reports/classification/fine-tuning_confusion_matrix.csv
```

#### `evaluate_detection`

Evaluates `models/detection/best.pt` and generates:

```text
reports/detection/yolov8_test_metrics.csv
```

#### `log_classification_results`

Reads the generated classification CSV and logs actual metrics and reports to
the `BrainScanAI-Classification` MLflow experiment.

#### `log_detection_results`

Reads the generated detection CSV and logs actual metrics and reports to the
`BrainScanAI-Detection` MLflow experiment.

#### `register_classification_model`

Registers the existing fine-tuned model as:

```text
BrainScanAI-InceptionV3
```

#### `register_detection_model`

Registers the existing YOLO model as:

```text
BrainScanAI-YOLOv8
```

The registration tasks use the current evaluation results. They do not use
hardcoded historical metrics.

## Trigger Airflow

### From the Airflow UI

1. Open http://localhost:8080.
2. Log in with the Airflow credentials.
3. Find `brainscanai_model_pipeline`.
4. Unpause the DAG.
5. Click the play button.
6. Select **Trigger DAG**.

### From the command line

```bash
docker exec BrainScanAI-v2-airflow \
  airflow dags trigger brainscanai_model_pipeline
```

List DAG runs:

```bash
docker exec BrainScanAI-v2-airflow \
  airflow dags list-runs \
  --dag-id brainscanai_model_pipeline
```

List task states for a run:

```bash
docker exec BrainScanAI-v2-airflow \
  airflow tasks states-for-dag-run \
  brainscanai_model_pipeline \
  <run-id>
```

List DAG tasks:

```bash
docker exec BrainScanAI-v2-airflow \
  airflow tasks list brainscanai_model_pipeline
```

## MLflow verification

Open:

```text
http://localhost:5000
```

Expected experiments:

```text
BrainScanAI-Classification
BrainScanAI-Detection
```

Expected registered models:

```text
BrainScanAI-InceptionV3
BrainScanAI-YOLOv8
```

Verify registered models through the MLflow API:

```bash
curl --get \
  http://localhost:5000/api/2.0/mlflow/registered-models/get \
  --data-urlencode "name=BrainScanAI-InceptionV3"

curl --get \
  http://localhost:5000/api/2.0/mlflow/registered-models/get \
  --data-urlencode "name=BrainScanAI-YOLOv8"
```

Registered versions should eventually have:

```text
status: READY
```

## Reports and artifacts

Reports are stored in:

```text
reports/
```

List generated reports:

```bash
find reports -maxdepth 3 -type f | sort
```

MLflow artifacts are stored in:

```text
artifacts/
```

The local MLflow backend database is mounted from:

```text
mlflow/mlflow.db
```

## Stop and restart

Stop containers:

```bash
docker compose stop
```

Start stopped containers:

```bash
docker compose start
```

Remove containers without deleting volumes:

```bash
docker compose down
```

Remove containers and volumes:

```bash
docker compose down -v
```

Use `down -v` carefully because it deletes PostgreSQL and Airflow metadata
stored in Docker volumes.

## Troubleshooting

### API does not start

```bash
docker logs --tail 200 BrainScanAI-v2-api
curl http://localhost:8000/health
```

The API should log:

```text
Both models loaded successfully!
Application startup complete.
```

### Detection crashes the API

The API uses a separate detection worker to isolate TensorFlow and Ultralytics.
Inspect the logs and container state:

```bash
docker logs --tail 200 BrainScanAI-v2-api
docker inspect BrainScanAI-v2-api
```

### Airflow cannot access Docker

```bash
docker exec BrainScanAI-v2-airflow id
docker exec BrainScanAI-v2-airflow docker ps
```

If permission is denied:

```bash
export DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
docker compose up -d --force-recreate airflow
```

### Airflow cannot find the DAG

```bash
docker exec BrainScanAI-v2-airflow airflow dags list
docker compose restart airflow
```

The DAG should be available at:

```text
/opt/airflow/airflow/dags/brainscan_pipeline.py
```

### Detection evaluation cannot find images

Use:

```text
datasets/detection_yolo/data.yaml
```

This file uses Docker-compatible relative paths. Do not use a YAML file with
host-specific absolute paths inside containers.

## Development rules

- Do not commit `.env` or credentials.
- Do not overwrite trained checkpoints.
- Do not add large ML dependencies to Airflow unnecessarily.
- Use `http://mlflow:5000` for container-to-container MLflow communication.
- Keep generated evaluation files under `reports/`.
- Reuse existing evaluation and registration modules.
- Do not add model-training tasks to the Airflow DAG.
