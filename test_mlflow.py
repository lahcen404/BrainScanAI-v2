from src.tracking.mlflow_tracker import log_experiment


log_experiment(experiment_name="BrainScanAI-teeest",
               run_name="first-teest",
               params={
                   "model": "YOLOv8",
                   "epochs":30,
                   "image_size":640,
               },
               metrics={
                "precision": 0.926,
                "recall": 0.927,
                "mAP50": 0.955,
                "mAP50_95": 0.719,
               },
               artifacts=[
                   "models/detection/best.pt",
                   "results/yolov8_test_metrics.csv"
               ],)

print("Experiiiiment logged successfull")