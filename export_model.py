import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/breast-cancer-clf@champion",
    dst_path="model",
)
print("Exported champion model to ./model")