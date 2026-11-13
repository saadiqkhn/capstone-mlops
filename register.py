import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5001")
client = MlflowClient()
experiment = client.get_experiment_by_name("breast-cancer-capstone")

# Selection rule: highest accuracy wins; on a tie, the fewest trees wins.
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
)
best_accuracy = max(r.data.metrics["accuracy"] for r in runs)
tied = [r for r in runs if r.data.metrics["accuracy"] == best_accuracy]
champion_run = min(tied, key=lambda r: int(r.data.params["n_estimators"]))

version = mlflow.register_model(
    f"runs:/{champion_run.info.run_id}/model", "breast-cancer-clf")
client.set_registered_model_alias(
    "breast-cancer-clf", "champion", version.version)

print(f"Champion: {champion_run.info.run_name} "
      f"(n_estimators={champion_run.data.params['n_estimators']}), "
      f"accuracy {best_accuracy:.4f}, registered as version {version.version}")