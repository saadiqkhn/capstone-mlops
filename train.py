import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("breast-cancer-capstone")

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for n in [50, 200]:
    with mlflow.start_run(run_name=f"rf-{n}"):
        params = {"n_estimators": n, "max_depth": 5, "random_state": 42}
        model = RandomForestClassifier(**params).fit(X_train, y_train)
        preds = model.predict(X_test)
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        })
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature, skops_trusted_types=["sklearn.tree._tree.Tree"])
