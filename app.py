import time
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
from sklearn.datasets import load_breast_cancer

COLS = load_breast_cancer().feature_names
model = mlflow.pyfunc.load_model("./model")

REQUESTS = Counter("predict_requests", "Prediction requests received")
LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds")

app = FastAPI()
app.mount("/metrics", make_asgi_app())

class Request(BaseModel):
    rows: list[list[float]]

@app.post("/predict")
def predict(req: Request):
    REQUESTS.inc()
    start = time.time()
    df = pd.DataFrame(req.rows, columns=COLS)
    preds = model.predict(df).tolist()
    LATENCY.observe(time.time() - start)
    return {"predictions": preds}