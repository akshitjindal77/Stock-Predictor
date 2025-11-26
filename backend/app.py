# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main.inference import predict_next_day_ensemble_api

app = FastAPI(title="Stock Predictor API")

# ---- CORS so React (Vite) can talk to FastAPI ----
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    ticker: str
    start: str | None = None
    end: str | None = None
    threshold: float = 0.002
    horizon: int = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def api_predict(body: PredictRequest):
    """
    Endpoint used by the React frontend.
    """
    result = predict_next_day_ensemble_api(
        ticker=body.ticker,
        start=body.start or "2020-01-01",
        end=body.end,
        threshold=body.threshold,
        horizon=body.horizon,
    )
    return result
