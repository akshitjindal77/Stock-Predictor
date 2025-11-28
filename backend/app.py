# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

from main.inference import predict_next_day_ensemble_api
from .schemas import PredictRequest, PredictResponse, TickerMetadata

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(body: PredictRequest) -> PredictResponse:
    """
    Endpoint used by the React frontend.
    """
    result_dict = predict_next_day_ensemble_api(
        ticker=body.ticker,
        start=body.start or "2020-01-01",
        end=body.end,
        threshold=body.threshold,
        horizon=body.horizon,
    )
    return PredictResponse(**result_dict)

@app.get("/api/metadata/{ticker}", response_model=TickerMetadata)
def get_ticker_metadata(ticker: str):
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch metadata: {e}")

    if not info:
        raise HTTPException(status_code=404, detail="Ticker not found")

    return TickerMetadata(
        ticker=ticker.upper(),
        longName=info.get("longName"),
        sector=info.get("sector"),
        marketCap=info.get("marketCap"),
        trailingPE=info.get("trailingPE"),
        dividendYield=info.get("dividendYield"),
        fiftyTwoWeekHigh=info.get("fiftyTwoWeekHigh"),
        fiftyTwoWeekLow=info.get("fiftyTwoWeekLow"),
        currency=info.get("currency"),
    )
