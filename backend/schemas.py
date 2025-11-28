from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL, MSFT")
    start: Optional[str] = Field(
        None, description="Optional start date, e.g. '2020-01-01'"
    )
    end: Optional[str] = Field(
        None, description="Optional end date, e.g. '2025-11-25'. Defaults to today."
    )
    threshold: Optional[float] = Field(
        None,
        description="Optional feature threshold used in label creation (default from backend).",
    )
    horizon: Optional[int] = Field(
        None,
        description="Optional prediction horizon in days (default 1).",
    )


class PredictResponse(BaseModel):
    ticker: str
    as_of: str
    prediction_for: Optional[str] = None 
    start: str
    end: str
    p_up: float
    p_down: float
    signal: str
    config: Dict[str, Any]
    meta: Dict[str, Any]
    history: Optional[List[Dict[str, Any]]] = None  
    metrics: Optional[Dict[str, Any]] = None


class TickerMetadata(BaseModel):
    ticker: str
    longName: Optional[str] = None
    sector: Optional[str] = None
    marketCap: Optional[float] = None
    trailingPE: Optional[float] = None
    dividendYield: Optional[float] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None
    currency: Optional[str] = None
