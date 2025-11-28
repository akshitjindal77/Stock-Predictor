import pytest
from main.inference import predict_next_day_ensemble_api

def test_predict_next_day_smoke():
    result = predict_next_day_ensemble_api(
        "AAPL", start="2023-01-01", end="2023-12-31", horizon=1
    )
    assert "p_up" in result
    assert "p_down" in result
    assert "history" in result
    assert result["ticker"] == "AAPL"
