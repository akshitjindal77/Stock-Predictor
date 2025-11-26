// src/api/predictions.js
import apiClient from "./client";

// Small list to populate example buttons in the UI.
export const EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"];

// Keep the signature simple for callers: positional args instead of an object.
export async function fetchPrediction(ticker, start, end) {
  const payload = {
    ticker: ticker.toUpperCase(),
    start: "2000-01-01",  // ← always large range (safe)
    end: end || null,
    threshold: 0.002,
    horizon: 1,
  };

  const response = await apiClient.post("/api/predict", payload);
  return response.data;
}
