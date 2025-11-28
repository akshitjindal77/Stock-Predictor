import apiClient from "./client";

// Small list to populate example buttons in the UI.
export const EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"];

// Keep the signature simple for callers: positional args instead of an object.
export async function fetchPrediction(ticker, start, end, horizon = 1) {
  const payload = {
    ticker: ticker.toUpperCase(),
    start: start || "2020-01-01",
    end: end || null, // backend treats None as "today"
    threshold: 0.002,
    horizon,          // <- now dynamic
  };

  const response = await apiClient.post("/api/predict", payload);
  return response.data;
}

export async function fetchMetadata(ticker) {
  const response = await apiClient.get(`/api/metadata/${ticker.toUpperCase()}`);
  return response.data;
}
