// src/App.jsx
import { useState } from "react";
import { fetchPrediction, EXAMPLE_TICKERS } from "./api/predictions";
import "./App.css";

function formatPercent(value) {
  if (value == null || Number.isNaN(value)) return "--";
  return (value * 100).toFixed(2) + "%";
}

function formatDate(dateStr) {
  if (!dateStr) return "--";
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString("en-CA");
}

function getSignalClass(signal) {
  if (!signal) return "";
  const s = signal.toUpperCase();
  if (s.includes("STRONG BUY") || (s.includes("BUY") && !s.includes("WEAK"))) {
    return "signal-bullish";
  }
  if (s.includes("WEAK") || s.includes("HOLD")) {
    return "signal-neutral";
  }
  return "signal-bearish";
}

function getRecommendationText(result) {
  if (!result) return "";
  const { signal, p_up } = result;
  const p = (p_up * 100).toFixed(1);

  if (signal === "STRONG BUY") {
    return `High-conviction long setup. Model sees ~${p}% chance of an up move tomorrow.`;
  }
  if (signal === "BUY") {
    return `Moderately bullish. Edge is in favour of going long, but position sizing and risk management still matter.`;
  }
  if (signal && signal.toUpperCase().includes("WEAK")) {
    return `Slight bullish tilt, but conviction is low. This is more of a hold / small position scenario.`;
  }
  return `No clear edge detected — either stay flat or follow your own short/hedging plan if you have one.`;
}

function getConfidenceInfo(p_up) {
  if (p_up == null) return null;

  // distance from 50% => 0 .. 0.5
  const distance = Math.abs(p_up - 0.5);
  const percent = Math.min(1, distance / 0.5) * 100;

  let level = "LOW";
  if (distance >= 0.12) level = "HIGH";
  else if (distance >= 0.06) level = "MEDIUM";

  return { level, percent };
}

function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [start, setStart] = useState("2010-01-01");
  const [end, setEnd] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hasPrediction = !!result;
  const history = result?.history ?? [];
  const confidence = result ? getConfidenceInfo(result.p_up) : null;
  const histAccuracy = result?.metrics?.recent_history_accuracy ?? null;
  const historyWindow = result?.metrics?.history_window ?? history.length ?? 0;

  const bullishDays = history.filter((d) => d.p_up >= 0.5).length;
  const totalDays = history.length || 1;
  const bullishPct = (bullishDays / totalDays) * 100;

  const handleExampleClick = (symbol) => {
    setTicker(symbol);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const trimmedTicker = ticker.trim().toUpperCase();
    if (!trimmedTicker) {
      setError("Please enter a ticker symbol (e.g., AAPL, TSLA).");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const data = await fetchPrediction(
        trimmedTicker,
        start || undefined,
        end || undefined
      );
      setResult(data);
    } catch (err) {
      console.error(err);
      if (err.response) {
        setError(
          err.response.data.detail || "Server error while fetching prediction."
        );
      } else {
        setError("Network error — could not reach backend API.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <h1 className="app-title">Stock Predictor</h1>
        <p className="app-subtitle">
          Ensemble model predicting next-day move (up / down).
        </p>
      </header>

      {/* 
        Layout:
        - Without prediction: form card is centered.
        - With prediction: form slides to the left, result card appears on the right.
      */}
      <main className={hasPrediction ? "app-main has-result" : "app-main"}>
        {/* Form card */}
        <section className="card form-card">
          <h2 className="card-title">Get Prediction</h2>

          <form className="prediction-form" onSubmit={handleSubmit}>
            <div className="form-row">
              <label>
                Ticker
                <input
                  type="text"
                  className="input"
                  placeholder="AAPL, MSFT, TSLA..."
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value)}
                />
              </label>
            </div>

            <div className="form-row two-col">
              <label>
                Start date
                <input
                  type="date"
                  className="input"
                  value={start}
                  onChange={(e) => setStart(e.target.value)}
                />
              </label>

              <label>
                End date (optional)
                <input
                  type="date"
                  className="input"
                  value={end}
                  onChange={(e) => setEnd(e.target.value)}
                />
              </label>
            </div>

            <div className="example-tickers">
              <span className="example-label">Try:</span>
              {EXAMPLE_TICKERS.map((sym) => (
                <button
                  key={sym}
                  type="button"
                  className="chip"
                  onClick={() => handleExampleClick(sym)}
                >
                  {sym}
                </button>
              ))}
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Predicting..." : "Predict Next Day"}
            </button>

            {error && <p className="error-text">{error}</p>}
          </form>
        </section>

        {/* Result card */}
        {hasPrediction && (
          <section className="card result-card">
            <h2 className="card-title">Prediction Result</h2>

            <div className="result-header">
              <div>
                <p className="result-ticker">{result.ticker}</p>
                <div className="signal-row">
                  <span
                    className={`signal-chip ${getSignalClass(result.signal)}`}
                  >
                    {result.signal || "NO SIGNAL"}
                  </span>
                  {confidence && (
                    <span className="confidence-label">
                      Confidence: <strong>{confidence.level}</strong>
                    </span>
                  )}
                </div>
              </div>

              <div className="prediction-dates">
                <p className="date-label">
                  Data from{" "}
                  <strong>{formatDate(result.start)}</strong> to{" "}
                  <strong>{formatDate(result.end)}</strong>
                </p>
                <p className="date-label">
                  As of: <strong>{formatDate(result.as_of)}</strong>
                </p>
                {result.prediction_for && (
                  <p className="date-label">
                    Prediction for:{" "}
                    <strong>{formatDate(result.prediction_for)}</strong>
                  </p>
                )}
              </div>
            </div>

            {/* Trade recommendation card */}
            <p className="trade-recommendation">
              {getRecommendationText(result)}
            </p>

            {/* P(up) / P(down) */}
            <div className="result-grid">
              <div className="prob-card">
                <h3>P(up)</h3>
                <p className="prob-value">{formatPercent(result.p_up)}</p>
              </div>
              <div className="prob-card">
                <h3>P(down)</h3>
                <p className="prob-value">{formatPercent(result.p_down)}</p>
              </div>
            </div>

            {/* Confidence meter */}
            {confidence && (
              <div className="confidence-meter-section">
                <div className="confidence-meter-label-row">
                  <span>Model confidence</span>
                  <span>{confidence.level}</span>
                </div>
                <div className="confidence-meter">
                  <div
                    className={`confidence-meter-fill confidence-${confidence.level.toLowerCase()}`}
                    style={{ width: `${confidence.percent.toFixed(0)}%` }}
                  />
                </div>
                <p className="confidence-hint">
                  Confidence is based on how far P(up) is from 50%. The further
                  away, the stronger the conviction.
                </p>
              </div>
            )}

            {/* 7-day rolling predictions + sentiment + historical accuracy */}
            {history.length > 0 && (
              <div className="history-section">
                <h3 className="history-title">7-day rolling predictions</h3>

                {/* mini "chart" for P(up) per day */}
                <div className="history-bars">
                  {history.map((day) => {
                    const bullish = day.p_up >= 0.5;
                    const height = Math.max(8, Math.round(day.p_up * 100));
                    const d = new Date(day.date);
                    const label = Number.isNaN(d.getTime())
                      ? day.date
                      : d.toLocaleDateString("en-CA", {
                          month: "2-digit",
                          day: "2-digit",
                        });

                    return (
                      <div key={day.date} className="history-bar-wrapper">
                        <div
                          className={bullish ? "history-bar bullish" : "history-bar bearish"}
                          style={{ height: `${height}%` }}
                          title={`Date: ${day.date}\nP(up): ${(day.p_up * 100).toFixed(
                            1
                          )}%\nActual: ${day.actual_label === 1 ? "UP" : "DOWN"}`}
                        />
                        <span className="history-bar-date">{label}</span>
                      </div>
                    );
                  })}
                </div>

                <div className="history-metrics">
                  {/* Sentiment bar */}
                  <div className="sentiment-card">
                    <p className="metric-label">Ticker sentiment (last 7 days)</p>
                    <div className="sentiment-bar">
                      <div
                        className="sentiment-segment bullish"
                        style={{ width: `${bullishPct}%` }}
                      />
                      <div
                        className="sentiment-segment bearish"
                        style={{ width: `${100 - bullishPct}%` }}
                      />
                    </div>
                    <p className="sentiment-caption">
                      Bullish days: {bullishDays} / {history.length}
                    </p>
                  </div>

                  {/* Historical accuracy */}
                  {histAccuracy != null && (
                    <div className="accuracy-card">
                      <p className="metric-label">Historical accuracy</p>
                      <div className="accuracy-meter">
                        <div
                          className="accuracy-meter-fill"
                          style={{
                            width: `${(histAccuracy * 100).toFixed(0)}%`,
                          }}
                        />
                      </div>
                      <p className="accuracy-caption">
                        Correct on <strong>{(histAccuracy * 100).toFixed(1)}%</strong> of
                        the last {historyWindow} days.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Feature list */}
            <details className="feature-details">
              <summary>Show feature names</summary>
              <pre className="feature-list">
                {(result.meta?.feature_names || []).join("\n")}
              </pre>
            </details>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>
          This is an educational demo — <strong>not</strong> financial advice.
          Real trading requires proper risk management and backtesting.
        </p>
      </footer>
    </div>
  );
}

export default App;
