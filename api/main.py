"""
FastAPI inference service.

Endpoints:
  POST /predict/signal   — CNN signal for a ticker
  POST /predict/backtest — DQN backtest over a date range
  GET  /health           — health check

Run:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.downloader import download
from src.features.indicators import add_all, to_macd_series
from src.models.cnn_agent import CNNAgent, prepare_inference_matrix
from src.models.dqn_agent import DQNAgent
from src.backtest.engine import BacktestEngine

app = FastAPI(
    title="Quantitative Trading API",
    description="DQN + CNN Q-Network inference endpoints",
    version="2.0.0",
)

# ── Request / Response models ─────────────────────────────────────────────────

class SignalRequest(BaseModel):
    ticker: str
    model_path: str = "models/cnn_agent"


class SignalResponse(BaseModel):
    ticker: str
    signal: str
    action: int
    q_values: dict[str, float]
    timestamp: str


class BacktestRequest(BaseModel):
    ticker: str
    start: str
    end: str
    model_path: str = "models/dqn_agent"
    use_macd: bool = True
    initial_capital: float = 1_000_000
    stop_loss_pct: float = 0.05
    max_position_pct: float = 0.20


class BacktestResponse(BaseModel):
    ticker: str
    metrics: dict
    buy_and_hold_return_pct: float
    num_buy_signals: int
    num_sell_signals: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/predict/signal", response_model=SignalResponse)
def predict_signal(req: SignalRequest):
    """Return CNN Long/Neutral/Short signal for today."""
    try:
        matrix = prepare_inference_matrix(req.ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data error: {e}")

    agent = CNNAgent()
    mp = Path(req.model_path)
    if mp.exists():
        try:
            agent.load(req.model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    result = agent.predict_signal(matrix)
    return SignalResponse(
        ticker=req.ticker,
        signal=result["signal"],
        action=result["action"],
        q_values=result["q_values"],
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    """Run DQN backtest over a historical period."""
    try:
        df = download(req.ticker, start=req.start, end=req.end)
        df = add_all(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data error: {e}")

    prices = df["Close"].tolist()
    macd = to_macd_series(prices) if req.use_macd else None

    agent = DQNAgent()
    mp = Path(req.model_path)
    if mp.exists():
        try:
            agent.load(req.model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    buys, sells, _ = agent.backtest(prices, macd=macd)

    engine = BacktestEngine(
        initial_capital=req.initial_capital,
        stop_loss_pct=req.stop_loss_pct,
        max_position_pct=req.max_position_pct,
    )
    result = engine.run(prices, buys, sells, ticker=req.ticker)
    bnh = engine.buy_and_hold_return(prices)

    return BacktestResponse(
        ticker=req.ticker,
        metrics=result.metrics,
        buy_and_hold_return_pct=round(bnh, 4),
        num_buy_signals=len(buys),
        num_sell_signals=len(sells),
    )
