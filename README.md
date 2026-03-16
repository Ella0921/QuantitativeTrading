# Quantitative Trading System v2

End-to-end ML pipeline for algorithmic trading, rebuilt from a group project.

**Stack:** Python · TensorFlow 2 · Streamlit · FastAPI · MLflow · Docker · Plotly

---

## Architecture

```
src/
├── data/
│   └── downloader.py      # yfinance + Parquet cache
├── features/
│   └── indicators.py      # MACD, RSI, Bollinger Bands, ATR
├── models/
│   ├── dqn_agent.py       # Deep Q-Network (TF2 Keras)
│   └── cnn_agent.py       # CNN Q-Network (TF2 Keras)
└── backtest/
    └── engine.py          # Backtest engine + Sharpe/MDD/WinRate

app/
└── streamlit_app.py       # Web UI (replaces tkinter GUI)

api/
└── main.py                # FastAPI inference endpoints

tests/
└── test_backtest.py       # pytest unit tests
```

## What changed from v1

| Area | Before | After |
|------|--------|-------|
| TensorFlow | `tf.compat.v1` + `Session` + `placeholder` | TF2 Keras subclassing |
| GUI | tkinter (Windows only) | Streamlit web app |
| Charts | matplotlib static | Plotly interactive |
| Risk management | None | Stop-loss + position sizing |
| Performance metrics | Total gains only | Sharpe, MDD, Win rate, Calmar |
| Model persistence | `tf.compat.v1.train.Saver` | `model.save()` / `model.load()` |
| Data caching | None (re-download every run) | Parquet cache |
| Experiment tracking | None | MLflow |
| API | None | FastAPI REST endpoints |
| Testing | None | pytest unit tests |
| Deployment | None | Docker + docker-compose |

## Quick start

```bash
pip install -r requirements.txt

# Web UI
streamlit run app/streamlit_app.py

# API server
uvicorn api.main:app --reload

# All services (Docker)
docker-compose up

# Tests
pytest tests/ -v
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict/signal` | CNN signal for a ticker |
| POST | `/predict/backtest` | DQN backtest over date range |

Example:
```bash
curl -X POST http://localhost:8000/predict/signal \
  -H "Content-Type: application/json" \
  -d '{"ticker": "^TWII", "model_path": "models/cnn_agent"}'
```

## Models

### DQN Agent (`src/models/dqn_agent.py`)
- Dense(256, ReLU) → Dense(128, ReLU) → Dense(19)
- Action space: 0 = hold, 1–9 = buy N units, 10–18 = sell N-9 units
- State: 30-day window of price or MACD differences

### CNN Q-Network (`src/models/cnn_agent.py`)
- Input: 32×32 binary image encoding price + volume
- 2× [Conv(5) → BN → ReLU → MaxPool(2)] → FC(32) → FC(3)
- Action space: Long / Neutral / Short

## Performance metrics

- **Sharpe ratio** — annualised risk-adjusted return
- **Max drawdown** — largest peak-to-trough decline
- **Win rate** — % of trades that were profitable
- **Calmar ratio** — annual return / max drawdown

---

*Originally a group project (2023). Rebuilt as a solo end-to-end ML pipeline to demonstrate DE, MLE, and MLOps skills.*
