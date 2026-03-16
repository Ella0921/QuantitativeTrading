# Quantitative Trading System v2

![CI](https://github.com/Ella0921/QuantitativeTrading/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

End-to-end ML pipeline for algorithmic trading — rebuilt from a group project to demonstrate DE, MLE, and MLOps skills.

**Stack:** Python 3.11 · TensorFlow 2 / Keras 3 · Streamlit · FastAPI · MLflow · Optuna · Docker · Plotly

---

## What this project covers

| Skill area | Implementation |
|---|---|
| **Data engineering** | yfinance + Parquet cache, modular feature pipeline (MACD, RSI, BB, ATR) |
| **ML modelling** | DQN Agent (TF2 Keras) + CNN Q-Network, Ensemble signal combining |
| **MLOps** | MLflow experiment tracking, Optuna hyperparameter search, model registry |
| **Backtest / quant** | Custom engine with stop-loss, position sizing, Sharpe / MDD / Win rate / Calmar |
| **API / system design** | FastAPI REST endpoints, Pydantic schemas, health check |
| **Deployment** | Docker + docker-compose (Streamlit + FastAPI + MLflow in one stack) |
| **Testing / CI** | pytest (41+ tests, 66%+ coverage), ruff lint, GitHub Actions |

---

## Architecture

```
src/
├── data/downloader.py        # yfinance + Parquet cache
├── features/indicators.py    # MACD, RSI, Bollinger Bands, ATR
├── models/
│   ├── dqn_agent.py          # Deep Q-Network (TF2 Keras)
│   ├── cnn_agent.py          # CNN Q-Network (TF2 Keras)
│   └── ensemble.py           # DQN + CNN signal fusion
└── backtest/engine.py        # Backtest engine + performance metrics

app/streamlit_app.py          # Web UI (Signal / Train / Backtest / Compare)
api/main.py                   # FastAPI REST endpoints
scripts/
├── train_mlflow.py           # CLI training with MLflow tracking
├── tune_hyperparams.py       # Optuna hyperparameter search
└── evaluate.py               # Model evaluation + HTML chart export
notebooks/
├── 01_data_exploration.py
└── 02_backtest_analysis.py
```

---

## Quick start

```bash
pip install -r requirements.txt

# Web UI
streamlit run app/streamlit_app.py

# Train
python scripts/train_mlflow.py --ticker ^TWII --model dqn --use-macd

# Tune hyperparams
python scripts/tune_hyperparams.py --ticker ^TWII --trials 50

# Evaluate
python scripts/evaluate.py --ticker ^TWII --model-path models/dqn_TWII.keras --html-out results/

# All services
docker-compose up
# Streamlit → :8501  |  FastAPI docs → :8000/docs  |  MLflow → :5000

# Tests
pytest tests/ -v --cov=src
```

---

## Models

**DQN Agent** — Dense(256)→Dense(128)→Dense(19). Action space: hold / buy N / sell N. State: 30-day MACD or price diff window.

**CNN Q-Network** — 2× [Conv→BN→ReLU→MaxPool] → FC. Input: 32×32 binary image encoding price+volume. Output: Long / Neutral / Short.

**Ensemble** — Three strategies: `cnn_gate` (CNN as regime filter), `vote` (majority), `dqn_only` (DQN with CNN confidence gate).

---

## API

```bash
GET  /health
POST /predict/signal    {"ticker": "^TWII", "model_path": "models/cnn_TWII.keras"}
POST /predict/backtest  {"ticker": "^TWII", "start": "2023-01-01", "end": "2024-12-31"}
```

---

## What changed from v1

| | v1 | v2 |
|---|---|---|
| TensorFlow | `tf.compat.v1` + Session | TF2 Keras |
| GUI | tkinter (Windows only) | Streamlit web app |
| Charts | matplotlib static | Plotly interactive |
| Risk management | None | Stop-loss + position sizing |
| Metrics | Total gains only | Sharpe, MDD, Win rate, Calmar |
| Experiment tracking | None | MLflow |
| Hyperparameter search | None | Optuna |
| API | None | FastAPI + Pydantic |
| Signal fusion | Single model | Ensemble (3 strategies) |
| Testing | None | 41+ pytest tests |
| CI/CD | None | GitHub Actions |
| Deployment | None | Docker + docker-compose |

*Originally a 4-person group project (2023). Rebuilt as a solo end-to-end pipeline.*
