# %% [markdown]
# # 02 — Backtest Analysis
#
# After training a model with `scripts/train_mlflow.py`, use this notebook to:
# - Load results and inspect trade-by-trade P&L
# - Compare DQN vs buy-and-hold on multiple tickers
# - Visualise drawdown periods

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.downloader import download
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.backtest.engine import BacktestEngine

# %% [markdown]
# ## Config

# %%
TICKER       = "^TWII"
MODEL_PATH   = "../models/dqn_TWII.keras"   # adjust after training
TEST_START   = "2023-01-01"
TEST_END     = "2024-12-31"
CAPITAL      = 1_000_000

# %% [markdown]
# ## Load data & model

# %%
df = download(TICKER, start=TEST_START, end=TEST_END)
df = add_all(df)
prices = df["Close"].tolist()
dates  = df.index.tolist()
macd   = to_macd_series(prices)

agent = DQNAgent()
try:
    agent.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Could not load model ({e}) — using untrained agent for demo")

# %% [markdown]
# ## Run backtest

# %%
buys, sells, portfolio = agent.backtest(prices, macd=macd)

engine = BacktestEngine(
    initial_capital=CAPITAL,
    stop_loss_pct=0.05,
    max_position_pct=0.20,
)
result = engine.run(prices, buys, sells, ticker=TICKER, model_name="DQN")
bnh    = engine.buy_and_hold_return(prices)

print(result.summary())
print(f"  Buy-and-hold  : {bnh:+.2f}%")
print(f"  Alpha         : {result.metrics['total_return_pct'] - bnh:+.2f}%")

# %% [markdown]
# ## Equity curve + drawdown

# %%
pv  = pd.Series(result.portfolio_values, index=dates[:len(result.portfolio_values)])
bnh_curve = pd.Series([CAPITAL * (p / prices[0]) for p in prices], index=dates)
drawdown  = (pv / pv.cummax() - 1) * 100

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    subplot_titles=("Equity curve", "Drawdown (%)"),
    vertical_spacing=0.08,
)
fig.add_trace(go.Scatter(x=pv.index, y=pv.values, name="DQN",
    line=dict(color="#534AB7", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=bnh_curve.index, y=bnh_curve.values, name="Buy & hold",
    line=dict(color="#888780", width=1, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown",
    fill="tozeroy", line=dict(color="#D85A30", width=1),
    fillcolor="rgba(216,90,48,0.15)"), row=2, col=1)
fig.add_hline(y=result.metrics["max_drawdown_pct"], line_dash="dot",
    line_color="#D85A30", opacity=0.6, row=2, col=1)

fig.update_layout(title=f"{TICKER} — backtest analysis", height=600)
fig.show()

# %% [markdown]
# ## Trade log

# %%
df_trades = pd.DataFrame(result.trades)
if not df_trades.empty:
    df_trades["pnl_pct"] = (df_trades["pnl"] / (df_trades["entry_price"] * df_trades["units"]) * 100).round(2)
    print(f"\nTotal trades: {len(df_trades)}")
    print(f"Winners: {(df_trades['pnl'] > 0).sum()}  Losers: {(df_trades['pnl'] < 0).sum()}")
    print(f"Avg P&L per trade: {df_trades['pnl'].mean():+.2f}")
    df_trades[["type","t","price","units","entry_price","pnl","pnl_pct"]].tail(20)

# %% [markdown]
# ## Monthly return heatmap

# %%
monthly = pv.resample("ME").last().pct_change().dropna() * 100
monthly_df = monthly.to_frame("return")
monthly_df["year"]  = monthly_df.index.year
monthly_df["month"] = monthly_df.index.month

pivot = monthly_df.pivot(index="year", columns="month", values="return")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig2 = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale=[[0,"#D85A30"],[0.5,"#F1EFE8"],[1,"#1D9E75"]],
    zmid=0,
    text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    colorbar=dict(title="Return %"),
))
fig2.update_layout(title="Monthly returns heatmap", height=300)
fig2.show()
