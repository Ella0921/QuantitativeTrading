# %% [markdown]
# # 01 — Data Exploration
#
# Demonstrates the data layer:
# - Downloading and caching OHLCV data
# - Computing technical indicators
# - Visual sanity checks
#
# Run as a Jupyter notebook with `jupytext` or execute cell-by-cell in VS Code.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all

# %% [markdown]
# ## Download data

# %%
TICKER = "^TWII"
df = download(TICKER, start="2019-01-01", end="2024-12-31")
df = add_all(df)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
df.tail()

# %% [markdown]
# ## Train / test split

# %%
df_train, df_test = get_train_test_split(df, train_end="2022-12-31", test_start="2023-01-01")
print(f"Train: {len(df_train)} days ({df_train.index[0].date()} → {df_train.index[-1].date()})")
print(f"Test : {len(df_test)} days  ({df_test.index[0].date()} → {df_test.index[-1].date()})")

# %% [markdown]
# ## Price + MACD chart

# %%
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=("Close price", "MACD histogram", "RSI"),
    vertical_spacing=0.06,
)

fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
    line=dict(color="#534AB7", width=1.5)), row=1, col=1)

colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in df["MACD_hist"]]
fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist",
    marker_color=colors, opacity=0.8), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_line"], name="MACD",
    line=dict(color="#534AB7", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
    line=dict(color="#D85A30", width=1)), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
    line=dict(color="#BA7517", width=1.5)), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#D85A30", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#1D9E75", opacity=0.5, row=3, col=1)

fig.update_layout(title=f"{TICKER} — price & indicators", height=700, showlegend=True)
fig.show()

# %% [markdown]
# ## Feature statistics

# %%
feature_cols = ["Close", "MACD_hist", "RSI", "BB_upper", "BB_lower", "ATR"]
df[feature_cols].describe().round(4)

# %% [markdown]
# ## Rolling correlation between MACD and next-day return

# %%
df["next_return"] = df["Close"].pct_change().shift(-1)
rolling_corr = df["MACD_hist"].rolling(60).corr(df["next_return"])

fig2 = go.Figure(go.Scatter(
    x=df.index, y=rolling_corr,
    mode="lines", name="60-day rolling corr(MACD_hist, next_return)",
    line=dict(color="#534AB7"),
))
fig2.add_hline(y=0, line_dash="dash", opacity=0.4)
fig2.update_layout(title="MACD predictive correlation (60-day rolling)", height=350)
fig2.show()
