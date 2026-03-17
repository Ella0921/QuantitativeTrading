"""
Baseline comparison script.

Trains and evaluates 5 strategies on the same dataset, then produces:
  results/comparison_table.csv     — full metrics table
  results/equity_curves.png        — equity curve comparison
  results/metrics_bar.png          — Sharpe / Return / MDD bar chart

These outputs are committed to the repo and embedded in README.md.

Usage:
    python scripts/run_comparison.py
    python scripts/run_comparison.py --ticker ^TWII --train-end 2022-12-31 --test-start 2023-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.baselines.strategies import (
    buy_and_hold_signals,
    sma_crossover_signals,
    macd_rsi_signals,
    rsi_mean_reversion_signals,
    BASELINE_NAMES,
)
from src.baselines.lstm_model import LSTMAgent
from src.backtest.engine import BacktestEngine

# ── Plot style ────────────────────────────────────────────────────────────────

COLORS = {
    "DQN Agent (ours)":       "#534AB7",
    "Buy & Hold":             "#888780",
    "SMA Crossover (5/20)":   "#1D9E75",
    "MACD + RSI Rules":       "#BA7517",
    "RSI Mean Reversion":     "#D85A30",
    "LSTM Baseline":          "#993556",
}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ── Run all strategies ────────────────────────────────────────────────────────

def run_all(
    ticker:      str   = "^TWII",
    train_start: str   = "2016-01-01",
    train_end:   str   = "2022-12-31",
    test_start:  str   = "2023-01-01",
    test_end:    str   = "2024-12-31",
    capital:     float = 1_000_000,
    dqn_iters:   int   = 200,
    lstm_epochs: int   = 30,
) -> tuple[dict, list[float], list]:
    """
    Train and evaluate all strategies. Returns:
      - metrics dict  {strategy_name: {metric: value}}
      - prices list   (test period)
      - dates list    (test period)
    """
    print(f"\nDownloading {ticker}...")
    df_full = download(ticker, start=train_start, end=test_end)
    df_full = add_all(df_full)
    df_train, df_test = get_train_test_split(df_full, train_end, test_start)

    prices_train = df_train["Close"].tolist()
    prices_test  = df_test["Close"].tolist()
    dates_test   = df_test.index.tolist()
    macd_train   = to_macd_series(prices_train)
    macd_test    = to_macd_series(prices_test)

    print(f"Train: {len(df_train)} days | Test: {len(df_test)} days\n")

    engine = BacktestEngine(
        initial_capital=capital,
        stop_loss_pct=0.05,
        max_position_pct=0.20,
    )

    results     = {}
    portfolios  = {}

    # ── 1. DQN Agent ─────────────────────────────────────────────────────────
    print("Training DQN Agent...")
    dqn = DQNAgent()
    dqn.train(prices_train, macd=macd_train, iterations=dqn_iters,
              initial_money=capital, checkpoint=dqn_iters)
    Path("models").mkdir(exist_ok=True)
    dqn.save("models/dqn_pretrained")

    buys, sells, _ = dqn.backtest(prices_test, macd=macd_test)
    r = engine.run(prices_test, buys, sells, ticker=ticker, model_name="DQN")
    results["DQN Agent (ours)"]    = r.metrics
    portfolios["DQN Agent (ours)"] = r.portfolio_values
    print(f"  DQN   → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}")

    # ── 2. Buy & Hold ─────────────────────────────────────────────────────────
    buys, sells = buy_and_hold_signals(prices_test)
    r = engine.run(prices_test, buys, sells)
    results["Buy & Hold"]    = r.metrics
    portfolios["Buy & Hold"] = r.portfolio_values
    print(f"  B&H   → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}")

    # ── 3. SMA Crossover ─────────────────────────────────────────────────────
    buys, sells = sma_crossover_signals(prices_test)
    r = engine.run(prices_test, buys, sells)
    results["SMA Crossover (5/20)"]    = r.metrics
    portfolios["SMA Crossover (5/20)"] = r.portfolio_values
    print(f"  SMA   → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}")

    # ── 4. MACD + RSI Rules ───────────────────────────────────────────────────
    buys, sells = macd_rsi_signals(df_test)
    r = engine.run(prices_test, buys, sells)
    results["MACD + RSI Rules"]    = r.metrics
    portfolios["MACD + RSI Rules"] = r.portfolio_values
    print(f"  MACD  → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}")

    # ── 5. RSI Mean Reversion ─────────────────────────────────────────────────
    buys, sells = rsi_mean_reversion_signals(df_test)
    r = engine.run(prices_test, buys, sells)
    results["RSI Mean Reversion"]    = r.metrics
    portfolios["RSI Mean Reversion"] = r.portfolio_values
    print(f"  RSI   → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}")

    # ── 6. LSTM Baseline ──────────────────────────────────────────────────────
    print("Training LSTM Baseline...")
    lstm = LSTMAgent(window=20)
    info = lstm.train(df_train, epochs=lstm_epochs)
    lstm.save("models/lstm_baseline")

    buys, sells = lstm.backtest(df_test)
    r = engine.run(prices_test, buys, sells)
    results["LSTM Baseline"]    = r.metrics
    portfolios["LSTM Baseline"] = r.portfolio_values
    print(f"  LSTM  → Return: {r.metrics['total_return_pct']:+.2f}%  Sharpe: {r.metrics['sharpe_ratio']:.3f}  (val_acc: {info['val_accuracy']:.3f})")

    return results, portfolios, prices_test, dates_test, capital


# ── Plot: Equity curves ───────────────────────────────────────────────────────

def plot_equity_curves(
    portfolios: dict,
    prices: list,
    dates: list,
    capital: float,
    ticker: str,
    out_path: str = "results/equity_curves.png",
):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalise to 100 for cleaner comparison
    for name, pv in portfolios.items():
        norm = [v / capital * 100 for v in pv]
        x    = dates[:len(norm)]
        lw   = 2.5 if "ours" in name else 1.2
        ls   = "-"  if "ours" in name else ("--" if name == "Buy & Hold" else ":")
        ax.plot(x, norm, label=name, color=COLORS.get(name, "#333"),
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.axhline(100, color="#ccc", linewidth=0.8, linestyle="--")
    ax.set_title(f"Strategy Comparison — {ticker} (out-of-sample test period)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Portfolio value (indexed to 100)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Plot: Metrics bar chart ───────────────────────────────────────────────────

def plot_metrics_bar(
    results: dict,
    out_path: str = "results/metrics_bar.png",
):
    names   = list(results.keys())
    sharpes = [results[n]["sharpe_ratio"]      for n in names]
    returns = [results[n]["total_return_pct"]  for n in names]
    mdds    = [abs(results[n]["max_drawdown_pct"]) for n in names]

    x   = np.arange(len(names))
    w   = 0.26
    bar_colors = [COLORS.get(n, "#888") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model vs Baseline — Key Metrics (out-of-sample)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, values, title, fmt, higher_better in zip(
        axes,
        [returns, sharpes, mdds],
        ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%, lower = better)"],
        ["{:.1f}%", "{:.3f}", "{:.1f}%"],
        [True, True, False],
    ):
        bars = ax.bar(x, values, color=bar_colors, edgecolor="white",
                      linewidth=0.5, alpha=0.88)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) * 0.02),
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=8.5)

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" (ours)", "\n★") for n in names],
            fontsize=8, rotation=15, ha="right"
        )
        ax.axhline(0, color="#ccc", linewidth=0.7)
        ax.set_ylim(min(values) * 1.3 if min(values) < 0 else 0,
                    max(values) * 1.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Save CSV table ────────────────────────────────────────────────────────────

def save_table(results: dict, out_path: str = "results/comparison_table.csv"):
    rows = []
    for name, m in results.items():
        rows.append({
            "Strategy":     name,
            "Return (%)":   round(m["total_return_pct"], 2),
            "Sharpe Ratio": round(m["sharpe_ratio"], 4),
            "Max DD (%)":   round(m["max_drawdown_pct"], 2),
            "Win Rate (%)": round(m["win_rate_pct"], 1),
            "Trades":       m["total_trades"],
            "Calmar":       round(m["calmar_ratio"], 4),
        })
    df = pd.DataFrame(rows)
    # Rank by Sharpe ratio
    df = df.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="Rank")
    print(f"Saved → {out_path}")

    # Print to console
    print("\n" + "=" * 72)
    print(df.to_string())
    print("=" * 72)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run baseline comparison")
    p.add_argument("--ticker",      default="^TWII")
    p.add_argument("--train-start", default="2016-01-01")
    p.add_argument("--train-end",   default="2022-12-31")
    p.add_argument("--test-start",  default="2023-01-01")
    p.add_argument("--test-end",    default="2024-12-31")
    p.add_argument("--capital",     default=1_000_000, type=float)
    p.add_argument("--dqn-iters",   default=200, type=int)
    p.add_argument("--lstm-epochs", default=30,  type=int)
    return p.parse_args()


def main():
    args = parse_args()

    results, portfolios, prices, dates, capital = run_all(
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        capital=args.capital,
        dqn_iters=args.dqn_iters,
        lstm_epochs=args.lstm_epochs,
    )

    df_table = save_table(results)
    plot_equity_curves(portfolios, prices, dates, capital, args.ticker)
    plot_metrics_bar(results)

    # Print DQN rank
    dqn_rank = df_table[df_table["Strategy"].str.contains("ours")].index
    if len(dqn_rank):
        rank = dqn_rank[0]
        total = len(df_table)
        print(f"\nDQN Agent ranked #{rank} out of {total} strategies by Sharpe ratio.")

    print("\nNext step: commit results/ to repo, then update README with these images.")


if __name__ == "__main__":
    main()
