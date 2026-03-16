"""
Model evaluation script.

Loads a trained model, runs backtest, prints full performance report,
and saves a Plotly HTML chart you can drop into a README or portfolio.

Usage:
    python scripts/evaluate.py --ticker ^TWII --model-path models/dqn_TWII.keras
    python scripts/evaluate.py --ticker 2330.TW --model-path models/dqn_2330TW.keras --html-out results/
    python scripts/evaluate.py --compare models/dqn_TWII.keras models/cnn_TWII.keras --ticker ^TWII
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.downloader import download, get_train_test_split
from src.features.indicators import add_all, to_macd_series
from src.models.dqn_agent import DQNAgent
from src.backtest.engine import BacktestEngine


# ── Single model evaluation ───────────────────────────────────────────────────

def evaluate(
    ticker: str,
    model_path: str,
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
    initial_capital: float = 1_000_000,
    stop_loss: float = 0.05,
    max_position: float = 0.20,
    use_macd: bool = True,
    html_out: str | None = None,
) -> dict:

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {ticker} ({test_start} → {test_end})...")
    df = download(ticker, start=test_start, end=test_end)
    df = add_all(df)
    prices = df["Close"].tolist()
    dates  = df.index.tolist()
    macd   = to_macd_series(prices) if use_macd else None

    # ── Model ─────────────────────────────────────────────────────────────────
    agent = DQNAgent()
    agent.load(model_path)

    # ── Backtest ──────────────────────────────────────────────────────────────
    buys, sells, portfolio = agent.backtest(prices, macd=macd)
    engine = BacktestEngine(
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss,
        max_position_pct=max_position,
    )
    result = engine.run(
        prices, buys, sells,
        ticker=ticker,
        model_name=Path(model_path).stem,
    )
    bnh = engine.buy_and_hold_return(prices)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "─" * 52)
    print(result.summary())
    print(f"  Buy-and-hold   : {bnh:+.2f}%")
    print(f"  Alpha          : {result.metrics['total_return_pct'] - bnh:+.2f}%")
    print("─" * 52)

    # ── Plotly chart ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Price & signals", "Equity curve vs buy-and-hold"),
        vertical_spacing=0.08,
    )

    # Price + signals
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode="lines", name="Price",
        line=dict(color="#888780", width=1.5),
    ), row=1, col=1)

    if result.buy_dates:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in result.buy_dates],
            y=[prices[i] for i in result.buy_dates],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="#1D9E75"),
        ), row=1, col=1)

    if result.sell_dates:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in result.sell_dates],
            y=[prices[i] for i in result.sell_dates],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="#D85A30"),
        ), row=1, col=1)

    # Equity curves
    bnh_curve = [initial_capital * (p / prices[0]) for p in prices]
    fig.add_trace(go.Scatter(
        x=dates[:len(result.portfolio_values)],
        y=result.portfolio_values,
        mode="lines", name="DQN portfolio",
        line=dict(color="#534AB7", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bnh_curve,
        mode="lines", name="Buy & hold",
        line=dict(color="#888780", width=1, dash="dash"),
    ), row=2, col=1)

    m = result.metrics
    fig.update_layout(
        title=(
            f"{ticker} — DQN backtest  |  "
            f"Return: {m['total_return_pct']:+.2f}%  |  "
            f"Sharpe: {m['sharpe_ratio']:.3f}  |  "
            f"MDD: {m['max_drawdown_pct']:.2f}%  |  "
            f"Win rate: {m['win_rate_pct']:.1f}%"
        ),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    if html_out:
        out_dir = Path(html_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker.replace('^','').replace('.','_')}_backtest.html"
        fig.write_html(str(out_path))
        print(f"\nChart saved → {out_path}")
    else:
        fig.show()

    return {
        "ticker": ticker,
        "model_path": model_path,
        "metrics": result.metrics,
        "buy_and_hold_pct": bnh,
        "alpha": result.metrics["total_return_pct"] - bnh,
    }


# ── Multi-model comparison ────────────────────────────────────────────────────

def compare(
    ticker: str,
    model_paths: list[str],
    test_start: str = "2023-01-01",
    test_end: str = "2024-12-31",
    initial_capital: float = 1_000_000,
    html_out: str | None = None,
):
    print(f"\nComparing {len(model_paths)} models on {ticker}...")

    df = download(ticker, start=test_start, end=test_end)
    df = add_all(df)
    prices = df["Close"].tolist()
    dates  = df.index.tolist()

    fig = go.Figure()
    bnh_curve = [initial_capital * (p / prices[0]) for p in prices]
    fig.add_trace(go.Scatter(
        x=dates, y=bnh_curve,
        mode="lines", name="Buy & hold",
        line=dict(color="#888780", width=1, dash="dash"),
    ))

    results_table = []
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#993556"]

    for i, mp in enumerate(model_paths):
        macd = to_macd_series(prices)
        agent = DQNAgent()
        agent.load(mp)
        buys, sells, portfolio = agent.backtest(prices, macd=macd)

        engine = BacktestEngine(initial_capital=initial_capital)
        result = engine.run(prices, buys, sells, ticker=ticker, model_name=Path(mp).stem)
        bnh    = engine.buy_and_hold_return(prices)
        m      = result.metrics

        label = f"{Path(mp).stem} ({m['total_return_pct']:+.1f}%)"
        fig.add_trace(go.Scatter(
            x=dates[:len(result.portfolio_values)],
            y=result.portfolio_values,
            mode="lines", name=label,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

        results_table.append({
            "Model":          Path(mp).stem,
            "Return %":       f"{m['total_return_pct']:+.2f}",
            "Sharpe":         f"{m['sharpe_ratio']:.3f}",
            "Max DD %":       f"{m['max_drawdown_pct']:.2f}",
            "Win rate %":     f"{m['win_rate_pct']:.1f}",
            "Trades":         m["total_trades"],
            "Alpha vs B&H":   f"{m['total_return_pct'] - bnh:+.2f}",
        })

    # Summary table
    df_table = pd.DataFrame(results_table)
    print("\n" + df_table.to_string(index=False))

    fig.update_layout(
        title=f"{ticker} — model comparison (equity curves)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        yaxis_title="Portfolio value",
    )

    if html_out:
        out_dir = Path(html_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker.replace('^','').replace('.','_')}_comparison.html"
        fig.write_html(str(out_path))
        print(f"Chart saved → {out_path}")
    else:
        fig.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",      default="^TWII")
    p.add_argument("--model-path",  default=None,           help="Single model .keras path")
    p.add_argument("--compare",     nargs="+", default=None, help="Multiple model paths to compare")
    p.add_argument("--test-start",  default="2023-01-01")
    p.add_argument("--test-end",    default="2024-12-31")
    p.add_argument("--capital",     default=1_000_000, type=float)
    p.add_argument("--stop-loss",   default=0.05, type=float)
    p.add_argument("--max-position",default=0.20, type=float)
    p.add_argument("--no-macd",     action="store_true")
    p.add_argument("--html-out",    default=None, help="Directory to save HTML charts")
    p.add_argument("--json-out",    default=None, help="Path to save metrics JSON")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        compare(
            ticker=args.ticker,
            model_paths=args.compare,
            test_start=args.test_start,
            test_end=args.test_end,
            initial_capital=args.capital,
            html_out=args.html_out,
        )
    elif args.model_path:
        result = evaluate(
            ticker=args.ticker,
            model_path=args.model_path,
            test_start=args.test_start,
            test_end=args.test_end,
            initial_capital=args.capital,
            stop_loss=args.stop_loss,
            max_position=args.max_position,
            use_macd=not args.no_macd,
            html_out=args.html_out,
        )
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Metrics saved → {args.json_out}")
    else:
        print("Provide --model-path or --compare. Use --help for details.")


if __name__ == "__main__":
    main()
