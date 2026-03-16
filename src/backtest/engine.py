"""
Backtest engine with position sizing, stop-loss, and performance metrics.

Original project had zero risk management — all-in, no stops, no sizing.
This module adds what's missing and computes standard quant metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker: str
    model_name: str
    initial_capital: float
    final_value: float
    portfolio_values: list[float]
    buy_dates: list
    sell_dates: list
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"[{self.ticker}] {self.model_name}\n"
            f"  Total return   : {m.get('total_return_pct', 0):+.2f}%\n"
            f"  Sharpe ratio   : {m.get('sharpe_ratio', 0):.3f}\n"
            f"  Max drawdown   : {m.get('max_drawdown_pct', 0):.2f}%\n"
            f"  Win rate       : {m.get('win_rate_pct', 0):.1f}%\n"
            f"  Total trades   : {m.get('total_trades', 0)}\n"
            f"  Calmar ratio   : {m.get('calmar_ratio', 0):.3f}\n"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulates trading with:
    - Configurable stop-loss per position
    - Maximum position size as % of capital
    - Performance metrics: Sharpe, Max Drawdown, Win Rate, Calmar
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        stop_loss_pct: float = 0.05,
        max_position_pct: float = 0.20,
        transaction_cost_pct: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost_pct

    def run(
        self,
        prices: list | np.ndarray,
        buy_indices: list[int],
        sell_indices: list[int],
        dates: list | pd.DatetimeIndex | None = None,
        ticker: str = "UNKNOWN",
        model_name: str = "agent",
    ) -> BacktestResult:
        """
        Simulate trades from signal lists.

        buy_indices / sell_indices: integer positions in `prices` list.
        """
        prices = list(prices)
        n = len(prices)
        capital = self.initial_capital
        portfolio_values: list[float] = []
        trades: list[dict] = []

        # Track open positions: list of (entry_price, units)
        positions: list[tuple[float, int]] = []
        holding = 0

        buy_set = set(buy_indices)
        sell_set = set(sell_indices)
        actual_buys: list[int] = []
        actual_sells: list[int] = []

        for t, price in enumerate(prices):
            # ── Stop-loss check on open positions ───────────────────────────
            remaining = []
            for entry_price, units in positions:
                if price < entry_price * (1 - self.stop_loss_pct):
                    proceeds = units * price * (1 - self.transaction_cost)
                    capital += proceeds
                    holding -= units
                    pnl = proceeds - units * entry_price
                    trades.append({
                        "type": "stop_loss",
                        "t": t,
                        "price": price,
                        "units": units,
                        "entry_price": entry_price,
                        "pnl": pnl,
                    })
                    actual_sells.append(t)
                else:
                    remaining.append((entry_price, units))
            positions = remaining

            # ── Buy signal ──────────────────────────────────────────────────
            if t in buy_set:
                max_spend = capital * self.max_position_pct
                units = max(1, int(max_spend / price))
                cost = units * price * (1 + self.transaction_cost)
                if capital >= cost:
                    capital -= cost
                    holding += units
                    positions.append((price, units))
                    actual_buys.append(t)
                    trades.append({
                        "type": "buy",
                        "t": t,
                        "price": price,
                        "units": units,
                        "entry_price": price,
                        "pnl": 0,
                    })

            # ── Sell signal ─────────────────────────────────────────────────
            elif t in sell_set and holding > 0:
                for entry_price, units in positions:
                    proceeds = units * price * (1 - self.transaction_cost)
                    capital += proceeds
                    holding -= units
                    pnl = proceeds - units * entry_price
                    trades.append({
                        "type": "sell",
                        "t": t,
                        "price": price,
                        "units": units,
                        "entry_price": entry_price,
                        "pnl": pnl,
                    })
                    actual_sells.append(t)
                positions = []

            portfolio_values.append(capital + holding * price)

        # Close remaining positions at last price
        if holding > 0 and prices:
            last_price = prices[-1]
            for entry_price, units in positions:
                proceeds = units * last_price * (1 - self.transaction_cost)
                capital += proceeds
                pnl = proceeds - units * entry_price
                trades.append({
                    "type": "close",
                    "t": n - 1,
                    "price": last_price,
                    "units": units,
                    "entry_price": entry_price,
                    "pnl": pnl,
                })

        final_value = capital
        metrics = self.compute_metrics(portfolio_values, trades)

        result = BacktestResult(
            ticker=ticker,
            model_name=model_name,
            initial_capital=self.initial_capital,
            final_value=final_value,
            portfolio_values=portfolio_values,
            buy_dates=actual_buys,
            sell_dates=actual_sells,
            trades=trades,
            metrics=metrics,
        )
        return result

    # ── Metrics ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        portfolio_values: list[float],
        trades: list[dict] | None = None,
    ) -> dict:
        if len(portfolio_values) < 2:
            return {}

        pv = np.array(portfolio_values)
        returns = pd.Series(pv).pct_change().dropna()

        # Total return
        total_return = (pv[-1] / pv[0] - 1) * 100

        # Sharpe ratio (annualised, assuming daily returns, 252 trading days)
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else 0.0
        )

        # Max drawdown
        roll_max = pd.Series(pv).cummax()
        drawdown = (pd.Series(pv) / roll_max - 1) * 100
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = (
            (returns.mean() * 252) / abs(max_drawdown / 100)
            if max_drawdown != 0
            else 0.0
        )

        # Win rate from completed trades
        if trades:
            closed = [t for t in trades if t["type"] in ("sell", "stop_loss", "close")]
            win_rate = (
                sum(1 for t in closed if t["pnl"] > 0) / len(closed) * 100
                if closed
                else 0.0
            )
            total_trades = len(closed)
            avg_pnl = np.mean([t["pnl"] for t in closed]) if closed else 0.0
        else:
            win_rate = 0.0
            total_trades = 0
            avg_pnl = 0.0

        return {
            "total_return_pct": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_drawdown, 4),
            "calmar_ratio": round(calmar, 4),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": total_trades,
            "avg_pnl_per_trade": round(avg_pnl, 2),
        }

    # ── Benchmark comparison ─────────────────────────────────────────────────

    @staticmethod
    def buy_and_hold_return(prices: list | np.ndarray) -> float:
        """Simple buy-and-hold return % for comparison baseline."""
        prices = list(prices)
        if len(prices) < 2:
            return 0.0
        return (prices[-1] / prices[0] - 1) * 100
