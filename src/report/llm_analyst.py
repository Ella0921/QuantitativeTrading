"""
LLM analyst — calls Claude API to generate a natural language
trading analysis report from structured quantitative data.

Designed to be called from the Streamlit app after a backtest run.
The prompt bundles all relevant numbers so the LLM never needs to
fetch data itself — it only interprets what we hand it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import pandas as pd


# ── Input dataclass ───────────────────────────────────────────────────────────

@dataclass
class AnalystInput:
    ticker: str
    signal: str                    # "Long" | "Neutral" | "Short"
    signal_q: dict                 # {"Long": 0.8, "Neutral": 0.1, "Short": 0.1}
    current_price: float
    price_chg_1d: float
    price_chg_5d: float
    price_chg_30d: float

    macd_hist: float
    macd_line: float
    macd_signal_line: float
    rsi: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    atr: float

    backtest_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int

    benchmark_ticker: str
    benchmark_return_30d: float
    asset_return_30d: float
    date_str: str


def from_app_data(
    ticker: str,
    df: pd.DataFrame,
    signal_result: dict,
    backtest_metrics: dict,
    benchmark_ticker: str = "^TWII",
    benchmark_df: pd.DataFrame | None = None,
) -> AnalystInput:
    """
    Convenience constructor — builds AnalystInput from the data
    already available in the Streamlit app after a backtest run.
    """
    latest = df.iloc[-1]
    prices = df["Close"]

    price_chg_1d  = prices.pct_change(1).iloc[-1]  * 100
    price_chg_5d  = prices.pct_change(5).iloc[-1]  * 100
    price_chg_30d = prices.pct_change(30).iloc[-1] * 100

    asset_30d = price_chg_30d
    bench_30d = 0.0
    if benchmark_df is not None and len(benchmark_df) > 30:
        bp = benchmark_df["Close"]
        bench_30d = bp.pct_change(30).iloc[-1] * 100

    return AnalystInput(
        ticker=ticker,
        signal=signal_result.get("signal", "Neutral"),
        signal_q=signal_result.get("q_values", {}),
        current_price=float(latest["Close"]),
        price_chg_1d=float(price_chg_1d),
        price_chg_5d=float(price_chg_5d),
        price_chg_30d=float(price_chg_30d),
        macd_hist=float(latest.get("MACD_hist", 0)),
        macd_line=float(latest.get("MACD_line", 0)),
        macd_signal_line=float(latest.get("MACD_signal", 0)),
        rsi=float(latest.get("RSI", 50)),
        bb_upper=float(latest.get("BB_upper", 0)),
        bb_mid=float(latest.get("BB_mid", 0)),
        bb_lower=float(latest.get("BB_lower", 0)),
        atr=float(latest.get("ATR", 0)),
        backtest_return_pct=backtest_metrics.get("total_return_pct", 0),
        sharpe_ratio=backtest_metrics.get("sharpe_ratio", 0),
        max_drawdown_pct=backtest_metrics.get("max_drawdown_pct", 0),
        win_rate_pct=backtest_metrics.get("win_rate_pct", 0),
        total_trades=backtest_metrics.get("total_trades", 0),
        benchmark_ticker=benchmark_ticker,
        benchmark_return_30d=float(bench_30d),
        asset_return_30d=float(asset_30d),
        date_str=str(df.index[-1].date()),
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(inp: AnalystInput) -> str:
    bb_pos = (
        "above upper band (potentially overbought)"
        if inp.current_price > inp.bb_upper else
        "below lower band (potentially oversold)"
        if inp.current_price < inp.bb_lower else
        "within normal range"
    )
    macd_trend = "bullish" if inp.macd_hist > 0 else "bearish"
    rsi_desc   = (
        f"overbought at {inp.rsi:.1f}" if inp.rsi > 70 else
        f"oversold at {inp.rsi:.1f}"   if inp.rsi < 30 else
        f"neutral at {inp.rsi:.1f}"
    )
    alpha = inp.asset_return_30d - inp.benchmark_return_30d

    return f"""You are a quantitative analyst. Based on the data below, write a concise trading analysis report in Traditional Chinese (繁體中文).

=== DATA ({inp.date_str}) ===
Ticker: {inp.ticker} | Price: {inp.current_price:.2f}
Returns: 1D {inp.price_chg_1d:+.2f}% | 5D {inp.price_chg_5d:+.2f}% | 30D {inp.price_chg_30d:+.2f}%

Model signal: {inp.signal}
Confidence — Long: {inp.signal_q.get('Long',0):.3f} | Neutral: {inp.signal_q.get('Neutral',0):.3f} | Short: {inp.signal_q.get('Short',0):.3f}

MACD: {macd_trend} (hist {inp.macd_hist:+.4f}, line {inp.macd_line:.4f} vs signal {inp.macd_signal_line:.4f})
RSI(14): {rsi_desc}
Bollinger Bands: price is {bb_pos}
ATR(14): {inp.atr:.2f}

Backtest (out-of-sample): return {inp.backtest_return_pct:+.2f}% | Sharpe {inp.sharpe_ratio:.3f} | MaxDD {inp.max_drawdown_pct:.2f}% | Win rate {inp.win_rate_pct:.1f}% | {inp.total_trades} trades

vs {inp.benchmark_ticker} (30D): {inp.ticker} {inp.asset_return_30d:+.2f}% vs benchmark {inp.benchmark_return_30d:+.2f}% | Alpha: {alpha:+.2f}%

=== REPORT FORMAT ===
Write exactly four sections with these headers:

【今日訊號】
State the model signal and explain the 2-3 most important technical reasons. Reference specific numbers.

【技術指標分析】
Interpret MACD, RSI, and Bollinger Bands together. Identify the dominant market regime.

【績效評估】
Summarise backtest performance. Compare Sharpe ratio and return to a typical benchmark expectation (Sharpe > 1 = good). Comment on risk (max drawdown, win rate).

【市場比較與建議】
Compare 30-day performance vs benchmark. State whether the model is adding alpha. Give a 1-sentence actionable suggestion based on the combined signal and technical picture.

Keep the entire report under 400 Chinese characters. Use precise numbers. Do not add disclaimers or caveats beyond the report sections."""


# ── API call ──────────────────────────────────────────────────────────────────

async def generate_report_async(inp: AnalystInput) -> str:
    """
    Async version for use in Streamlit with st.spinner.
    Calls the Anthropic API via the anthropic Python SDK.
    Returns the report text, or an error message string.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": build_prompt(inp)}],
        )
        return message.content[0].text
    except ImportError:
        return _fallback_report(inp)
    except Exception as e:
        return f"⚠️ LLM report unavailable: {e}\n\n{_fallback_report(inp)}"


def generate_report_sync(inp: AnalystInput) -> str:
    """Synchronous version — simpler to call from non-async contexts."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": build_prompt(inp)}],
        )
        return message.content[0].text
    except ImportError:
        return _fallback_report(inp)
    except Exception as e:
        return f"⚠️ LLM report unavailable: {e}\n\n{_fallback_report(inp)}"


def _fallback_report(inp: AnalystInput) -> str:
    """
    Rule-based fallback report when Claude API is unavailable.
    Uses the same data to generate a template-based report.
    """
    macd_str = "MACD 柱狀圖轉正，動能偏多" if inp.macd_hist > 0 else "MACD 柱狀圖為負，動能偏空"
    rsi_str  = (
        f"RSI({inp.rsi:.1f}) 進入超買區，留意回調風險" if inp.rsi > 70 else
        f"RSI({inp.rsi:.1f}) 進入超賣區，存在反彈機會" if inp.rsi < 30 else
        f"RSI({inp.rsi:.1f}) 處於中性區間"
    )
    bb_str = (
        "價格突破布林上軌，短期偏強" if inp.current_price > inp.bb_upper else
        "價格跌破布林下軌，短期偏弱" if inp.current_price < inp.bb_lower else
        "價格在布林通道內正常波動"
    )
    alpha = inp.asset_return_30d - inp.benchmark_return_30d

    signal_zh = {"Long": "做多", "Neutral": "觀望", "Short": "做空"}.get(inp.signal, inp.signal)

    return f"""【今日訊號】
模型輸出 **{signal_zh}** 訊號（置信度 Long: {inp.signal_q.get('Long',0):.3f}）。{macd_str}，{rsi_str}。

【技術指標分析】
{macd_str}。{rsi_str}。{bb_str}，ATR 為 {inp.atr:.2f}，顯示目前日均波動幅度約 {inp.atr/inp.current_price*100:.1f}%。

【績效評估】
回測期間總報酬 {inp.backtest_return_pct:+.2f}%，Sharpe ratio {inp.sharpe_ratio:.3f}（>1 為良好水準），最大回撤 {inp.max_drawdown_pct:.2f}%，勝率 {inp.win_rate_pct:.1f}%，共執行 {inp.total_trades} 筆交易。

【市場比較與建議】
近 30 日 {inp.ticker} 報酬 {inp.asset_return_30d:+.2f}%，相較 {inp.benchmark_ticker} 的 {inp.benchmark_return_30d:+.2f}%，超額報酬（Alpha）為 {alpha:+.2f}%。綜合訊號與指標，建議{"增加持倉" if inp.signal == "Long" else "減少持倉" if inp.signal == "Short" else "暫時觀望"}。"""
