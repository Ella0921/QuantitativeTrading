"""
Streamlit Web UI — replaces the original tkinter main.py.

Tabs:
  1. Signal — CNN live signal prediction (replaces "信號燈")
  2. Train   — DQN training (replaces "train and test" train half)
  3. Backtest — DQN backtest with Plotly chart + metrics (replaces test half)
  4. Compare  — side-by-side model comparison (new)

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.downloader import download, get_train_test_split
from src.features.indicators import to_macd_series, add_all
from src.models.dqn_agent import DQNAgent
from src.models.cnn_agent import CNNAgent, prepare_inference_matrix
from src.backtest.engine import BacktestEngine

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantitative Trading System",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quantitative Trading System")
st.caption("DQN Agent + CNN Q-Network | Rebuilt with TF2 · Streamlit · Plotly")

# ── Session state init ────────────────────────────────────────────────────────

for key in ["dqn_agent", "cnn_agent", "backtest_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_signal, tab_train, tab_backtest, tab_compare = st.tabs(
    ["🔴 Signal", "🏋 Train", "📊 Backtest", "⚖ Compare"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signal (CNN)
# ─────────────────────────────────────────────────────────────────────────────

with tab_signal:
    st.subheader("CNN Signal — Long / Neutral / Short")
    st.info(
        "Uses the CNN Q-Network to predict a trading signal for today "
        "based on the last 32 days of price + volume data.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        ticker_sig = st.text_input("Ticker", value="^TWII", key="sig_ticker")
        model_path_sig = st.text_input(
            "CNN model path",
            value="models/cnn_agent",
            key="sig_model_path",
        )
        predict_btn = st.button("Get Signal", type="primary")

    with col2:
        if predict_btn:
            with st.spinner("Downloading data and computing signal..."):
                try:
                    matrix = prepare_inference_matrix(ticker_sig)
                    agent = CNNAgent()
                    mp = Path(model_path_sig)
                    if mp.exists():
                        agent.load(model_path_sig)
                    else:
                        st.warning("No saved model found — using untrained weights (for demo).")

                    result = agent.predict_signal(matrix)
                    signal = result["signal"]
                    q_vals = result["q_values"]

                    color = {"Long": "🟥", "Neutral": "⬜", "Short": "🟩"}.get(signal, "⬜")
                    st.markdown(f"## {color} {signal}")
                    st.write("**Q-values:**")
                    qdf = pd.DataFrame(
                        [{"Action": k, "Q-value": round(v, 4)} for k, v in q_vals.items()]
                    )
                    st.dataframe(qdf, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train (DQN)
# ─────────────────────────────────────────────────────────────────────────────

with tab_train:
    st.subheader("Train DQN Agent")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker_tr = st.text_input("Ticker", value="^TWII", key="tr_ticker")
        train_start = st.text_input("Train start", value="2016-01-01")
        train_end = st.text_input("Train end", value="2022-12-31")
    with c2:
        iterations = st.slider("Iterations", 50, 500, 200, 50)
        initial_money = st.number_input("Initial capital", value=1_000_000, step=100_000)
        use_macd = st.checkbox("Use MACD features", value=True)
    with c3:
        save_path = st.text_input("Save model to", value="models/dqn_agent")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            value=1e-5,
        )

    train_btn = st.button("Start Training", type="primary")

    if train_btn:
        with st.spinner("Downloading data..."):
            df_full = download(ticker_tr, start=train_start, end=train_end)
            df_full = add_all(df_full)
            prices = df_full["Close"].tolist()
            macd = to_macd_series(prices) if use_macd else None

        progress_bar = st.progress(0, text="Training...")
        status = st.empty()

        agent = DQNAgent(learning_rate=learning_rate)

        # Monkey-patch train to update progress (simple version)
        checkpoint = max(1, iterations // 20)
        results = agent.train(
            prices,
            macd=macd,
            iterations=iterations,
            initial_money=initial_money,
            checkpoint=checkpoint,
        )
        progress_bar.progress(100, text="Done!")
        st.session_state["dqn_agent"] = agent

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(save_path)

        st.success(f"Training complete. Model saved to `{save_path}`")

        # Loss chart
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(y=results["losses"], mode="lines", name="Loss", line=dict(color="#534AB7"))
        )
        fig_loss.update_layout(
            title="Training loss", xaxis_title="Epoch", yaxis_title="MSE loss",
            height=300, margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_loss, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Backtest
# ─────────────────────────────────────────────────────────────────────────────

with tab_backtest:
    st.subheader("Backtest — DQN Agent")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker_bt = st.text_input("Ticker", value="^TWII", key="bt_ticker")
        test_start = st.text_input("Test start", value="2023-01-01")
        test_end = st.text_input("Test end", value="2024-12-31")
    with c2:
        bt_model_path = st.text_input("Load model from", value="models/dqn_agent", key="bt_model")
        use_macd_bt = st.checkbox("Use MACD features", value=True, key="bt_macd")
        bt_capital = st.number_input("Initial capital", value=1_000_000, step=100_000, key="bt_cap")
    with c3:
        stop_loss = st.slider("Stop-loss %", 1, 20, 5) / 100
        max_pos = st.slider("Max position %", 5, 50, 20) / 100

    run_bt = st.button("Run Backtest", type="primary")

    if run_bt:
        with st.spinner("Running backtest..."):
            try:
                # Load data
                df_test = download(ticker_bt, start=test_start, end=test_end)
                df_test = add_all(df_test)
                prices = df_test["Close"].tolist()
                macd = to_macd_series(prices) if use_macd_bt else None

                # Load or use session agent
                if st.session_state["dqn_agent"] is not None and not Path(bt_model_path).exists():
                    agent = st.session_state["dqn_agent"]
                else:
                    agent = DQNAgent()
                    if Path(bt_model_path).exists():
                        agent.load(bt_model_path)

                # Get signals
                buys, sells, portfolio = agent.backtest(prices, macd=macd)

                # Run engine with risk management
                engine = BacktestEngine(
                    initial_capital=bt_capital,
                    stop_loss_pct=stop_loss,
                    max_position_pct=max_pos,
                )
                result = engine.run(
                    prices, buys, sells,
                    ticker=ticker_bt, model_name="DQN Agent"
                )
                st.session_state["backtest_result"] = result

                bnh = engine.buy_and_hold_return(prices)

                # ── Metrics cards ──────────────────────────────────────────
                m = result.metrics
                st.markdown("### Performance metrics")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Total return", f"{m['total_return_pct']:+.2f}%",
                           delta=f"vs B&H {bnh:+.1f}%")
                mc2.metric("Sharpe ratio", f"{m['sharpe_ratio']:.3f}")
                mc3.metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
                mc4.metric("Win rate", f"{m['win_rate_pct']:.1f}%")
                mc5.metric("Total trades", m['total_trades'])

                # ── Plotly chart ───────────────────────────────────────────
                fig = go.Figure()

                # Price line
                fig.add_trace(go.Scatter(
                    y=prices, mode="lines", name="Price",
                    line=dict(color="#888780", width=1.5),
                ))

                # Buy markers
                if result.buy_dates:
                    fig.add_trace(go.Scatter(
                        x=result.buy_dates,
                        y=[prices[i] for i in result.buy_dates],
                        mode="markers",
                        name="Buy",
                        marker=dict(symbol="triangle-up", size=10, color="#1D9E75"),
                    ))

                # Sell markers
                if result.sell_dates:
                    fig.add_trace(go.Scatter(
                        x=result.sell_dates,
                        y=[prices[i] for i in result.sell_dates],
                        mode="markers",
                        name="Sell",
                        marker=dict(symbol="triangle-down", size=10, color="#D85A30"),
                    ))

                fig.update_layout(
                    title=f"{ticker_bt} — DQN backtest signals",
                    xaxis_title="Trading day",
                    yaxis_title="Price",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Portfolio equity curve
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=result.portfolio_values, mode="lines", name="DQN portfolio",
                    line=dict(color="#534AB7", width=2),
                ))
                bnh_curve = [bt_capital * (p / prices[0]) for p in prices]
                fig2.add_trace(go.Scatter(
                    y=bnh_curve, mode="lines", name="Buy & hold",
                    line=dict(color="#888780", width=1, dash="dash"),
                ))
                fig2.update_layout(
                    title="Equity curve vs buy-and-hold",
                    xaxis_title="Trading day",
                    yaxis_title="Portfolio value",
                    height=350,
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Compare
# ─────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Model comparison")
    st.info(
        "Run backtests on multiple tickers or models to compare side-by-side. "
        "Run a backtest in the Backtest tab first, then add more here.",
        icon="ℹ️",
    )

    if st.session_state["backtest_result"] is None:
        st.warning("No backtest result yet — run a backtest in the Backtest tab.")
    else:
        res = st.session_state["backtest_result"]
        st.markdown(f"```\n{res.summary()}\n```")

        # Show trade log
        if res.trades:
            df_trades = pd.DataFrame(res.trades)
            df_trades = df_trades[["type", "t", "price", "units", "entry_price", "pnl"]]
            df_trades.columns = ["Type", "Day", "Price", "Units", "Entry price", "P&L"]
            df_trades["P&L"] = df_trades["P&L"].round(2)

            st.markdown("### Trade log")
            st.dataframe(
                df_trades.style.applymap(
                    lambda v: "color: #1D9E75" if isinstance(v, (int, float)) and v > 0
                    else ("color: #D85A30" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["P&L"],
                ),
                use_container_width=True,
                height=300,
            )
