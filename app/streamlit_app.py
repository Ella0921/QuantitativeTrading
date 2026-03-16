"""
Streamlit Web UI — v2 with improved UX.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from src.data.downloader import download
from src.features.indicators import to_macd_series, add_all
from src.models.dqn_agent import DQNAgent
from src.models.cnn_agent import CNNAgent, prepare_inference_matrix
from src.backtest.engine import BacktestEngine

st.set_page_config(page_title="Quantitative Trading System", page_icon="📈", layout="wide")

for key, default in [
    ("dqn_agent", None), ("last_model_path", "models/dqn_agent"),
    ("backtest_result", None), ("backtest_results_list", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📈 Quantitative Trading System")
st.caption("DQN Agent + CNN Q-Network · TF2 · Streamlit · Plotly")
st.divider()

tab_signal, tab_train, tab_backtest, tab_compare = st.tabs(
    ["🔴 Signal", "🏋️ Train", "📊 Backtest", "⚖️ Compare"]
)

# ─── TAB 1: Signal ────────────────────────────────────────────────────────────
with tab_signal:
    st.subheader("Today's Signal")
    st.caption("CNN Q-Network predicts Long / Neutral / Short based on the last 32 days of price + volume data.")

    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        with st.container(border=True):
            st.markdown("**Stock settings**")
            ticker_sig = st.text_input("Ticker symbol", value="^TWII",
                help="Yahoo Finance ticker. E.g. ^TWII (Taiwan), 2330.TW, AAPL", key="sig_ticker")
            model_path_sig = st.text_input("CNN model path", value="models/cnn_agent",
                help="Path to a saved CNN model. Leave as default for demo with untrained weights.")
            predict_btn = st.button("Get Signal", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            with st.spinner(f"Fetching {ticker_sig} data..."):
                try:
                    matrix = prepare_inference_matrix(ticker_sig)
                    agent_cnn = CNNAgent()
                    mp = Path(model_path_sig + ".keras")
                    if mp.exists():
                        agent_cnn.load(model_path_sig)
                    else:
                        st.caption("⚠️ No saved model — using untrained weights (demo).")
                    result = agent_cnn.predict_signal(matrix)
                    signal = result["signal"]
                    q_vals = result["q_values"]
                    cfg = {"Long": ("#1D9E75", "🟢 建議買入 / Go long"),
                           "Neutral": ("#888780", "⚪ 觀望 / Hold"),
                           "Short": ("#D85A30", "🔴 建議賣出 / Go short")}
                    color, label = cfg[signal]
                    with st.container(border=True):
                        st.markdown(f"<h2 style='color:{color};margin:0'>{label}</h2>",
                                    unsafe_allow_html=True)
                        st.divider()
                        st.markdown("**Model confidence (Q-values)**")
                        for action, qv in q_vals.items():
                            st.progress(min(max((qv + 1) / 2, 0.0), 1.0),
                                        text=f"{action}: {qv:.4f}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            with st.container(border=True):
                st.markdown("<p style='color:gray;text-align:center;padding:2rem 0'>"
                            "Enter a ticker and click Get Signal</p>", unsafe_allow_html=True)

# ─── TAB 2: Train ─────────────────────────────────────────────────────────────
with tab_train:
    st.subheader("Train DQN Agent")
    st.caption("Train a new model on historical data. The model is auto-loaded in the Backtest tab after training.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("**📅 Data settings**")
            ticker_tr = st.text_input("Ticker", value="^TWII", key="tr_ticker",
                                      help="Yahoo Finance ticker symbol")
            tr_c1, tr_c2 = st.columns(2)
            with tr_c1:
                train_start = st.date_input("Train start", value=date(2016, 1, 1))
            with tr_c2:
                train_end = st.date_input("Train end", value=date(2022, 12, 31))
            use_macd = st.toggle("Use MACD as state features", value=True,
                help="Uses MACD histogram differences as the agent's state vector. "
                     "Generally improves performance vs raw price differences.")

        with st.container(border=True):
            st.markdown("**💾 Model output**")
            save_path = st.text_input("Save model to", value="models/dqn_agent",
                help="Path without extension — a .keras file will be created.")

    with col_right:
        with st.container(border=True):
            st.markdown("**⚙️ Hyperparameters**")
            iterations = st.slider("Training iterations", 50, 500, 200, 50,
                help="Number of full passes over training data. More = slower but better.")
            initial_money = st.number_input("Simulated starting capital",
                value=1_000_000, step=100_000, min_value=100_000,
                help="Virtual capital used during training to compute rewards.")
            lr_labels = {"1e-6 (very slow)": 1e-6, "5e-6": 5e-6,
                         "1e-5 (default)": 1e-5, "5e-5": 5e-5, "1e-4 (fast)": 1e-4}
            lr_choice = st.select_slider("Learning rate", options=list(lr_labels.keys()),
                value="1e-5 (default)",
                help="Controls how fast weights update. Too high = unstable. Too low = slow.")
            learning_rate = lr_labels[lr_choice]

        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)

    if train_btn:
        prog = st.progress(0, text="Downloading data...")
        try:
            df_full = download(ticker_tr, start=str(train_start), end=str(train_end))
            df_full = add_all(df_full)
            prices  = df_full["Close"].tolist()
            macd    = to_macd_series(prices) if use_macd else None
            prog.progress(10, text=f"Downloaded {len(prices)} trading days — training...")

            agent = DQNAgent(learning_rate=learning_rate)
            results = agent.train(prices, macd=macd, iterations=iterations,
                                  initial_money=initial_money,
                                  checkpoint=max(1, iterations // 10))
            prog.progress(95, text="Saving model...")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(save_path)
            st.session_state["dqn_agent"] = agent
            st.session_state["last_model_path"] = save_path
            prog.progress(100, text="Done!")

            st.success(f"✅ Training complete — {iterations} epochs | "
                       f"Final return: {results['final_return_pct']:+.2f}% | "
                       f"Saved to `{save_path}.keras`")
            st.info("💡 Head to the **Backtest** tab to evaluate your model.", icon="👉")

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=results["losses"], mode="lines",
                name="Training loss", line=dict(color="#534AB7", width=2),
                fill="tozeroy", fillcolor="rgba(83,74,183,0.08)"))
            fig_loss.update_layout(title="Training loss per epoch",
                xaxis_title="Epoch", yaxis_title="MSE loss",
                height=260, margin=dict(t=40, b=30, l=40, r=20))
            st.plotly_chart(fig_loss, use_container_width=True)

        except Exception as e:
            prog.empty()
            st.error(f"Training failed: {e}")
            st.exception(e)

# ─── TAB 3: Backtest ──────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest")
    st.caption("Evaluate a trained model on out-of-sample data with realistic risk management.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("**📅 Test period**")
            ticker_bt = st.text_input("Ticker", value="^TWII", key="bt_ticker",
                                      help="Should match the ticker used during training.")
            bt_c1, bt_c2 = st.columns(2)
            with bt_c1:
                test_start = st.date_input("Start date", value=date(2023, 1, 1), key="bt_start")
            with bt_c2:
                test_end = st.date_input("End date", value=date(2024, 12, 31), key="bt_end")

        with st.container(border=True):
            st.markdown("**🤖 Model**")
            bt_model_path = st.text_input("Model path",
                value=st.session_state["last_model_path"], key="bt_model",
                help="Auto-filled after training. Points to the .keras file.")
            use_macd_bt = st.toggle("Use MACD features", value=True, key="bt_macd",
                                    help="Must match the setting used during training.")
            bt_capital = st.number_input("Starting capital", value=1_000_000,
                                         step=100_000, key="bt_cap")

    with col_right:
        with st.container(border=True):
            st.markdown("**🛡️ Risk management**")
            stop_loss = st.slider("Stop-loss per position", 1, 20, 5, format="%d%%",
                help="Auto-sell a position if it falls this % below entry price.") / 100
            max_pos   = st.slider("Max position size", 5, 50, 20, format="%d%%",
                help="Max % of capital in a single buy order.") / 100
            st.divider()
            st.caption(f"Each buy uses ≤ **{max_pos*100:.0f}%** of capital · "
                       f"Positions cut if they fall **{stop_loss*100:.0f}%**")

        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner("Running backtest..."):
            try:
                df_test = download(ticker_bt, start=str(test_start), end=str(test_end))
                df_test = add_all(df_test)
                prices  = df_test["Close"].tolist()
                dates   = df_test.index.tolist()
                macd    = to_macd_series(prices) if use_macd_bt else None

                if st.session_state["dqn_agent"] is not None:
                    agent = st.session_state["dqn_agent"]
                else:
                    agent = DQNAgent()
                    mp = Path(bt_model_path + ".keras")
                    if mp.exists():
                        agent.load(bt_model_path)
                    else:
                        st.warning("No model file found — using untrained agent.")

                buys, sells, _ = agent.backtest(prices, macd=macd)
                engine = BacktestEngine(initial_capital=bt_capital,
                                        stop_loss_pct=stop_loss, max_position_pct=max_pos)
                result = engine.run(prices, buys, sells, ticker=ticker_bt, model_name="DQN")
                bnh    = engine.buy_and_hold_return(prices)

                st.session_state["backtest_result"] = result
                st.session_state["backtest_results_list"].append({
                    "label": f"{ticker_bt} {test_start}→{test_end}",
                    "result": result, "prices": prices,
                    "dates": dates, "bnh": bnh, "capital": bt_capital,
                })

                m     = result.metrics
                alpha = m["total_return_pct"] - bnh
                st.divider()
                st.markdown("### Performance")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Total return", f"{m['total_return_pct']:+.2f}%",
                           delta=f"α {alpha:+.1f}% vs B&H")
                mc2.metric("Sharpe ratio", f"{m['sharpe_ratio']:.3f}",
                           help=">1 is good, >2 is excellent")
                mc3.metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
                mc4.metric("Win rate", f"{m['win_rate_pct']:.1f}%")
                mc5.metric("Trades", m["total_trades"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Price",
                    line=dict(color="#888780", width=1.5)))
                if result.buy_dates:
                    fig.add_trace(go.Scatter(
                        x=[dates[i] for i in result.buy_dates],
                        y=[prices[i] for i in result.buy_dates],
                        mode="markers", name="Buy",
                        marker=dict(symbol="triangle-up", size=11, color="#1D9E75")))
                if result.sell_dates:
                    fig.add_trace(go.Scatter(
                        x=[dates[i] for i in result.sell_dates],
                        y=[prices[i] for i in result.sell_dates],
                        mode="markers", name="Sell / Stop-loss",
                        marker=dict(symbol="triangle-down", size=11, color="#D85A30")))
                fig.update_layout(title=f"{ticker_bt} — signals", xaxis_title="Date",
                    yaxis_title="Price", height=380, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)

                bnh_curve = [bt_capital * (p / prices[0]) for p in prices]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dates[:len(result.portfolio_values)], y=result.portfolio_values,
                    mode="lines", name="DQN strategy",
                    line=dict(color="#534AB7", width=2),
                    fill="tozeroy", fillcolor="rgba(83,74,183,0.05)"))
                fig2.add_trace(go.Scatter(x=dates, y=bnh_curve, mode="lines",
                    name="Buy & hold", line=dict(color="#888780", width=1.5, dash="dash")))
                fig2.update_layout(title="Equity curve vs buy-and-hold",
                    xaxis_title="Date", yaxis_title="Portfolio value",
                    height=320, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=50, b=30))
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)

# ─── TAB 4: Compare ───────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Compare")
    st.caption("Each backtest you run is saved here for side-by-side comparison.")

    results_list = st.session_state["backtest_results_list"]

    if not results_list:
        st.info("Run at least one backtest in the **Backtest** tab to see results here.", icon="👉")
    else:
        rows = []
        for entry in results_list:
            m = entry["result"].metrics
            rows.append({
                "Run":        entry["label"],
                "Return %":   f"{m['total_return_pct']:+.2f}",
                "vs B&H":     f"{m['total_return_pct'] - entry['bnh']:+.2f}",
                "Sharpe":     f"{m['sharpe_ratio']:.3f}",
                "Max DD %":   f"{m['max_drawdown_pct']:.2f}",
                "Win rate %": f"{m['win_rate_pct']:.1f}",
                "Trades":     m["total_trades"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#993556"]
        fig_cmp = go.Figure()
        for i, entry in enumerate(results_list):
            r = entry["result"]
            fig_cmp.add_trace(go.Scatter(
                x=entry["dates"][:len(r.portfolio_values)], y=r.portfolio_values,
                mode="lines", name=entry["label"],
                line=dict(color=colors[i % len(colors)], width=2)))
        last = results_list[-1]
        bnh_ref = [last["capital"] * (p / last["prices"][0]) for p in last["prices"]]
        fig_cmp.add_trace(go.Scatter(x=last["dates"], y=bnh_ref, mode="lines",
            name="Buy & hold (latest)", line=dict(color="#888780", width=1, dash="dot")))
        fig_cmp.update_layout(height=380, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=20, b=30), yaxis_title="Portfolio value")
        st.plotly_chart(fig_cmp, use_container_width=True)

        if results_list[-1]["result"].trades:
            st.markdown("#### Trade log (latest run)")
            df_t = pd.DataFrame(results_list[-1]["result"].trades)
            df_t = df_t[["type","t","price","units","entry_price","pnl"]]
            df_t.columns = ["Type","Day","Exit price","Units","Entry price","P&L"]
            df_t["P&L"] = df_t["P&L"].round(0).astype(int)
            st.dataframe(
                df_t.style.applymap(
                    lambda v: "color:#1D9E75" if isinstance(v,(int,float)) and v>0
                    else ("color:#D85A30" if isinstance(v,(int,float)) and v<0 else ""),
                    subset=["P&L"]),
                use_container_width=True, height=280, hide_index=True)

        st.divider()
        if st.button("🗑️ Clear all results"):
            st.session_state["backtest_results_list"] = []
            st.rerun()
