"""
Interactive Streamlit Application for LOB Deep Learning Predictor.
Showcases real-time Level-2 Limit Order Book depth, microstructural feature engineering,
PyTorch neural network inference (1D-CNN & LSTM), and an interactive strategy backtester.
"""

import os
import sys
import time
import urllib.request
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F

from src.ingestion.synthetic_data import generate_synthetic_lob_dataset
from src.features.feature_engineering import LOBFeatureEngineer
from src.dataset.lob_dataset import create_lob_dataloaders
from src.models import get_lob_model
from src.evaluation.backtester import LOBBacktester

# Streamlit Page Setup
st.set_page_config(
    page_title="LOB Deep Learning Predictor | Quant Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Terminal Theme)
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; }
    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1a233a 100%);
        border: 1px solid #233152;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-long { color: #00e676; font-weight: bold; }
    .badge-short { color: #ff5252; font-weight: bold; }
    .badge-neutral { color: #bdbdbd; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset(num_ticks: int = 4000, seed: int = 42):
    """Generate or retrieve cached realistic LOB data."""
    return generate_synthetic_lob_dataset(num_ticks=num_ticks, seed=seed)


def fetch_live_binance_snapshot(symbol: str = "BTCUSDT", depth: int = 10) -> pd.DataFrame:
    """Fetch real-time Level-2 depth snapshot from Binance public API."""
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={depth}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode())

    record = {"timestamp": int(time.time() * 1000)}
    bids = data["bids"]
    asks = data["asks"]

    for i in range(depth):
        lvl = i + 1
        record[f"ask_price_{lvl}"] = float(asks[i][0])
        record[f"ask_volume_{lvl}"] = float(asks[i][1])
        record[f"bid_price_{lvl}"] = float(bids[i][0])
        record[f"bid_volume_{lvl}"] = float(bids[i][1])

    return pd.DataFrame([record])


@st.cache_resource
def load_trained_model(model_type: str, in_features: int, lookback_window: int = 50):
    """Load or initialize PyTorch model (supports CPU / CUDA dynamic dispatch)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_lob_model(model_type=model_type, in_features=in_features, lookback_window=lookback_window)
    checkpoint_path = "checkpoints/best_lob_model.pt"

    if os.path.exists(checkpoint_path) and model_type == "cnn":
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
        except Exception:
            pass

    model.to(device)
    model.eval()
    return model, device


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white", output_format="auto")
    st.image("https://img.shields.io/badge/CUDA-12.8%20%7C%20RTX%205060%20Ti-76B900?logo=nvidia&logoColor=white", output_format="auto")
    st.image("https://img.shields.io/badge/Market-Binance%20L2%20Depth-F0B90B?logo=binance&logoColor=black", output_format="auto")

    st.markdown("## ⚙️ Model & Pipeline Controls")
    model_type = st.selectbox("Neural Architecture", options=["cnn", "lstm"], index=0, format_func=lambda x: "1D-CNN (Temporal Receptive Field)" if x == "cnn" else "Recurrent LSTM (Deep Sequential)")
    lookback_window = st.slider("Lookback Window (N ticks)", min_value=20, max_value=100, value=50, step=10)

    st.markdown("---")
    st.markdown("## 📊 Strategy Backtest Parameters")
    confidence_threshold = st.slider("Trade Conviction Threshold", min_value=0.50, max_value=0.90, value=0.70, step=0.05)
    fee_bps = st.slider("Transaction Fee (bps)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

    st.markdown("---")
    st.markdown(
        """
        ### 👨‍💻 Project Information
        * **Domain**: Quantitative Finance & High-Frequency Trading
        * **Target**: Short-Term Mid-Price Direction (`-1, 0, 1`)
        * **Compute**: WSL2 / CUDA 12.8 / RTX 5060 Ti
        """
    )


# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.title("⚡ Limit Order Book (LOB) Deep Learning Predictor")
st.markdown(
    """
    *A high-frequency quantitative market microstructure pipeline in **PyTorch** designed to predict discrete mid-price movements
    from Level-2 Limit Order Book depth snapshots.*
    """
)

# Load data and prepare feature engineering
raw_df = get_cached_dataset(num_ticks=3500)
fe = LOBFeatureEngineer(rolling_window=50, future_horizon=10, threshold=0.00005)
clean_df, feature_cols = fe.transform(raw_df)
model, device = load_trained_model(model_type, in_features=len(feature_cols), lookback_window=lookback_window)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Real-Time LOB Depth",
    "🧠 Deep Learning Inference",
    "💰 Backtesting & PnL",
    "📐 Architecture & Formulas",
])

# ==============================================================================
# TAB 1: ORDER BOOK DEPTH & MICROSTRUCTURE
# ==============================================================================
with tab1:
    st.subheader("Level-2 Order Book Microstructure (10 Levels)")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        live_fetch = st.button("🌐 Fetch Live Binance BTC Snapshot", type="primary")

    if live_fetch:
        try:
            live_snapshot = fetch_live_binance_snapshot("BTCUSDT", depth=10)
            current_row = live_snapshot.iloc[0]
            st.success("Successfully ingested live Level-2 snapshot from Binance WebSocket/REST API!")
        except Exception as e:
            st.warning(f"Could not connect to Binance API: {e}. Using simulated snapshot.")
            current_row = clean_df.iloc[-1]
    else:
        current_row = clean_df.iloc[-1]

    # Extract price & volume ladders
    ask_prices = [current_row[f"ask_price_{i}"] for i in range(1, 11)]
    ask_vols = [current_row[f"ask_volume_{i}"] for i in range(1, 11)]
    bid_prices = [current_row[f"bid_price_{i}"] for i in range(1, 11)]
    bid_vols = [current_row[f"bid_volume_{i}"] for i in range(1, 11)]

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    mid_price = (best_ask + best_bid) / 2.0
    spread = best_ask - best_bid
    obi_l1 = (bid_vols[0] - ask_vols[0]) / (bid_vols[0] + ask_vols[0] + 1e-8)
    obi_total = (sum(bid_vols) - sum(ask_vols)) / (sum(bid_vols) + sum(ask_vols) + 1e-8)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mid-Price ($)", f"${mid_price:,.2f}")
    m2.metric("Bid-Ask Spread ($)", f"${spread:.2f}")
    m3.metric("L1 Imbalance (OBI)", f"{obi_l1:+.3f}")
    m4.metric("Total 10-Lvl OBI", f"{obi_total:+.3f}")

    # Plot Cumulative Depth Chart
    cum_bids = np.cumsum(bid_vols)
    cum_asks = np.cumsum(ask_vols)

    fig_depth = go.Figure()
    fig_depth.add_trace(
        go.Scatter(
            x=bid_prices[::-1],
            y=cum_bids[::-1],
            fill="tozeroy",
            name="Cumulative Bids (Buy)",
            line=dict(color="#00e676", width=2),
            fillcolor="rgba(0, 230, 118, 0.2)",
        )
    )
    fig_depth.add_trace(
        go.Scatter(
            x=ask_prices,
            y=cum_asks,
            fill="tozeroy",
            name="Cumulative Asks (Sell)",
            line=dict(color="#ff5252", width=2),
            fillcolor="rgba(255, 82, 82, 0.2)",
        )
    )

    fig_depth.update_layout(
        title="Level-2 Order Book Cumulative Depth Chart",
        xaxis_title="Price ($)",
        yaxis_title="Cumulative Volume (Contracts)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_depth, width="stretch")


# ==============================================================================
# TAB 2: DEEP LEARNING INFERENCE
# ==============================================================================
with tab2:
    st.subheader("Model Inference on Sliding Window Sequences")

    st.markdown(
        f"Feeding a sliding window sequence of **{lookback_window} historical ticks** "
        f"across **{len(feature_cols)} normalized features** through the **{model_type.upper()}** model:"
    )

    # Take the latest lookback window
    window_data = clean_df[feature_cols].iloc[-lookback_window:].values.astype(np.float32)
    input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(input_tensor).cpu().numpy()[0]

    p_down, p_flat, p_up = probs[0], probs[1], probs[2]

    # Signal Logic
    if p_up > confidence_threshold:
        signal = "BUY / LONG"
        color = "#00e676"
    elif p_down > confidence_threshold:
        signal = "SELL / SHORT"
        color = "#ff5252"
    else:
        signal = "HOLD / FLAT"
        color = "#ffd600"

    col_gauges, col_verdict = st.columns([2, 1])

    with col_gauges:
        fig_prob = go.Figure(
            go.Bar(
                x=[p_down * 100, p_flat * 100, p_up * 100],
                y=["Down (-1)", "Stationary (0)", "Up (+1)"],
                orientation="h",
                marker=dict(color=["#ff5252", "#9e9e9e", "#00e676"]),
                text=[f"{p_down * 100:.1f}%", f"{p_flat * 100:.1f}%", f"{p_up * 100:.1f}%"],
                textposition="auto",
            )
        )
        fig_prob.add_vline(x=confidence_threshold * 100, line_dash="dash", line_color="yellow", annotation_text="Conviction Threshold")
        fig_prob.update_layout(
            title="Model Softmax Output Probabilities (%)",
            xaxis_title="Probability (%)",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig_prob, width="stretch")

    with col_verdict:
        st.markdown(
            f"""
            <div style="background-color: #131b2e; border: 2px solid {color}; border-radius: 10px; padding: 20px; text-align: center; margin-top: 25px;">
                <h4 style="margin: 0; color: #bdbdbd;">Generated Signal</h4>
                <h1 style="color: {color}; margin: 10px 0; font-size: 32px;">{signal}</h1>
                <p style="margin: 0; color: #e0e0e0;">Active Hardware: <b>{device.type.upper()}</b></p>
                <p style="margin: 0; color: #9e9e9e; font-size: 13px;">Threshold: {confidence_threshold * 100:.0f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Feature heatmap of sequence
    st.markdown("#### Input Feature Sequence Tensor (Lookback = 50 Ticks)")
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=window_data.T,
            x=list(range(lookback_window)),
            y=feature_cols,
            colorscale="Viridis",
            colorbar=dict(title="Z-Score"),
        )
    )
    fig_heat.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Tick History (t - Lookback -> t)",
        yaxis_title="Features",
    )
    st.plotly_chart(fig_heat, width="stretch")


# ==============================================================================
# TAB 3: BACKTESTING & STRATEGY SIMULATION
# ==============================================================================
with tab3:
    st.subheader("Quantitative Strategy Simulation (Out-of-Sample Validation)")

    # Prepare validation dataset
    train_loader, val_loader, _ = create_lob_dataloaders(
        df=clean_df,
        feature_cols=feature_cols,
        lookback_window=lookback_window,
        batch_size=256,
        train_split=0.75,
        num_workers=0,
    )

    all_val_probs = []
    all_val_targets = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            probs = model.predict_proba(bx).cpu().numpy()
            all_val_probs.append(probs)
            all_val_targets.append(by.numpy())

    val_probs = np.vstack(all_val_probs)
    val_targets = np.concatenate(all_val_targets)
    val_preds = np.argmax(val_probs, axis=1)

    val_start = int(len(clean_df) * 0.75) + lookback_window - 1
    val_prices = clean_df["mid_price"].iloc[val_start:].values[:len(val_probs)]

    backtester = LOBBacktester(
        confidence_threshold=confidence_threshold,
        transaction_fee_bps=fee_bps,
    )
    metrics = backtester.compute_metrics(val_targets, val_preds)
    results = backtester.simulate_strategy(val_probs, val_prices)

    # Metric Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades", f"{results['total_trades']}")
    c2.metric("Win Rate", f"{results['win_rate'] * 100:.1f}%")
    c3.metric("Gross PnL", f"${results['total_gross_pnl']:.2f}")
    c4.metric("Fees Paid", f"${results['total_fees']:.2f}")
    c5.metric("Net PnL", f"${results['total_net_pnl']:.2f}", delta=f"{results['total_net_pnl']:.2f}")

    # Cumulative PnL chart
    pos = np.zeros(len(val_probs))
    pos[val_probs[:, 2] > confidence_threshold] = 1
    pos[val_probs[:, 0] > confidence_threshold] = -1

    price_diffs = np.zeros(len(val_prices))
    price_diffs[1:] = np.diff(val_prices)
    held_pos = np.zeros(len(val_prices))
    held_pos[1:] = pos[:-1]

    gross_pnl = held_pos * price_diffs
    chg = np.zeros(len(val_prices))
    chg[0] = abs(pos[0])
    chg[1:] = np.abs(np.diff(pos))
    fees = chg * val_prices * (fee_bps / 10000.0)
    net_pnl = gross_pnl - fees

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(y=np.cumsum(gross_pnl), mode="lines", name="Gross PnL ($)", line=dict(color="#29b6f6", width=2)))
    fig_pnl.add_trace(go.Scatter(y=np.cumsum(net_pnl), mode="lines", name="Net PnL (after fees)", line=dict(color="#00e676", width=2.5)))

    fig_pnl.update_layout(
        title="Strategy Cumulative Profit & Loss (Validation Set)",
        xaxis_title="Ticks",
        yaxis_title="PnL ($)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_pnl, width="stretch")


# ==============================================================================
# TAB 4: ARCHITECTURE & MATHEMATICAL FORMULATION
# ==============================================================================
with tab4:
    st.subheader("Mathematical Foundations & Microstructural Formulas")

    st.markdown(
        r"""
        ### 1. Microstructural Order Book Signals
        * **Mid-Price**:
          $$P_{mid, t} = \frac{P_{ask, 1, t} + P_{bid, 1, t}}{2}$$
        * **Bid-Ask Spread**:
          $$S_t = P_{ask, 1, t} - P_{bid, 1, t}$$
        * **Order Book Imbalance (OBI)**:
          $$OBI_t = \frac{V_{bid, 1, t} - V_{ask, 1, t}}{V_{bid, 1, t} + V_{ask, 1, t}}$$

        ### 2. Causal Rolling Z-Score Normalization
        To prevent lookahead bias (data leakage), normalization statistics are computed strictly over historical tick windows $W$:
        $$z_t = \frac{x_t - \mu_W}{\sigma_W + \epsilon}, \quad \text{where } \mu_W = \frac{1}{W}\sum_{i=0}^{W-1} x_{t-i}$$

        ### 3. Smoothed Forward Direction Labeling
        Rather than predicting noisy tick-to-tick jumps, the mid-price is smoothed over future horizon $k$ ticks:
        $$\bar{P}_{mid, future}(t) = \frac{1}{k}\sum_{j=1}^{k} P_{mid}(t + j), \quad R(t) = \frac{\bar{P}_{mid, future}(t) - P_{mid}(t)}{P_{mid}(t)}$$
        The target label $y_t \in \{-1, 0, 1\}$ is determined by threshold $\theta$:
        $$y_t = \begin{cases} +1 & \text{if } R(t) > \theta \\ -1 & \text{if } R(t) < -\theta \\ 0 & \text{otherwise} \end{cases}$$
        """
    )
