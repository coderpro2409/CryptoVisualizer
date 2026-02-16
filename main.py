import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Crypto Volatility Visualizer", layout="wide")
st.title("Crypto Volatility Visualizer – BTC/USD")


# ------------------------------------------------------------
# Sidebar – controls
# ------------------------------------------------------------
st.sidebar.header("Data view")

timeframe = st.sidebar.selectbox(
    "Candle timeframe",
    ["1D", "1W", "1M"]
)

st.sidebar.header("Simulation parameters")

pattern = st.sidebar.selectbox(
    "Pattern",
    ["Sine / Cosine", "Random noise"]
)

amplitude = st.sidebar.slider(
    "Amplitude",
    0, 1000, 150
)

frequency = st.sidebar.slider(
    "Frequency",
    1, 100, 4
)

# ------------------------------------------------------------
# Drift slider (negative → down, positive → up)
# ------------------------------------------------------------
st.sidebar.markdown("### Trend (Drift)")

drift_strength = st.sidebar.slider(
    "Downtrend  ⟵   Drift   ⟶  Uptrend",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=1.0
)


# ------------------------------------------------------------
# Timeframe mapping
# ------------------------------------------------------------
tf_map = {
    "1D": "1D",
    "1W": "1W",
    "1M": "1M"
}


# ------------------------------------------------------------
# Load and resample data
# ------------------------------------------------------------
@st.cache_data
def load_and_resample(path, rule):

    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp")

    ohlcv = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    ohlcv = ohlcv.dropna().reset_index()

    return ohlcv


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data = load_and_resample(
    "btcusd_1-min_data.csv",
    tf_map[timeframe]
)

MAX_POINTS = 900
data = data.tail(MAX_POINTS).copy()


# ------------------------------------------------------------
# Simulated price (pattern + drift)
# ------------------------------------------------------------
t = np.arange(len(data))
base_price = data["Close"].values

if pattern == "Sine / Cosine":
    wave = amplitude * np.sin(2 * np.pi * frequency * t / len(t))
else:
    wave = np.random.normal(0, amplitude, size=len(t))


# linear drift (true slope)
total_drift_fraction = drift_strength / 100.0
start_price = base_price[0]

drift_line = np.linspace(
    0,
    total_drift_fraction * start_price,
    len(base_price)
)

data["Simulated"] = base_price + wave + drift_line


# ------------------------------------------------------------
# Volatility + drift statistics
# ------------------------------------------------------------
returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

volatility = returns.std()
avg_drift = returns.mean()

c1, c2 = st.columns(2)

c1.metric("Volatility index (std of log returns)", f"{volatility:.6f}")
c2.metric("Average drift (mean return)", f"{avg_drift:.6f}")


# ------------------------------------------------------------
# Hover text for candlestick (old Plotly safe)
# ------------------------------------------------------------
hover_text = (
    "Open: " + data["Open"].round(2).astype(str) +
    "<br>High: " + data["High"].round(2).astype(str) +
    "<br>Low: " + data["Low"].round(2).astype(str) +
    "<br>Close: " + data["Close"].round(2).astype(str)
)


# ------------------------------------------------------------
# MAIN GRAPH – actual candlestick + simulated line
# ------------------------------------------------------------
st.subheader("Actual BTC price (candlestick) with simulated trend overlay")

fig = go.Figure()

# Actual price – candlestick
fig.add_trace(
    go.Candlestick(
        x=data["Timestamp"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        text=hover_text,
        hoverinfo="x+text",
        name="Actual price"
    )
)

# Simulated price – line on the same chart
fig.add_trace(
    go.Scatter(
        x=data["Timestamp"],
        y=data["Simulated"],
        mode="lines",
        name="Simulated price (with drift)"
    )
)

fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    height=560
)

fig.update_xaxes(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor"
)

fig.update_yaxes(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor"
)

st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# High vs Low
# ------------------------------------------------------------
st.subheader("High vs Low comparison")

fig_hl = go.Figure()

fig_hl.add_trace(go.Scatter(
    x=data["Timestamp"],
    y=data["High"],
    name="High",
    mode="lines"
))

fig_hl.add_trace(go.Scatter(
    x=data["Timestamp"],
    y=data["Low"],
    name="Low",
    mode="lines"
))

fig_hl.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=320
)

st.plotly_chart(fig_hl, use_container_width=True)


# ------------------------------------------------------------
# Volume
# ------------------------------------------------------------
st.subheader("Trading volume")

fig_vol = go.Figure()

fig_vol.add_trace(go.Bar(
    x=data["Timestamp"],
    y=data["Volume"],
    name="Volume"
))

fig_vol.update_layout(
    xaxis_title="Time",
    yaxis_title="Volume",
    hovermode="x unified",
    height=280
)

st.plotly_chart(fig_vol, use_container_width=True)


# ------------------------------------------------------------
# Stable vs volatile periods
# ------------------------------------------------------------
st.subheader("Stable vs volatile periods")

rolling_vol = returns.rolling(10).std()
threshold = rolling_vol.median()

sv = data.iloc[1:].copy()
sv["State"] = np.where(rolling_vol > threshold, "Volatile", "Stable")

fig_sv = go.Figure()

fig_sv.add_trace(go.Scatter(
    x=sv["Timestamp"],
    y=sv["Close"],
    mode="lines",
    name="Price"
))

fig_sv.add_trace(go.Scatter(
    x=sv[sv["State"] == "Volatile"]["Timestamp"],
    y=sv[sv["State"] == "Volatile"]["Close"],
    mode="markers",
    name="Volatile points",
    marker=dict(size=6)
))

fig_sv.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=320
)

st.plotly_chart(fig_sv, use_container_width=True)