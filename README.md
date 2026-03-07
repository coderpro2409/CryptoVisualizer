Streamlit Dashboard Project Project Overview

The Crypto Volatility Visualizer is an interactive dashboard built using Python and Streamlit to help users understand cryptocurrency price movements and market volatility.

Cryptocurrency markets such as Bitcoin (BTC) are highly volatile, meaning their prices can change rapidly in short periods of time. For beginners and investors, it can be difficult to understand how these fluctuations occur and how risk behaves in financial markets.

This project visualizes real BTC/USD market data and combines it with mathematical simulations such as sine waves, random noise, and drift trends to help users explore and understand market behaviour.

The dashboard allows users to interactively adjust parameters such as amplitude, frequency, and drift, and instantly observe how these mathematical functions affect simulated price movements.

The application was developed using Python, Pandas, NumPy, Plotly, and Streamlit.

Project Team
Het Thakkar Prahaan Sanghvi

CRS: Artificial Intelligence – AI

Project Objectives
The objective of this project is to:

Analyze cryptocurrency price data

Understand market volatility

Apply mathematical functions to simulate price movements

Build an interactive financial dashboard

Deploy the application using Streamlit Cloud

This project connects mathematics, data science, and financial analysis into a practical AI application.

Dataset Used
The project uses Bitcoin historical price data containing the following columns:

Column Description
Timestamp Time at which the trade data was recorded Open Price at the beginning of the time period High Highest price during the period Low Lowest price during the period Close Final price at the end of the period Volume Total trading volume

The dataset is cleaned and processed using Pandas before visualization.

Data Preparation
The dataset undergoes several preprocessing steps:

Loading Data
The dataset is loaded using pandas.read_csv().

Timestamp Conversion
The Timestamp column is converted into a proper datetime format for accurate time-based plotting.

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

Data Cleaning
Missing timestamps are removed

Data is resampled based on timeframe

Only relevant columns are selected

Resampling
Users can select different timeframes:

1 Day

1 Week

1 Month

This helps users visualize price behaviour at different time scales.

Mathematical Simulation
This simulates a long-term upward or downward market trend.

Streamlit Dashboard Features

The dashboard includes interactive controls that allow users to explore different market scenarios.

Sidebar Controls

Users can modify several parameters:

Candle Timeframe

1 Day

1 Week

1 Month

Pattern Type

Sine / Cosine waves

Random noise

Amplitude

Controls the size of price fluctuations.

Frequency

Controls the speed of market cycles.

Drift

Controls the long-term upward or downward trend.

Visualizations
The dashboard contains multiple financial visualizations.

Candlestick Chart with Simulation
This chart displays:

Actual BTC price (candlestick)

Simulated price movement (line)

The candlestick chart shows real market behaviour while the simulation demonstrates how mathematical models affect prices.

High vs Low Price Comparison
This graph shows daily high and low prices.

It helps visualize price volatility within each time period.

Trading Volume Analysis
A bar chart is used to display trading volume.

This allows users to observe whether higher trading activity corresponds with larger price movements.

Stable vs Volatile Period Detection
The application calculates rolling volatility using log returns.

returns = np.log(data["Close"] / data["Close"].shift(1))v A threshold is used to classify periods as:

Stable

Volatile

Volatile points are highlighted on the chart.

Key Metrics Displayed

The dashboard displays two important financial indicators:

Volatility Index

Standard deviation of log returns.

𝜎
𝑠 𝑡 𝑑 ( 𝑙 𝑜 𝑔 𝑟 𝑒 𝑡 𝑢 𝑟 𝑛 𝑠 ) σ=std(log returns)

This measures how unstable the market is.

Average Drift

Mean return of the dataset.

𝜇
𝑚 𝑒 𝑎 𝑛 ( 𝑙 𝑜 𝑔 𝑟 𝑒 𝑡 𝑢 𝑟 𝑛 𝑠 ) μ=mean(log returns)

This indicates the average market trend.

Technologies Used Technology Purpose Python Core programming language Pandas Data cleaning and processing NumPy Mathematical calculations Plotly Interactive financial charts Streamlit Dashboard interface GitHub Version control Streamlit Cloud Application deployment How to Run the Project

Install Dependencies pip install streamlit pandas numpy plotly
Run the Streamlit App streamlit run main.py
Open in Browser
The dashboard will open automatically at:

https://crypto-volatility-visualizer-kniks82zrj5njaymrxwnwv.streamlit.app/)Deployment

The project is deployed using Streamlit Cloud.

Conclusion

The Crypto Volatility Visualizer demonstrates how mathematics, programming, and data visualization can be combined to better understand financial markets.

By allowing users to interactively adjust parameters such as amplitude, frequency, and drift, the dashboard helps illustrate how volatility behaves in cryptocurrency markets.

This project shows how AI and mathematical modeling can make complex financial concepts easier to understand for learners and investors.
