import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Streamlit App Title
st.title("Backtesting Trading Strategies")

# Sidebar - User inputs
st.sidebar.header("User Input Parameters")

# Stock ticker input
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL', max_chars=10)

# Date range input
start_date = st.sidebar.date_input('Start Date', value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', value=datetime.date.today())

# Moving average strategy parameters
st.sidebar.subheader("Moving Average Strategy Parameters")
short_window = st.sidebar.number_input('Short Moving Average Window', min_value=1, max_value=50, value=20)
long_window = st.sidebar.number_input('Long Moving Average Window', min_value=50, max_value=200, value=50)

# Fetch the stock data
@st.cache
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Return'] = data['Adj Close'].pct_change()
    return data

# Get data based on user inputs
data = get_data(ticker, start_date, end_date)

# Display the stock data if checkbox is checked
if st.sidebar.checkbox("Show Raw Data", False):
    st.subheader(f"Raw Data for {ticker}")
    st.write(data.tail())

# Moving Average Crossover Strategy
def moving_average_crossover(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Adj Close']
    signals['Short_MA'] = data['Adj Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data['Adj Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['Signal'] = 0.0
    signals['Signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1.0, 0.0)
    signals['Position'] = signals['Signal'].diff()
    return signals

# Generate the signals
signals = moving_average_crossover(data, short_window, long_window)

# Plot the stock price and moving averages with buy/sell signals
def plot_signals(signals):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot adjusted closing price
    ax.plot(signals.index, signals['Price'], label='Price', alpha=0.5)
    
    # Plot short and long moving averages
    ax.plot(signals.index, signals['Short_MA'], label=f'Short MA ({short_window})', alpha=0.75)
    ax.plot(signals.index, signals['Long_MA'], label=f'Long MA ({long_window})', alpha=0.75)
    
    # Plot buy signals
    ax.plot(signals[signals['Position'] == 1].index,
            signals['Short_MA'][signals['Position'] == 1],
            '^', markersize=10, color='g', lw=0, label='Buy Signal')

    # Plot sell signals
    ax.plot(signals[signals['Position'] == -1].index,
            signals['Short_MA'][signals['Position'] == -1],
            'v', markersize=10, color='r', lw=0, label='Sell Signal')
    
    ax.set_title(f"Moving Average Crossover Strategy: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    st.pyplot(fig)

# Call the plotting function
plot_signals(signals)

# Calculate the portfolio performance
def portfolio_performance(signals, initial_capital=100000):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = 100 * signals['Signal']  # 100 shares per buy
    portfolio = positions.multiply(signals['Price'], axis=0)
    pos_diff = positions.diff()

    portfolio['holdings'] = positions['Stock'] * signals['Price']
    portfolio['cash'] = initial_capital - (pos_diff['Stock'] * signals['Price']).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    return portfolio

# Generate portfolio performance metrics
portfolio = portfolio_performance(signals)

# Plot the portfolio value over time
st.subheader("Portfolio Performance")
def plot_portfolio_performance(portfolio):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio.index, portfolio['total'], label='Portfolio Value', alpha=0.75)
    ax.set_title(f"Portfolio Value Over Time: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    st.pyplot(fig)

plot_portfolio_performance(portfolio)

# Show portfolio performance metrics
st.subheader("Performance Metrics")
total_return = portfolio['total'][-1] / portfolio['total'][0] - 1
st.write(f"Total Return: {total_return*100:.2f}%")
st.write(f"Final Portfolio Value: ${portfolio['total'][-1]:.2f}")
