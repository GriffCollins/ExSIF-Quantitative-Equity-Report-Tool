import yfinance as yf
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.random import Generator, PCG64

#Initial Display
st.title("ExSIF Quantitative Equity Report Tool")
st.write("This website uses yfinance API to retrieve stock price data.")
ticker = st.text_input("Enter ticker symbol:").upper()
period = st.text_input("Enter period (1y, 2y, 5y, max, etc.):")

if ticker and period:
    df = yf.download(tickers=ticker, period=period, auto_adjust=True)
    close = df["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    simple_returns = close.pct_change().dropna()
    daily_std = simple_returns[ticker].std(ddof=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['Close'], label='Closing Price')
    ax.set_title('Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_drawdown = drawdown[ticker].min()

    st.write(f"Max Drawdown: {max_drawdown:.2%}")

    #Historical VaR (10-day)
    horizon = 10
    rolling_log_returns = log_returns.rolling(horizon).sum().dropna()
    rolling_simple_returns = np.exp(rolling_log_returns) - 1
    historical_VaR = np.percentile(rolling_simple_returns, 5)

    #Parametric VaR
    z_score = 1.6448536270       #95% z-score
    parametric_VaR = -1 * z_score * daily_std
    annual_VaR = parametric_VaR * np.sqrt(252)

    #Monte Carlo VaR
    rng = Generator(PCG64(seed=42))
    num_simulations = 100000
    simulation_days = 252
    mu = log_returns[ticker].mean()
    sigma = log_returns[ticker].std(ddof=1)
    portfolio_returns = np.zeros(num_simulations)

    for i in range(num_simulations):
        z = rng.normal(size=simulation_days)
        log_mu = mu - 0.5 * sigma ** 2
        log_returns_sim = log_mu + sigma * z
        cumulative_return = np.exp(log_returns_sim.sum()) - 1
        portfolio_returns[i] = cumulative_return

    losses = -portfolio_returns
    VaR = np.quantile(losses, 0.95)
    tail_losses = losses[losses > VaR]
    CVaR = tail_losses.mean()
    VaR = -VaR
    CVaR = -CVaR

    #VaR Metrics Display
    st.subheader("VaR Metrics: ")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("10-day Historical VaR", f"{historical_VaR:.2%}")
    col2.metric("Daily Parametric VaR", f"{parametric_VaR:.2%}")
    col3.metric("Annual Parametric VaR", f"{annual_VaR:.2%}")
    col4.metric("Monte Carlo 95% VaR", f"{VaR:.2%}")
    col5.metric("Monte Carlo 95% Expected Shortfall", f"{CVaR:.2%}")

    #Sharpe ratio
    risk_free_rate = 0.05
    annual_mean = mu * 252
    annual_std = sigma * np.sqrt(252)
    sharpe = (annual_mean - risk_free_rate) / annual_std

    #Beta
    market = yf.download("SPY", period=period, auto_adjust=True)
    market_returns = market["Close"].pct_change().dropna()
    merged = pd.concat([simple_returns, market_returns], axis=1)
    merged.columns = ["stock", "market"]
    merged = merged.dropna()

    X = sm.add_constant(merged["market"])
    y = merged["stock"]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params["market"]

    #Treynor Ratio
    simple_annual_mean = simple_returns[ticker].mean()*252
    treynor = (simple_annual_mean - risk_free_rate) / beta

    #Risk-Adjusted Performance Metrics Display
    st.subheader("Risk Adjusted Performances: ")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", value=f'{sharpe:.4f}')
    col2.metric("Alpha", value=f'{alpha:.4f}')
    col3.metric("Daily Beta", value=f'{beta:.4f}')
    col4.metric("Treynor Ratio", value=f'{treynor:.4f}')

    def rolling_beta(df, window):
        # Initialise the dataframe
        results = pd.DataFrame(index=df.index, columns=["alpha", "beta"])

        for i in range(window, len(df)):
            y = df["stock"].iloc[i - window:i]  # Stock returns
            X = sm.add_constant(df['market'].iloc[i - window:i])  # Market returns

            model = sm.OLS(y, X).fit()
            results.iloc[i] = [model.params['const'], model.params['market']]

        return results.dropna()

    # Rolling volatility function and multiply by sqrt(252) to annualise
    def rolling_volatility(df, window, trading_days):
        return df.rolling(window).std() * np.sqrt(trading_days)

    rolling_params = rolling_beta(merged, 30)
    results = pd.concat([merged, rolling_params], axis=1).dropna()
    results['Stock_Volatility'] = rolling_volatility(results['stock'], 90, 252)
    results['Market_Volatility'] = rolling_volatility(results['market'], 90, 252)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    results['beta'].plot(ax=ax1, title=f"30-Day Rolling Beta ({ticker} vs SPY)")
    ax1.axhline(1, color='r', linestyle='--')
    ax1.set_ylabel("Beta")
    ax1.grid(True)

    results['alpha'].plot(ax=ax2, title="30-Day Rolling Alpha")
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_ylabel("Alpha")
    ax2.grid(True)

    st.pyplot(fig)

    fig, ax = plt.subplots()
    results['Stock_Volatility'].plot(label="" + ticker + " Volatility", color='blue')
    results['Market_Volatility'].plot(label='SPY Volatility', color='orange', alpha=0.7)
    ax.set_title('90-Day Rolling Annualized Volatility')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid()
    st.pyplot(fig)






