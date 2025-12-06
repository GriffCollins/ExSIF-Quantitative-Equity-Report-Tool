import yfinance as yf
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.random import Generator, PCG64
from scipy.stats import chi2

#Initial Display
st.title("ExSIF Quantitative Equity Report Tool")
st.write("This website uses the yfinance API to retrieve stock price data.")
ticker = st.text_input("Enter ticker symbol:").upper()
period = st.text_input("Enter period (1y, 2y, 5y, max, etc.):")

if ticker and period:
    #Stock intake and primary calculations
    df = yf.download(tickers=ticker, period=period, auto_adjust=True)
    close = df["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    simple_returns = close.pct_change().dropna()
    daily_std = simple_returns[ticker].std(ddof=1)

    #Plot of Closing Price
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Closing Price')
    ax.set_title('Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    #Max Drawdown
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_drawdown = drawdown[ticker].min()

    #Historical VaR (10-day)
    horizon = 10
    rolling_log_returns = log_returns.rolling(horizon).sum().dropna()
    rolling_simple_returns = np.exp(rolling_log_returns) - 1
    historical_VaR = np.percentile(rolling_simple_returns, 5)
    historical_tail = rolling_simple_returns[rolling_simple_returns > historical_VaR]
    historical_CVaR = historical_tail.mean()

    #Historical Results Display
    st.subheader("Historical Risk Metrics: ")
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown", f"{max_drawdown: .2%}")
    col2.metric("10-day Historical VaR", f"{historical_VaR:.2%}")
    col3.metric('10-day Expected Shortfall', f'{historical_CVaR:.2%}')
    
    #Parametric VaR Calculator
    z_score = 1.6448536270       #95% z-score
    parametric_VaR = -1 * z_score * daily_std
    annual_VaR = parametric_VaR * np.sqrt(252)

    #Parametric VaR test
    df = df[['Close']].dropna().reset_index(drop=True)
    df['breach_signal'] = 0
    df['simple_returns'] = df['Close'].pct_change().dropna()
    std = df['simple_returns'].std(ddof=1)
    z_score = 1.6448536270
    parametric_VaR = -1 * z_score * std
    df.loc[df['simple_returns'] < parametric_VaR, 'breach_signal'] = 1

    #Tests whether frequency of VaR breaches exceeds stated confidence level (0.05)
    def kupiec_test(exceptions, alpha):
        exceptions = np.asarray(exceptions)
        T = len(exceptions)
        N = exceptions.sum()
        p_hat = N / T
        L0 = (1 - alpha) ** (T - N) * (alpha ** N)
        L1 = (1 - p_hat) ** (T - N) * (p_hat ** N)
        LR_uc = -2 * np.log(L0 / L1)
        p_value = 1 - chi2.cdf(LR_uc, df=1)

        return LR_uc, p_value, p_hat

    #Tests whether VaR breachs occur independently
    def christoffersen_independence_test(exceptions):
        I = np.asarray(exceptions).astype(int)
        prev = I[:-1]
        curr = I[1:]
        n00 = int(np.sum((prev == 0) & (curr == 0)))
        n01 = int(np.sum((prev == 0) & (curr == 1)))
        n10 = int(np.sum((prev == 1) & (curr == 0)))
        n11 = int(np.sum((prev == 1) & (curr == 1)))
        denom0 = n00 + n01
        denom1 = n10 + n11
        total_trans = n00 + n01 + n10 + n11
        pi01 = n01 / denom0 if denom0 > 0 else np.nan
        pi11 = n11 / denom1 if denom1 > 0 else np.nan
        T = len(I)
        N = int(I.sum())
        pi_hat = N / T

        def loglike(count_success, count_total, p):
            if count_total == 0:
                return 0.0
            p = np.clip(p, 1e-12, 1 - 1e-12)
            k = count_success
            return k * np.log(p) + (count_total - k) * np.log(1 - p)

        logL_ind = loglike(n01 + n11, total_trans, pi_hat)
        logL_markov = 0.0
        logL_markov += loglike(n01, denom0, pi01 if not np.isnan(pi01) else 0.0)
        logL_markov += loglike(n11, denom1, pi11 if not np.isnan(pi11) else 0.0)
        LR_ind = -2.0 * (logL_ind - logL_markov)
        LR_ind = max(0.0, LR_ind)
        p_value = 1.0 - chi2.cdf(LR_ind, df=1)

        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "denom0": denom0, "denom1": denom1, "total_trans": total_trans,
            "pi01": pi01, "pi11": pi11, "pi_hat": pi_hat,
            "LR_ind": LR_ind, "p_value": p_value
        }

    #Combines both
    def christoffersen_conditional_coverage(LR_uc, LR_ind):
        LR_cc = LR_uc + LR_ind
        p_value = 1 - chi2.cdf(LR_cc, df=2)

        return LR_cc, p_value

    #Run functions
    LR_uc, p_value, p_hat = kupiec_test(df['breach_signal'], 0.05)
    results = christoffersen_independence_test(df['breach_signal'])
    LR_ind = results['LR_ind']
    LR_cc, p_value = christoffersen_conditional_coverage(LR_uc, LR_ind)

    #Parametric Results Display
    st.subheader("Parametric Risk Results (Normal): ")
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Parametric VaR", f"{parametric_VaR:.2%}")
    col2.metric("Annual Parametric VaR", f"{annual_VaR:.2%}")

    if p_value >= 0.05:
        col3.success(f"Model assumptions have passed the test.")
    else:
        col3.error(f"Model assumptions have failed the test, ignore results.")

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

    #Monte Carlo Graph Paths Calculator
    num_simulations = 200
    num_days = 252
    drift = mu - 0.5 * sigma**2
    z = rng.normal(size=(num_days, num_simulations))
    log_ret_paths = drift + sigma * z
    start_price = close[ticker].iloc[-1]
    price_paths = start_price * np.exp(np.cumsum(log_ret_paths, axis=0))

    #Monte Carlo Graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_paths)
    ax.set_title("Monte Carlo Price Paths")
    ax.set_xlabel("Days")
    ax.set_ylabel("Simulated Price")
    ax.grid(True)
    st.pyplot(fig)

    #VaR Metrics Display
    st.subheader("Monte Carlo Simulation Results: ")
    col1, col2 = st.columns(2)
    col1.metric("Monte Carlo 95% VaR", f"{VaR:.2%}")
    col2.metric("Monte Carlo 95% Expected Shortfall", f"{CVaR:.2%}")

    #Sharpe ratio
    risk_free_rate = 0.05
    annual_mean = mu * 252
    annual_std = sigma * np.sqrt(252)
    sharpe = (annual_mean - risk_free_rate) / annual_std

    #1 year CAPM regression
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
    alpha = (1+alpha)**252-1

    #Treynor Ratio
    simple_annual_mean = simple_returns[ticker].mean()*252
    treynor = (simple_annual_mean - risk_free_rate) / beta

    #Rolling CAPM regresssion
    def rolling_beta(df, window):
        results = pd.DataFrame(index=df.index, columns=["alpha", "beta"])

        for i in range(window, len(df)):
            y = df["stock"].iloc[i - window:i]
            X = sm.add_constant(df['market'].iloc[i - window:i])
            model = sm.OLS(y, X).fit()
            results.iloc[i] = [model.params['const'], model.params['market']]

        return results.dropna()

    def rolling_volatility(df, window, trading_days):
        return df.ewm(span=window).std() * np.sqrt(trading_days)

    rolling_params = rolling_beta(merged, 30)
    results = pd.concat([merged, rolling_params], axis=1).dropna()
    results['Stock_Volatility'] = rolling_volatility(results['stock'], 90, 252)
    results['Market_Volatility'] = rolling_volatility(results['market'], 90, 252)

    #30-day rolling CAPM regression graph
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

    #Risk-Adjusted Performance Metrics Display
    st.subheader("Risk Adjusted Performances: ")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", value=f'{sharpe:.4f}')
    col2.metric("Alpha", value=f'{alpha:.4f}')
    col3.metric("Daily Beta", value=f'{beta:.4f}')
    col4.metric("Treynor Ratio", value=f'{treynor:.4f}')

    #90-day rolling volatility graph
    fig, ax = plt.subplots()
    results['Stock_Volatility'].plot(label="" + ticker + " Volatility", color='blue')
    results['Market_Volatility'].plot(label='SPY Volatility', color='orange', alpha=0.7)
    ax.set_title('90-Day Rolling Annualized Volatility')
    ax.set_ylabel('Volatility')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

