import yfinance as yf
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from numpy.random import Generator, PCG64
from scipy.stats import chi2

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="exSIF Quant Report",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}
.stApp { background-color: #0a0a0a; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.stTextInput > div > div > input {
    background-color: #111;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.stTextInput > label { color: #888; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.1em; }

[data-testid="metric-container"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #666; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #e8e8e8; font-family: 'IBM Plex Mono', monospace; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #555;
    text-transform: uppercase;
    border-top: 1px solid #1e1e1e;
    padding-top: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.ticker-banner {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 0;
}
.ticker-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}
.pass-box {
    background: #0d1f0d;
    border: 1px solid #1a3a1a;
    color: #4caf50;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
}
.fail-box {
    background: #1f0d0d;
    border: 1px solid #3a1a1a;
    color: #f44336;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
}
.stSuccess, .stError { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; letter-spacing: 0.2em; color: #333; margin-bottom: 0.25rem;'>
EXETER STUDENT INVESTMENT FUND
</div>
<div style='font-size: 1.5rem; font-family: IBM Plex Mono, monospace; font-weight: 600; color: #e8e8e8; margin-bottom: 0.1rem;'>
Quantitative Equity Report
</div>
<div style='font-family: IBM Plex Mono, monospace; font-size: 0.65rem; color: #333; letter-spacing: 0.1em; margin-bottom: 2rem;'>
─────────────────────────────────────────────
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
col_a, col_b, col_c = st.columns([2, 2, 6])
with col_a:
    ticker = st.text_input("TICKER", placeholder="e.g. AAPL").upper().strip()
with col_b:
    period = st.text_input("PERIOD", placeholder="1y, 2y, 5y, max").strip()

if not ticker or not period:
    st.markdown("<div style='color:#333; font-family: IBM Plex Mono, monospace; font-size:0.75rem; margin-top:3rem;'>↑ enter a ticker and period to generate report</div>", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
with st.spinner(""):
    raw = yf.download(tickers=ticker, period=period, auto_adjust=True)

if raw.empty:
    st.error(f"No data returned for **{ticker}**. Check the ticker symbol and period.")
    st.stop()

# Flatten multi-level columns if present
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

close = raw["Close"].squeeze().dropna()

if len(close) < 30:
    st.error("Not enough data — try a longer period (e.g. 2y).")
    st.stop()

log_returns    = np.log(close / close.shift(1)).dropna()
simple_returns = close.pct_change().dropna()
daily_std      = simple_returns.std(ddof=1)

# ─────────────────────────────────────────────
# TICKER BANNER
# ─────────────────────────────────────────────
latest_price  = close.iloc[-1]
price_change  = close.iloc[-1] - close.iloc[-2]
pct_change    = price_change / close.iloc[-2]
arrow         = "▲" if price_change >= 0 else "▼"
colour        = "#4caf50" if price_change >= 0 else "#f44336"

st.markdown(f"""
<div class='ticker-banner'>{ticker}</div>
<div class='ticker-sub'>
    <span style='color:#e8e8e8; font-size:1.1rem;'>${latest_price:,.2f}</span>
    &nbsp;&nbsp;
    <span style='color:{colour}'>{arrow} {abs(pct_change):.2%}</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    period: {period}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
DARK_BG   = "#0a0a0a"
PANEL_BG  = "#0f0f0f"
GRID_COL  = "#1a1a1a"
TEXT_COL  = "#888888"
ACC1      = "#e8e8e8"
ACC2      = "#4a9eff"
ACC3      = "#ff6b35"

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(ACC1)
    ax.title.set_fontfamily("monospace")
    ax.title.set_fontsize(9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COL)
    ax.spines["bottom"].set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")

def make_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=DARK_BG, **kwargs)
    return fig

# ─────────────────────────────────────────────
# SECTION 1 — PRICE & DRAWDOWN
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>01 &nbsp;/&nbsp; Price History & Drawdown</div>", unsafe_allow_html=True)

running_max  = close.cummax()
drawdown     = (close - running_max) / running_max
max_drawdown = drawdown.min()

fig = make_fig(figsize=(14, 6))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig)

ax1 = fig.add_subplot(gs[0])
ax1.plot(close.index, close.values, color=ACC1, linewidth=0.9, alpha=0.95)
ax1.fill_between(close.index, close.values, close.values.min(), alpha=0.05, color=ACC2)
style_ax(ax1)
ax1.set_title(f"{ticker} — Closing Price")
ax1.set_xticklabels([])
ax1.set_ylabel("Price (USD)")

ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(drawdown.index, drawdown.values, 0, color=ACC3, alpha=0.6)
ax2.axhline(max_drawdown, color=ACC3, linewidth=0.7, linestyle="--")
style_ax(ax2)
ax2.set_title("Drawdown")
ax2.set_ylabel("Drawdown %")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("Max Drawdown", f"{max_drawdown:.2%}")
c2.metric("Current Drawdown", f"{drawdown.iloc[-1]:.2%}")

# ─────────────────────────────────────────────
# SECTION 2 — HISTORICAL VAR
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>02 &nbsp;/&nbsp; Historical Risk Metrics</div>", unsafe_allow_html=True)

horizon = 10
rolling_log   = log_returns.rolling(horizon).sum().dropna()
rolling_simp  = np.exp(rolling_log) - 1

if len(rolling_simp) == 0:
    st.error("Not enough data to compute 10-day historical VaR. Try a longer period.")
    st.stop()

historical_VaR = np.percentile(rolling_simp, 5)
historical_ES  = rolling_simp[rolling_simp <= historical_VaR].mean()

# Return distribution chart
fig, ax = make_fig(figsize=(14, 4)), None
fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
ax.set_facecolor(PANEL_BG)
counts, bins, patches = ax.hist(
    rolling_simp * 100, bins=60,
    color=ACC2, alpha=0.6, edgecolor="none"
)
# Colour tail red
for patch, left in zip(patches, bins[:-1]):
    if left / 100 <= historical_VaR:
        patch.set_facecolor(ACC3)
        patch.set_alpha(0.8)

ax.axvline(historical_VaR * 100, color=ACC3, linewidth=1.2, linestyle="--",
           label=f"95% VaR: {historical_VaR:.2%}")
style_ax(ax)
ax.set_title(f"{ticker} — 10-Day Rolling Return Distribution")
ax.set_xlabel("Return (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("10-Day Historical VaR (95%)", f"{historical_VaR:.2%}")
c2.metric("10-Day Historical ES (95%)", f"{historical_ES:.2%}")

# ─────────────────────────────────────────────
# SECTION 3 — PARAMETRIC VAR + BACKTESTS
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>03 &nbsp;/&nbsp; Parametric VaR & Backtesting</div>", unsafe_allow_html=True)

z_95           = 1.6448536270
parametric_VaR = -z_95 * daily_std
annual_VaR     = parametric_VaR * np.sqrt(252)

# Build breach series from simple_returns (avoid overwriting close/df)
breach_series = (simple_returns < parametric_VaR).astype(int)

# ── Kupiec POF Test ──
def kupiec_test(exceptions, alpha=0.05):
    I   = np.asarray(exceptions)
    T   = len(I)
    N   = int(I.sum())
    p   = N / T if T > 0 else np.nan
    L0  = (1 - alpha) ** (T - N) * (alpha ** N)
    L1  = (1 - p) ** (T - N) * (p ** N) if 0 < p < 1 else 1e-300
    LR  = -2 * np.log(L0 / L1)
    pv  = 1 - chi2.cdf(LR, df=1)
    return LR, pv, p

# ── Christoffersen Independence Test ──
def christoffersen_independence(exceptions):
    I    = np.asarray(exceptions).astype(int)
    prev = I[:-1]; curr = I[1:]
    n00  = int(((prev == 0) & (curr == 0)).sum())
    n01  = int(((prev == 0) & (curr == 1)).sum())
    n10  = int(((prev == 1) & (curr == 0)).sum())
    n11  = int(((prev == 1) & (curr == 1)).sum())
    d0   = n00 + n01; d1 = n10 + n11
    pi01 = n01 / d0 if d0 > 0 else 1e-10
    pi11 = n11 / d1 if d1 > 0 else 1e-10
    T    = len(I); N = int(I.sum())
    pi   = N / T

    def ll(k, n, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return k * np.log(p) + (n - k) * np.log(1 - p) if n > 0 else 0.0

    logL_ind    = ll(n01 + n11, n00 + n01 + n10 + n11, pi)
    logL_markov = ll(n01, d0, pi01) + ll(n11, d1, pi11)
    LR          = max(0.0, -2 * (logL_ind - logL_markov))
    pv          = 1 - chi2.cdf(LR, df=1)
    return LR, pv

LR_uc, p_uc, p_hat = kupiec_test(breach_series, 0.05)
LR_ind, p_ind       = christoffersen_independence(breach_series)
LR_cc               = LR_uc + LR_ind
p_cc                = 1 - chi2.cdf(LR_cc, df=2)

# Breach chart
fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
style_ax(ax)
ax.plot(simple_returns.index, simple_returns.values * 100,
        color=ACC1, linewidth=0.5, alpha=0.7, label="Daily Return")
ax.axhline(parametric_VaR * 100, color=ACC3, linewidth=1.0, linestyle="--",
           label=f"VaR ({parametric_VaR:.2%})")
breach_dates = simple_returns[breach_series.values.astype(bool)].index
ax.scatter(breach_dates, simple_returns[breach_dates].values * 100,
           color=ACC3, s=12, zorder=5, label=f"Breaches ({int(breach_series.sum())})")
ax.set_title(f"{ticker} — Parametric VaR Breach Map")
ax.set_ylabel("Daily Return (%)")
ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Daily Parametric VaR (95%)", f"{parametric_VaR:.2%}")
c2.metric("Annual Parametric VaR (95%)", f"{annual_VaR:.2%}")
c3.metric("Empirical Breach Rate", f"{p_hat:.2%}", delta=f"expected 5.00%", delta_color="off")

# Backtest results table
bt_data = {
    "Test": ["Kupiec POF", "Christoffersen Independence", "Conditional Coverage (CC)"],
    "LR Statistic": [f"{LR_uc:.4f}", f"{LR_ind:.4f}", f"{LR_cc:.4f}"],
    "p-value": [f"{p_uc:.4f}", f"{p_ind:.4f}", f"{p_cc:.4f}"],
    "Result": [
        "✓ Pass" if p_uc >= 0.05 else "✗ Fail",
        "✓ Pass" if p_ind >= 0.05 else "✗ Fail",
        "✓ Pass" if p_cc >= 0.05 else "✗ Fail",
    ]
}
bt_df = pd.DataFrame(bt_data)
st.dataframe(bt_df, use_container_width=True, hide_index=True)

overall_pass = (p_uc >= 0.05) and (p_ind >= 0.05) and (p_cc >= 0.05)
if overall_pass:
    st.markdown("<div class='pass-box'>✓ &nbsp;Model passed all three backtests — parametric VaR assumptions hold.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='fail-box'>✗ &nbsp;Model failed one or more backtests — interpret parametric VaR with caution.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 4 — MONTE CARLO
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>04 &nbsp;/&nbsp; Monte Carlo Simulation</div>", unsafe_allow_html=True)

rng             = Generator(PCG64(seed=42))
N_SIM           = 100_000
SIM_DAYS        = 252
mu              = log_returns.mean()
sigma           = log_returns.std(ddof=1)
log_mu          = mu - 0.5 * sigma ** 2

# Vectorised simulation
z_mat           = rng.normal(size=(SIM_DAYS, N_SIM))
log_paths       = log_mu + sigma * z_mat
cum_returns     = np.exp(log_paths.sum(axis=0)) - 1
losses          = -cum_returns
MC_VaR_loss     = np.quantile(losses, 0.95)
MC_ES_loss      = losses[losses > MC_VaR_loss].mean()
MC_VaR          = -MC_VaR_loss
MC_ES           = -MC_ES_loss

# Price paths (200 for display)
N_PATHS         = 200
z_paths         = rng.normal(size=(SIM_DAYS, N_PATHS))
log_ret_paths   = log_mu + sigma * z_paths
start_price     = close.iloc[-1]
price_paths     = start_price * np.exp(np.cumsum(log_ret_paths, axis=0))

# Percentile bands
p5   = np.percentile(price_paths, 5,  axis=1)
p25  = np.percentile(price_paths, 25, axis=1)
p50  = np.percentile(price_paths, 50, axis=1)
p75  = np.percentile(price_paths, 75, axis=1)
p95  = np.percentile(price_paths, 95, axis=1)
days = np.arange(SIM_DAYS)

fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
style_ax(ax)
# Plot a subset of raw paths very faintly
for i in range(min(80, N_PATHS)):
    ax.plot(days, price_paths[:, i], color=ACC2, linewidth=0.3, alpha=0.08)
ax.fill_between(days, p5,  p95, color=ACC2, alpha=0.12, label="5–95th pct")
ax.fill_between(days, p25, p75, color=ACC2, alpha=0.22, label="25–75th pct")
ax.plot(days, p50, color=ACC2, linewidth=1.5, label="Median")
ax.axhline(start_price, color=ACC3, linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_title(f"{ticker} — Monte Carlo Price Paths (1Y, GBM, n={N_PATHS})")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Simulated Price (USD)")
ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
st.pyplot(fig)
plt.close(fig)

# Return distribution
fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=DARK_BG)
style_ax(ax)
ax.hist(cum_returns * 100, bins=120, color=ACC2, alpha=0.5, edgecolor="none")
ax.axvline(MC_VaR * 100, color=ACC3, linewidth=1.2, linestyle="--",
           label=f"95% VaR: {MC_VaR:.2%}")
ax.axvline(MC_ES * 100, color="#ff9933", linewidth=1.0, linestyle=":",
           label=f"95% ES: {MC_ES:.2%}")
style_ax(ax)
ax.set_title(f"{ticker} — Simulated 1Y Return Distribution (n={N_SIM:,})")
ax.set_xlabel("1Y Return (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
c1.metric("MC 95% Annual VaR", f"{MC_VaR:.2%}")
c2.metric("MC 95% Annual ES", f"{MC_ES:.2%}")
c3.metric("MC Median 1Y Return", f"{np.median(cum_returns):.2%}")

# ─────────────────────────────────────────────
# SECTION 5 — CAPM & RISK-ADJUSTED METRICS
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>05 &nbsp;/&nbsp; CAPM & Risk-Adjusted Performance</div>", unsafe_allow_html=True)

with st.spinner(""):
    mkt_raw = yf.download("SPY", period=period, auto_adjust=True)

if isinstance(mkt_raw.columns, pd.MultiIndex):
    mkt_raw.columns = mkt_raw.columns.get_level_values(0)

market_returns = mkt_raw["Close"].squeeze().pct_change().dropna()
merged = pd.concat([simple_returns, market_returns], axis=1, join="inner").dropna()
merged.columns = ["stock", "market"]

if len(merged) < 30:
    st.warning("Insufficient overlapping data with SPY for CAPM.")
    st.stop()

# Full-period OLS
X     = sm.add_constant(merged["market"])
model = sm.OLS(merged["stock"], X).fit()
alpha_daily = model.params["const"]
beta        = model.params["market"]
alpha_ann   = (1 + alpha_daily) ** 252 - 1

# Risk-free rate
rf = 0.05
annual_mu      = mu * 252
annual_sigma   = sigma * np.sqrt(252)
sharpe         = (annual_mu - rf) / annual_sigma
treynor        = (simple_returns.mean() * 252 - rf) / beta
info_ratio     = alpha_ann / (model.resid.std() * np.sqrt(252)) if model.resid.std() > 0 else np.nan
sortino_down   = simple_returns[simple_returns < 0].std(ddof=1) * np.sqrt(252)
sortino        = (annual_mu - rf) / sortino_down if sortino_down > 0 else np.nan
calmar         = annual_mu / abs(max_drawdown) if max_drawdown != 0 else np.nan

# Rolling beta/alpha
def rolling_ols(df, window=63):
    idx    = df.index[window:]
    alphas = np.full(len(df), np.nan)
    betas  = np.full(len(df), np.nan)
    for i in range(window, len(df)):
        ys = df["stock"].iloc[i - window:i].values
        xs = df["market"].iloc[i - window:i].values
        A  = np.vstack([np.ones(len(xs)), xs]).T
        res = np.linalg.lstsq(A, ys, rcond=None)[0]
        alphas[i] = res[0]
        betas[i]  = res[1]
    out = pd.DataFrame({"alpha": alphas, "beta": betas}, index=df.index)
    return out.dropna()

rolling = rolling_ols(merged, window=63)
merged_full = pd.concat([merged, rolling], axis=1).dropna()

# Rolling volatility
ev_stock  = merged_full["stock"].ewm(span=90).std()  * np.sqrt(252)
ev_market = merged_full["market"].ewm(span=90).std() * np.sqrt(252)

# Beta + Alpha chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                facecolor=DARK_BG, gridspec_kw={"hspace": 0.1})
merged_full["beta"].plot(ax=ax1, color=ACC2, linewidth=1.0)
ax1.axhline(1.0, color=ACC3, linewidth=0.8, linestyle="--", alpha=0.7)
ax1.axhline(beta, color=ACC1, linewidth=0.6, linestyle=":", alpha=0.4,
            label=f"Full-period β = {beta:.3f}")
style_ax(ax1)
ax1.set_title(f"{ticker} — 63-Day Rolling Beta vs SPY")
ax1.set_ylabel("Beta")
ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

merged_full["alpha"].plot(ax=ax2, color=ACC2, linewidth=0.8, alpha=0.7)
ax2.axhline(0, color=ACC3, linewidth=0.8, linestyle="--", alpha=0.7)
ax2.fill_between(merged_full.index, merged_full["alpha"], 0,
                 where=merged_full["alpha"] > 0, color="#4caf50", alpha=0.2)
ax2.fill_between(merged_full.index, merged_full["alpha"], 0,
                 where=merged_full["alpha"] < 0, color=ACC3, alpha=0.2)
style_ax(ax2)
ax2.set_title("63-Day Rolling Daily Alpha")
ax2.set_ylabel("Alpha (daily)")
st.pyplot(fig)
plt.close(fig)

# Volatility chart
fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
style_ax(ax)
ev_stock.plot(ax=ax, color=ACC2, linewidth=1.0, label=f"{ticker}")
ev_market.plot(ax=ax, color=ACC3, linewidth=1.0, alpha=0.8, label="SPY")
ax.set_title("90-Day EWM Annualised Volatility")
ax.set_ylabel("Volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
st.pyplot(fig)
plt.close(fig)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe Ratio",  f"{sharpe:.3f}")
c2.metric("Sortino Ratio", f"{sortino:.3f}")
c3.metric("Treynor Ratio", f"{treynor:.4f}")
c4.metric("Calmar Ratio",  f"{calmar:.3f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jensen's Alpha (Ann.)", f"{alpha_ann:.2%}")
c2.metric("Beta",                   f"{beta:.4f}")
c3.metric("R²",                     f"{model.rsquared:.4f}")
c4.metric("Annual Volatility",      f"{annual_sigma:.2%}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='
    margin-top: 4rem;
    border-top: 1px solid #1a1a1a;
    padding-top: 1rem;
    font-family: IBM Plex Mono, monospace;
    font-size: 0.6rem;
    color: #2a2a2a;
    letter-spacing: 0.08em;
'>
EXETER STUDENT INVESTMENT FUND &nbsp;·&nbsp; QUANTITATIVE STRATEGY &nbsp;·&nbsp;
FOR INTERNAL USE ONLY &nbsp;·&nbsp; DATA: YAHOO FINANCE VIA YFINANCE API
</div>
""", unsafe_allow_html=True)
ax.grid()
st.pyplot(fig)
