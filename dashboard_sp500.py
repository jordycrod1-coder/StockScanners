# ============================================================
# S&P 500 PROFESSIONAL MARKET + SECTOR ROTATION DASHBOARD
# ============================================================
#
# Designed around short-term sector emergence:
#   5D / 10D / 20D relative strength
#   market breadth
#   sector breadth
#   volume participation
#   volatility regime
#   short/intermediate RRG
#   cross-asset confirmation
#
# Requirements:
#   pip install yfinance pandas numpy requests matplotlib lxml
#
# Run the entire script from the beginning after restarting the
# Jupyter kernel if you previously ran an older version.
#
# LAYOUT NOTE (this version): every chart that plots a daily value against
# a *date* x-axis (price, RSI, VIX, breadth, A/D line, new highs/lows,
# up/down volume, growth/value, small/large, TRIN, McClellan) has been
# pulled into one vertical stack, all sharing the SPY price panel's x-axis
# and the same TIMESERIES_LOOKBACK_DAYS lookback window (~252 trading days,
# i.e. back to roughly Sep 2025). Only the bottom chart in the stack shows
# date tick labels since the x-axis is shared/aligned across all of them.
# Charts whose x-axis is NOT time (categorical bar charts, sector tables,
# the weekly heatmap, the correlation matrix, and the RRGs, whose axes are
# RS-Ratio / RS-Momentum, not dates) stay where they were, below the stack.
# ============================================================

import io
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend — required for GitHub Actions (no display)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# OUTPUT (for GitHub Pages)
# ============================================================
# Every chart is saved as a PNG into OUTPUT_DIR instead of being shown
# inline, and an index.html tying them together is written at the end.
# The GitHub Actions workflow publishes OUTPUT_DIR to GitHub Pages on
# every scheduled run, so the same Pages link always reflects the latest
# run.
OUTPUT_DIR = os.environ.get("DASHBOARD_OUTPUT_DIR", "site")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATED_IMAGES = []  # [(filename, title), ...] in the order charts are saved


def save_current_figure(fig_obj, filename, title):
    """Save fig_obj to OUTPUT_DIR/filename, close it, and register it for index.html."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig_obj.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_obj)
    GENERATED_IMAGES.append((filename, title))

# ============================================================
# SETTINGS
# ============================================================

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"

MARKET_HISTORY = "3y"  # bumped from 2y: monthly RRG needs ~14 months of history
# before its rolling(10)/rolling(4) windows even produce a first data point, so 2y
# only left room for ~10 valid monthly RRG points instead of a full 12.
CONSTITUENT_HISTORY = "1y"

SECTOR_ETFS = {
    "XLE": "Energy",
    "XLB": "Materials",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLK": "Information Technology",
    "XLC": "Communication Services",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

SECTOR_COLORS = {
    "Energy": "#2ca02c",
    "Materials": "#8c564b",
    "Industrials": "#7f7f7f",
    "Consumer Discretionary": "#e377c2",
    "Consumer Staples": "#ff7f0e",
    "Health Care": "#d62728",
    "Financials": "#bcbd22",
    "Information Technology": "#1f77b4",
    "Communication Services": "#9467bd",
    "Utilities": "#17becf",
    "Real Estate": "#c49c94",
}

RS_WINDOWS = [5, 10, 20, 60]

BREADTH_MA_WINDOWS = [20, 50, 200]

# TIMESERIES_LOOKBACK_DAYS is the single source of truth for how far back
# every date-axis chart in the vertical stack looks. It matches the SPY
# price panel (tail(252), ~1 trading year -> roughly Sep 2025 to today), so
# all stacked charts share both the same x-axis limits (via sharex) AND the
# same underlying date range.
TIMESERIES_LOOKBACK_DAYS = 252
BREADTH_LOOKBACK_DAYS = TIMESERIES_LOOKBACK_DAYS
MCCLELLAN_LOOKBACK_DAYS = TIMESERIES_LOOKBACK_DAYS

HEATMAP_LOOKBACK_WEEKS = 13
RRG_WEEKS = 8
SHORT_RRG_DAYS = 20
MONTHLY_RRG_MONTHS = 12

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
    }
)


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal_period=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line


# The next three (Stochastic, MFI, CMF) are only used by the SPY 12-panel side
# column (Daily/Weekly/Monthly x Price+Volume/MACD/RSI+Stoch/MFI+CMF) placed
# beside the 13-chart time-aligned stack — the rest of the dashboard doesn't
# need them.
def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%K"] = df["%K"].rolling(window=smooth_k).mean()
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    return df


def calculate_mfi(df, period=14):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    up = typical_price > typical_price.shift(1)
    down = typical_price < typical_price.shift(1)
    positive_mf = raw_money_flow.where(up, 0.0).rolling(window=period).sum()
    negative_mf = raw_money_flow.where(down, 0.0).rolling(window=period).sum()
    money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)
    return 100 - (100 / (1 + money_flow_ratio))


def calculate_cmf(df, period=20):
    hl_range = df["High"] - df["Low"]
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range.replace(0, np.nan)
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"]
    return mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def clean_download(ticker, period, interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def pct_return(series, days):
    s = series.dropna()
    if len(s) <= days:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-days - 1] - 1) * 100)


def safe_last(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def slope_pct(series, lookback=10):
    s = pd.Series(series).dropna()
    if len(s) <= lookback:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100)


def pct_above_ma(close_wide, window):
    ma = close_wide.rolling(window).mean()
    return (close_wide > ma).mean(axis=1) * 100


def classify_trend(close):
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    c = safe_last(close)
    m20 = safe_last(ma20)
    m50 = safe_last(ma50)
    m200 = safe_last(ma200)
    if np.isnan(m200):
        return "Insufficient data"
    slope50 = slope_pct(ma50, 20)
    if c > m20 > m50 > m200 and slope50 > 0:
        return "Strong Uptrend"
    if c > m50 and slope50 > 0:
        return "Uptrend"
    if c < m20 < m50 < m200 and slope50 < 0:
        return "Strong Downtrend"
    if c < m50 and slope50 < 0:
        return "Downtrend"
    return "Mixed"


def get_sp500_table():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text))[0]
    table.columns = [str(c).strip() for c in table.columns]
    table["Symbol"] = table["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return table


def get_constituent_sector_table():
    table = get_sp500_table()
    return table[["Symbol", "GICS Sector"]].rename(columns={"Symbol": "Ticker", "GICS Sector": "Sector"})


def get_sector_members(constituents_df, sector_name, available):
    members = constituents_df.loc[constituents_df["Sector"] == sector_name, "Ticker"].tolist()
    return [ticker for ticker in members if ticker in available]


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

print("Downloading SPY / VIX / cross-asset data...")

spy = clean_download(MARKET_TICKER, MARKET_HISTORY)
vix = clean_download(VIX_TICKER, MARKET_HISTORY)
vix3m = clean_download(VIX3M_TICKER, MARKET_HISTORY)

if spy is None:
    raise RuntimeError("SPY data could not be downloaded.")

cross_asset_tickers = {
    "10Y": "^TNX",
    "3M T-Bill": "^IRX",  # NOTE: was previously mislabeled "2Y" — ^IRX is the 13-week
    # (3-month) T-Bill rate. Yahoo Finance has no reliable free 2Y yield ticker.
    "DXY": "DX-Y.NYB",
    "Oil": "CL=F",
    "Gold": "GC=F",
    "HYG": "HYG",
    "LQD": "LQD",
    "PutCall": "^CPC",  # CBOE equity put/call ratio — Yahoo's coverage of this can be
    # spotty/delayed; code below tolerates it being unavailable without breaking anything.
}

cross_asset = {}
for name, ticker in cross_asset_tickers.items():
    try:
        cross_asset[name] = clean_download(ticker, MARKET_HISTORY)
    except Exception:
        cross_asset[name] = None


def get_cross_close(name):
    df = cross_asset.get(name)
    if df is None or df.empty:
        return np.nan
    return safe_last(df["Close"])


last_10y_yield = get_cross_close("10Y")
last_3m_yield = get_cross_close("3M T-Bill")
if not np.isnan(last_10y_yield) and not np.isnan(last_3m_yield):
    yield_curve_spread = last_10y_yield - last_3m_yield
else:
    yield_curve_spread = np.nan

if np.isnan(yield_curve_spread):
    yield_curve_status = "N/A"
elif yield_curve_spread < 0:
    yield_curve_status = "Inverted (recession signal)"
else:
    yield_curve_status = "Normal"

last_putcall = get_cross_close("PutCall")

# ============================================================
# STYLE / SIZE ROTATION (Growth vs Value, Small vs Large Cap)
# ============================================================

STYLE_TICKERS = {"Growth": "IWF", "Value": "IWD", "SmallCap": "IWM"}

style_data = {}
for name, ticker in STYLE_TICKERS.items():
    try:
        style_data[name] = clean_download(ticker, MARKET_HISTORY)
    except Exception:
        style_data[name] = None

growth_value_ratio = None
if style_data.get("Growth") is not None and style_data.get("Value") is not None:
    aligned_gv = pd.concat(
        [style_data["Growth"]["Close"], style_data["Value"]["Close"]], axis=1, join="inner"
    ).dropna()
    if not aligned_gv.empty:
        aligned_gv.columns = ["growth", "value"]
        growth_value_ratio = aligned_gv["growth"] / aligned_gv["value"]

small_large_ratio = None
if style_data.get("SmallCap") is not None:
    aligned_sl = pd.concat(
        [style_data["SmallCap"]["Close"], spy["Close"]], axis=1, join="inner"
    ).dropna()
    if not aligned_sl.empty:
        aligned_sl.columns = ["small", "large"]
        small_large_ratio = aligned_sl["small"] / aligned_sl["large"]

print("Downloading sector ETFs...")

sector_data = {}
for etf in SECTOR_ETFS:
    try:
        df = clean_download(etf, MARKET_HISTORY)
        if df is not None:
            sector_data[etf] = df
    except Exception as e:
        print(f"Could not download {etf}: {e}")

# ============================================================
# S&P 500 CONSTITUENTS
# ============================================================

print("Downloading current S&P 500 constituent list...")

constituents_df = get_constituent_sector_table()
all_tickers = constituents_df["Ticker"].dropna().unique().tolist()

print(f"Downloading {len(all_tickers)} S&P 500 constituent histories...")

BATCH_SIZE = 75
close_frames = []
volume_frames = []

for start in range(0, len(all_tickers), BATCH_SIZE):
    batch = all_tickers[start : start + BATCH_SIZE]
    print(f"  Batch {start + 1}-{min(start + BATCH_SIZE, len(all_tickers))}")
    try:
        batch_data = yf.download(
            batch,
            period=CONSTITUENT_HISTORY,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if batch_data is None or batch_data.empty:
            continue
        if isinstance(batch_data.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in batch_data.columns.get_level_values(0):
                    continue
                try:
                    close_frames.append(batch_data[ticker]["Close"].rename(ticker))
                    volume_frames.append(batch_data[ticker]["Volume"].rename(ticker))
                except Exception:
                    continue
        else:
            if len(batch) == 1 and "Close" in batch_data.columns:
                ticker = batch[0]
                close_frames.append(batch_data["Close"].rename(ticker))
                volume_frames.append(batch_data["Volume"].rename(ticker))
    except Exception as e:
        print(f"  Batch failed: {e}")

if not close_frames:
    raise RuntimeError("No S&P 500 constituent data was downloaded.")

close_wide = pd.concat(close_frames, axis=1)
volume_wide = pd.concat(volume_frames, axis=1)
close_wide = close_wide.loc[:, ~close_wide.columns.duplicated()].sort_index()
volume_wide = volume_wide.loc[:, ~volume_wide.columns.duplicated()].sort_index()

print(f"Usable constituent price series: {close_wide.shape[1]} / {len(all_tickers)}")

# ============================================================
# MARKET INDICATORS
# ============================================================

spy["MA20"] = spy["Close"].rolling(20).mean()
spy["MA50"] = spy["Close"].rolling(50).mean()
spy["MA200"] = spy["Close"].rolling(200).mean()
spy["RSI"] = calculate_rsi(spy["Close"])
spy["MACD"], spy["Signal"] = calculate_macd(spy["Close"])

last_close = safe_last(spy["Close"])
last_ma20 = safe_last(spy["MA20"])
last_ma50 = safe_last(spy["MA50"])
last_ma200 = safe_last(spy["MA200"])
last_rsi = safe_last(spy["RSI"])
last_macd = safe_last(spy["MACD"])
last_signal = safe_last(spy["Signal"])

market_returns = {f"{d}D": pct_return(spy["Close"], d) for d in [1, 5, 10, 20, 60]}

ath_252 = spy["Close"].rolling(252, min_periods=100).max()
distance_from_ath = (last_close / safe_last(ath_252) - 1) * 100

ma_slopes = {
    "20MA slope 10D": slope_pct(spy["MA20"], 10),
    "50MA slope 10D": slope_pct(spy["MA50"], 10),
    "200MA slope 20D": slope_pct(spy["MA200"], 20),
}

# ============================================================
# MARKET BREADTH
# ============================================================

breadth = {}
for window in BREADTH_MA_WINDOWS:
    breadth[window] = pct_above_ma(close_wide, window).tail(BREADTH_LOOKBACK_DAYS)

daily_pct = close_wide.pct_change()
advances = (daily_pct > 0).sum(axis=1)
declines = (daily_pct < 0).sum(axis=1)
unchanged = (daily_pct == 0).sum(axis=1)
ad_net = advances - declines
ad_line = ad_net.cumsum().tail(BREADTH_LOOKBACK_DAYS)

# ============================================================
# MCCLELLAN OSCILLATOR / SUMMATION INDEX
# Standard formula: EMA(19) of net advances minus EMA(39) of net advances.
# NOTE: the Summation Index is technically supposed to start from a specific
# seeded value carried forward since the 1960s — ours is a relative cumulative
# sum starting from the available data window, so it's directionally correct
# (rising/falling, overbought/oversold zones) but won't match an "official"
# absolute level from a data vendor with decades of history.
# ============================================================

ema19 = ad_net.ewm(span=19, adjust=False).mean()
ema39 = ad_net.ewm(span=39, adjust=False).mean()
mcclellan_osc = (ema19 - ema39).dropna()
mcclellan_summation = mcclellan_osc.cumsum()

mcclellan_osc_plot = mcclellan_osc.tail(MCCLELLAN_LOOKBACK_DAYS)
mcclellan_summation_plot = mcclellan_summation.tail(MCCLELLAN_LOOKBACK_DAYS)
latest_mcclellan = safe_last(mcclellan_osc)

# ============================================================
# NEW 52-WEEK HIGHS / LOWS
# ============================================================

rolling_max = close_wide.rolling(252, min_periods=100).max()
rolling_min = close_wide.rolling(252, min_periods=100).min()
new_highs = (close_wide >= rolling_max).sum(axis=1).tail(BREADTH_LOOKBACK_DAYS)
new_lows = (close_wide <= rolling_min).sum(axis=1).tail(BREADTH_LOOKBACK_DAYS)

# ============================================================
# UP / DOWN VOLUME
# ============================================================

up_volume = volume_wide.where(daily_pct > 0, 0).sum(axis=1)
down_volume = volume_wide.where(daily_pct < 0, 0).sum(axis=1)
up_down_volume_ratio = up_volume / down_volume.replace(0, np.nan)
up_volume_pct = (up_volume / (up_volume + down_volume).replace(0, np.nan)) * 100

# ============================================================
# ARMS INDEX (TRIN)
# TRIN = (Advances/Declines) / (Up Volume/Down Volume)
# TRIN > 1 = more volume in decliners than the A/D count alone suggests (bearish)
# TRIN < 1 = more volume in advancers (bullish)
# ============================================================

trin = (advances / declines.replace(0, np.nan)) / (up_volume / down_volume.replace(0, np.nan))
trin = trin.replace([np.inf, -np.inf], np.nan)
trin_plot = trin.tail(BREADTH_LOOKBACK_DAYS)
latest_trin = safe_last(trin)

# ============================================================
# BREADTH SNAPSHOT
# ============================================================

latest_breadth = {window: safe_last(breadth[window]) for window in BREADTH_MA_WINDOWS}

breadth_change_5d = {}
breadth_change_20d = {}
for window in BREADTH_MA_WINDOWS:
    series = breadth[window].dropna()
    if len(series) > 5:
        breadth_change_5d[window] = series.iloc[-1] - series.iloc[-6]
    else:
        breadth_change_5d[window] = np.nan
    if len(series) > 20:
        breadth_change_20d[window] = series.iloc[-1] - series.iloc[-21]
    else:
        breadth_change_20d[window] = np.nan

latest_adv = int(safe_last(advances))
latest_dec = int(safe_last(declines))
latest_highs = int(safe_last(new_highs))
latest_lows = int(safe_last(new_lows))
latest_upvol_pct = safe_last(up_volume_pct)
latest_ud_ratio = safe_last(up_down_volume_ratio)

# ============================================================
# SPY RELATIVE VOLUME
# ============================================================

spy_avg_vol20 = spy["Volume"].rolling(20).mean()
spy_rvol = safe_last(spy["Volume"] / spy_avg_vol20)

# ============================================================
# VIX REGIME
# ============================================================

if vix is not None:
    last_vix = safe_last(vix["Close"])
    vix_5d_change = pct_return(vix["Close"], 5)
    vix_20d_change = pct_return(vix["Close"], 20)
else:
    last_vix = np.nan
    vix_5d_change = np.nan
    vix_20d_change = np.nan

if vix3m is not None:
    last_vix3m = safe_last(vix3m["Close"])
else:
    last_vix3m = np.nan

if not np.isnan(last_vix) and not np.isnan(last_vix3m):
    vix_term_spread = last_vix - last_vix3m
else:
    vix_term_spread = np.nan

if np.isnan(last_vix):
    vix_zone = "N/A"
elif last_vix < 15:
    vix_zone = "Low"
elif last_vix < 20:
    vix_zone = "Normal"
elif last_vix < 30:
    vix_zone = "Elevated"
else:
    vix_zone = "High"

if np.isnan(vix_term_spread):
    vol_structure = "N/A"
elif vix_term_spread > 0:
    vol_structure = "Stress / Backwardation"
else:
    vol_structure = "Normal / Contango"

# ============================================================
# MARKET REGIME SCORE
# ============================================================

regime_score = 0
regime_score += 1 if last_close > last_ma20 else -1
regime_score += 1 if last_close > last_ma50 else -1
regime_score += 1 if last_close > last_ma200 else -1
regime_score += 1 if last_ma50 > last_ma200 else -1
regime_score += 1 if last_macd > last_signal else -1
regime_score += 1 if last_rsi > 50 else -1
regime_score += 1 if latest_breadth[20] > 50 else -1
regime_score += 1 if latest_breadth[50] > 50 else -1
regime_score += 1 if latest_upvol_pct > 50 else -1
if not np.isnan(last_vix):
    if last_vix < 20:
        regime_score += 1
    elif last_vix > 30:
        regime_score -= 1

if regime_score >= 7:
    regime_label = "BULLISH / CONSTRUCTIVE"
elif regime_score <= -4:
    regime_label = "BEARISH / CAUTIONARY"
else:
    regime_label = "NEUTRAL / MIXED"

# ============================================================
# SECTOR STATISTICS
# ============================================================

spy_rs_returns = {f"{d}D": pct_return(spy["Close"], d) for d in RS_WINDOWS}

sector_stats = {}

for etf, sector_name in SECTOR_ETFS.items():
    df = sector_data.get(etf)
    if df is None or len(df) < 220:
        continue

    close = df["Close"]
    volume = df["Volume"]
    rsi = calculate_rsi(close)
    macd_line, signal_line = calculate_macd(close)

    returns = {f"{d}D": pct_return(close, d) for d in RS_WINDOWS}

    rs = {}
    for d in RS_WINDOWS:
        key = f"{d}D"
        sector_return = returns.get(key, np.nan)
        spy_return = spy_rs_returns.get(key, np.nan)
        rs[f"RS{d}"] = sector_return - spy_return

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rel_vol = safe_last(volume / volume.rolling(20).mean())
    trend = classify_trend(close)

    members = get_sector_members(constituents_df, sector_name, set(close_wide.columns))

    if members:
        sector_close = close_wide[members]
        sector_b20_series = pct_above_ma(sector_close, 20)
        sector_b50_series = pct_above_ma(sector_close, 50)
        sector_b200_series = pct_above_ma(sector_close, 200)
        sector_b20 = safe_last(sector_b20_series)
        sector_b50 = safe_last(sector_b50_series)
        sector_b200 = safe_last(sector_b200_series)

        sector_daily = sector_close.pct_change()
        sector_adv = int((sector_daily.iloc[-1] > 0).sum())
        sector_dec = int((sector_daily.iloc[-1] < 0).sum())
        denominator = max(1, sector_adv + sector_dec)
        sector_adv_pct = sector_adv / denominator * 100

        if len(sector_b20_series.dropna()) > 5:
            sector_5d_breadth = sector_b20_series.dropna().iloc[-1] - sector_b20_series.dropna().iloc[-6]
        else:
            sector_5d_breadth = np.nan
    else:
        sector_b20 = np.nan
        sector_b50 = np.nan
        sector_b200 = np.nan
        sector_adv_pct = np.nan
        sector_5d_breadth = np.nan

    rs_acceleration = rs.get("RS5", np.nan) - rs.get("RS20", np.nan)
    rs_acceleration_10 = rs.get("RS10", np.nan) - rs.get("RS20", np.nan)

    emergence_score = 0.0
    emergence_score += np.clip(rs.get("RS5", np.nan), -5, 5) * 0.8
    emergence_score += np.clip(rs.get("RS10", np.nan), -5, 5) * 0.6
    emergence_score += np.clip(rs.get("RS20", np.nan), -8, 8) * 0.35
    emergence_score += np.clip(rs_acceleration, -8, 8) * 0.7
    emergence_score += np.clip(rs_acceleration_10, -8, 8) * 0.5

    if not np.isnan(sector_b20):
        emergence_score += (sector_b20 - 50) / 10
        emergence_score += np.clip(sector_5d_breadth, -10, 10) / 3

    latest_sector_rsi = safe_last(rsi)
    latest_sector_macd = safe_last(macd_line)
    latest_sector_signal = safe_last(signal_line)

    emergence_score += 1 if latest_sector_rsi > 50 else -1
    emergence_score += 1 if latest_sector_macd > latest_sector_signal else -1
    if not np.isnan(rel_vol):
        emergence_score += 1 if rel_vol > 1 else -1

    # --- Per-sector regime score/label — same 9-part formula as the
    # market-wide MARKET REGIME SCORE above, just built from this sector
    # ETF's own price/MA/RSI/MACD and its own members' breadth, plus the
    # market-wide VIX component (VIX is a market-wide signal, so it applies
    # the same way to every sector). Powers the sector regime banner grid. ---
    sector_last_close = safe_last(close)
    sector_last_ma20 = safe_last(ma20)
    sector_last_ma50 = safe_last(ma50)
    sector_last_ma200 = safe_last(ma200)

    sector_regime_score = 0
    sector_regime_score += 1 if sector_last_close > sector_last_ma20 else -1
    sector_regime_score += 1 if sector_last_close > sector_last_ma50 else -1
    sector_regime_score += 1 if sector_last_close > sector_last_ma200 else -1
    sector_regime_score += 1 if sector_last_ma50 > sector_last_ma200 else -1
    sector_regime_score += 1 if latest_sector_macd > latest_sector_signal else -1
    sector_regime_score += 1 if latest_sector_rsi > 50 else -1
    sector_regime_score += 1 if (not np.isnan(sector_b20) and sector_b20 > 50) else -1
    sector_regime_score += 1 if (not np.isnan(sector_b50) and sector_b50 > 50) else -1
    sector_regime_score += 1 if (not np.isnan(sector_adv_pct) and sector_adv_pct > 50) else -1
    if not np.isnan(last_vix):
        if last_vix < 20:
            sector_regime_score += 1
        elif last_vix > 30:
            sector_regime_score -= 1

    if sector_regime_score >= 7:
        sector_regime_label = "BULLISH / CONSTRUCTIVE"
    elif sector_regime_score <= -4:
        sector_regime_label = "BEARISH / CAUTIONARY"
    else:
        sector_regime_label = "NEUTRAL / MIXED"

    if (
        rs.get("RS20", np.nan) > 0
        and rs.get("RS10", np.nan) > 0
        and rs.get("RS5", np.nan) > 0
        and rs_acceleration > 0
    ):
        emergence_label = "EMERGING"
    elif rs.get("RS20", np.nan) > 0 and rs.get("RS5", np.nan) > 0:
        emergence_label = "LEADING"
    elif rs_acceleration > 0 and rs.get("RS5", np.nan) > 0:
        emergence_label = "IMPROVING"
    elif rs.get("RS20", np.nan) < 0 and rs_acceleration < 0:
        emergence_label = "LAGGING"
    elif rs_acceleration < 0:
        emergence_label = "WEAKENING"
    else:
        emergence_label = "NEUTRAL"

    sector_stats[etf] = {
        "sector_name": sector_name,
        "returns": returns,
        "rs": rs,
        "rsi": latest_sector_rsi,
        "macd_bull": latest_sector_macd > latest_sector_signal,
        "rel_vol": rel_vol,
        "trend": trend,
        "b20": sector_b20,
        "b50": sector_b50,
        "b200": sector_b200,
        "adv_pct": sector_adv_pct,
        "breadth_accel": sector_5d_breadth,
        "rs_accel": rs_acceleration,
        "rs_accel_10": rs_acceleration_10,
        "emergence_score": emergence_score,
        "emergence": emergence_label,
        "last_close": sector_last_close,
        "above_ma20": sector_last_close > sector_last_ma20 if not np.isnan(sector_last_ma20) else False,
        "above_ma50": sector_last_close > sector_last_ma50 if not np.isnan(sector_last_ma50) else False,
        "above_ma200": sector_last_close > sector_last_ma200 if not np.isnan(sector_last_ma200) else False,
        "regime_score": sector_regime_score,
        "regime_label": sector_regime_label,
    }

# ============================================================
# WEEKLY SECTOR HEATMAP
# ============================================================

weekly_returns = pd.DataFrame()
for etf in SECTOR_ETFS:
    df = sector_data.get(etf)
    if df is None:
        continue
    weekly_close = df["Close"].resample("W").last()
    weekly_returns[etf] = weekly_close.pct_change() * 100
weekly_returns = weekly_returns.dropna(how="all").tail(HEATMAP_LOOKBACK_WEEKS)

# ============================================================
# SECTOR CORRELATION MATRIX
# Are sectors moving together (risk-on/risk-off, one macro driver) or
# diverging (genuine rotation, idiosyncratic drivers)? Built from daily
# returns over a shorter recent window than the RS/breadth metrics above,
# since correlation regimes can shift faster than trend/momentum readings.
# ============================================================

SECTOR_CORR_LOOKBACK_DAYS = 60

sector_daily_returns = pd.DataFrame()
for etf in SECTOR_ETFS:
    df = sector_data.get(etf)
    if df is None:
        continue
    sector_daily_returns[etf] = df["Close"].pct_change()
sector_daily_returns = sector_daily_returns.tail(SECTOR_CORR_LOOKBACK_DAYS)
sector_corr = sector_daily_returns.corr()

# ============================================================
# RRG CALCULATIONS
# ============================================================


def build_rrg_paths(mode="weekly"):
    """
    mode:
      'daily'   — short-term RRG, last SHORT_RRG_DAYS trading days
      'weekly'  — intermediate RRG, last RRG_WEEKS weeks
      'monthly' — longer-term RRG, last MONTHLY_RRG_MONTHS months
    Same RS-Ratio / RS-Momentum math throughout (10-period ratio smoothing,
    4-period momentum smoothing) — only the bar interval and tail length change.
    """
    paths = {}

    if mode == "daily":
        spy_base = spy["Close"].tail(SHORT_RRG_DAYS + 30)
        for etf in SECTOR_ETFS:
            df = sector_data.get(etf)
            if df is None:
                continue
            sec = df["Close"].reindex(spy_base.index).dropna()
            aligned = pd.concat([sec, spy_base], axis=1, join="inner").dropna()
            if len(aligned) < 15:
                continue
            aligned.columns = ["sector", "spy"]
            rs_line = aligned["sector"] / aligned["spy"]
            rs_ratio = 100 * rs_line / rs_line.rolling(10).mean()
            rs_momentum = 100 * rs_ratio / rs_ratio.rolling(4).mean()
            combined = pd.concat(
                [rs_ratio.rename("rs_ratio"), rs_momentum.rename("rs_momentum")], axis=1
            ).dropna()
            if len(combined) >= 2:
                paths[etf] = combined.tail(SHORT_RRG_DAYS)
        return paths

    # weekly / monthly share the same resample-then-smooth pattern
    resample_rule = "W" if mode == "weekly" else "ME"
    tail_len = RRG_WEEKS if mode == "weekly" else MONTHLY_RRG_MONTHS
    min_bars_required = 15 if mode == "weekly" else max(15, tail_len + 10)

    spy_resampled = spy["Close"].resample(resample_rule).last()
    for etf in SECTOR_ETFS:
        df = sector_data.get(etf)
        if df is None:
            continue
        sec_resampled = df["Close"].resample(resample_rule).last()
        aligned = pd.concat([sec_resampled, spy_resampled], axis=1, join="inner").dropna()
        if len(aligned) < min_bars_required:
            continue
        aligned.columns = ["sector", "spy"]
        rs_line = aligned["sector"] / aligned["spy"]
        rs_ratio = 100 * rs_line / rs_line.rolling(10).mean()
        rs_momentum = 100 * rs_ratio / rs_ratio.rolling(4).mean()
        combined = pd.concat(
            [rs_ratio.rename("rs_ratio"), rs_momentum.rename("rs_momentum")], axis=1
        ).dropna()
        if len(combined) >= 2:
            paths[etf] = combined.tail(tail_len)

    return paths


rrg_weekly = build_rrg_paths(mode="weekly")
rrg_short = build_rrg_paths(mode="daily")
rrg_monthly = build_rrg_paths(mode="monthly")

# ============================================================
# CROSS-ASSET SUMMARY
# ============================================================

cross_rows = []
for name, df in cross_asset.items():
    if df is None or df.empty:
        continue
    latest = safe_last(df["Close"])
    r5 = pct_return(df["Close"], 5)
    r20 = pct_return(df["Close"], 20)
    cross_rows.append([name, latest, r5, r20])

cross_df = pd.DataFrame(cross_rows, columns=["Asset", "Last", "5D %", "20D %"])

# ============================================================
# DECISION SUMMARY DASHBOARD (standalone figure)
# ============================================================
# This is kept as its own standalone figure, saved and generated BEFORE the
# big GridSpec dashboard below, for the same reason the sector-strength
# charts further down are also kept standalone: the main dashboard's
# GridSpec row math is precisely tuned to N_ROWS, so adding rows to it would
# require re-deriving every row index that follows. A separate figure avoids
# that risk entirely.
#
# The goal here is a fast, top-of-page answer to "what's the market doing
# and should I be leaning bullish, cautious, or defensive right now" —
# organized into the same kind of topic sections as the rest of the
# dashboard, each with a green/yellow/red read, plus a per-sector status bar
# chart (green = leading, yellow = neutral, red = lagging) in the same
# bars-per-sector style as the RS charts elsewhere in this script.
# ============================================================

STATUS_BG = {"green": "#c8e6c9", "yellow": "#fff9c4", "red": "#ffcdd2"}
STATUS_FG = {"green": "#1a7a1a", "yellow": "#8a6d00", "red": "#b02020"}
STATUS_WORD = {"green": "GOOD", "yellow": "CAUTION", "red": "NEGATIVE"}


def draw_status_card(ax, title, status, headline, detail_lines):
    """Draws one colored status card: title, GOOD/CAUTION/NEGATIVE word,
    a bold headline stat, and a few smaller detail lines underneath."""
    ax.axis("off")
    ax.set_facecolor(STATUS_BG.get(status, "#f0f0f0"))
    fg = STATUS_FG.get(status, "#333333")
    ax.text(0.05, 0.88, title, fontsize=11, fontweight="bold", color="#555555",
            transform=ax.transAxes, va="top")
    ax.text(0.05, 0.70, STATUS_WORD.get(status, status.upper()), fontsize=15,
            fontweight="bold", color=fg, transform=ax.transAxes, va="top")
    ax.text(0.05, 0.52, headline, fontsize=9.5, fontweight="bold", color="#222222",
            transform=ax.transAxes, va="top")
    for i, line in enumerate(detail_lines):
        ax.text(0.05, 0.36 - i * 0.16, line, fontsize=8.7, color="#333333",
                transform=ax.transAxes, va="top")


# --- classify each topic into green / yellow / red ---

trend_above_all = last_close > last_ma20 > last_ma50 > last_ma200
trend_below_all = last_close < last_ma20 < last_ma50 < last_ma200
trend_status = "green" if (trend_above_all and ma_slopes["50MA slope 10D"] > 0) else (
    "red" if (trend_below_all and ma_slopes["50MA slope 10D"] < 0) else "yellow"
)

macd_bull_spy = last_macd > last_signal
if last_rsi > 55 and macd_bull_spy:
    momentum_status = "green"
elif last_rsi < 45 and not macd_bull_spy:
    momentum_status = "red"
else:
    momentum_status = "yellow"

vix_status = "red" if (vix_zone == "High" or vol_structure == "Stress / Backwardation") else (
    "green" if (vix_zone in ("Low", "Normal") and vol_structure == "Normal / Contango") else "yellow"
)

breadth_status = "green" if (latest_breadth[20] > 55 and latest_breadth[50] > 55) else (
    "red" if (latest_breadth[20] < 45 and latest_breadth[50] < 45) else "yellow"
)

internals_status = "green" if (latest_trin < 1.0 and latest_mcclellan > 0 and latest_highs > latest_lows) else (
    "red" if (latest_trin > 1.2 and latest_mcclellan < 0 and latest_lows > latest_highs) else "yellow"
)

volume_status = "green" if latest_upvol_pct > 55 else ("red" if latest_upvol_pct < 45 else "yellow")

macro_status = "red" if yield_curve_status == "Inverted (recession signal)" else (
    "green" if yield_curve_status == "Normal" else "yellow"
)

sector_emergence_counts = {"green": 0, "yellow": 0, "red": 0}
for _etf, _stats in sector_stats.items():
    if _stats["emergence"] in ("EMERGING", "LEADING"):
        sector_emergence_counts["green"] += 1
    elif _stats["emergence"] in ("LAGGING", "WEAKENING"):
        sector_emergence_counts["red"] += 1
    else:
        sector_emergence_counts["yellow"] += 1
rotation_status = "green" if sector_emergence_counts["green"] > sector_emergence_counts["red"] + 2 else (
    "red" if sector_emergence_counts["red"] > sector_emergence_counts["green"] + 2 else "yellow"
)

ranked_for_summary = sorted(sector_stats.items(), key=lambda kv: kv[1]["emergence_score"], reverse=True)

overall_status = "green" if regime_score >= 7 else ("red" if regime_score <= -4 else "yellow")

# --- build the figure ---

fig_ds = plt.figure(figsize=(16, 15))
gs_ds = gridspec.GridSpec(
    4, 4,
    height_ratios=[1.3, 2.6, 2.6, 6.0],
    hspace=0.55, wspace=0.18,
    figure=fig_ds, top=0.97, bottom=0.03, left=0.04, right=0.98,
)

# Row 0 — overall regime banner, full width
ax_overall = fig_ds.add_subplot(gs_ds[0, :])
ax_overall.axis("off")
ax_overall.set_facecolor(STATUS_BG[overall_status])
ax_overall.text(0.015, 0.72, "OVERALL MARKET STATUS", fontsize=13, fontweight="bold",
                 color="#666666", transform=ax_overall.transAxes, va="center")
ax_overall.text(0.015, 0.30,
                 f"{regime_label}  ({STATUS_WORD[overall_status]})   |   Score {regime_score:+d} / 10",
                 fontsize=22, fontweight="bold", color=STATUS_FG[overall_status],
                 transform=ax_overall.transAxes, va="center")
ax_overall.text(0.72, 0.5,
                 f"SPY ${last_close:,.2f}\n5D {market_returns['5D']:+.1f}%  |  20D {market_returns['20D']:+.1f}%",
                 fontsize=11, color="#222222", transform=ax_overall.transAxes, va="center", ha="left")

# Row 1 — Trend / Momentum / Volatility / Breadth
ax_trend = fig_ds.add_subplot(gs_ds[1, 0])
draw_status_card(
    ax_trend, "TREND", trend_status,
    f"SPY ${last_close:,.2f}  |  20D {market_returns['20D']:+.1f}%",
    [
        f"Above 20/50/200MA: {'Y' if last_close > last_ma20 else 'N'}/"
        f"{'Y' if last_close > last_ma50 else 'N'}/{'Y' if last_close > last_ma200 else 'N'}",
        f"50MA slope (10D): {ma_slopes['50MA slope 10D']:+.1f}%",
        f"Dist. from 52W high: {distance_from_ath:+.1f}%",
    ],
)

ax_mom = fig_ds.add_subplot(gs_ds[1, 1])
draw_status_card(
    ax_mom, "MOMENTUM", momentum_status,
    f"RSI {last_rsi:.1f}  |  MACD {'Bullish' if macd_bull_spy else 'Bearish'}",
    [
        "Overbought (>70)" if last_rsi > 70 else ("Oversold (<30)" if last_rsi < 30 else "Neutral RSI zone"),
        f"1D {market_returns['1D']:+.1f}%  |  10D {market_returns['10D']:+.1f}%",
    ],
)

ax_vix = fig_ds.add_subplot(gs_ds[1, 2])
draw_status_card(
    ax_vix, "VOLATILITY", vix_status,
    f"VIX {last_vix:.1f}  ({vix_zone})",
    [
        f"5D {vix_5d_change:+.1f}%  |  20D {vix_20d_change:+.1f}%",
        f"Term structure: {vol_structure}",
    ],
)

ax_breadth = fig_ds.add_subplot(gs_ds[1, 3])
draw_status_card(
    ax_breadth, "BREADTH", breadth_status,
    f">20MA {latest_breadth[20]:.0f}%  |  >50MA {latest_breadth[50]:.0f}%",
    [
        f">200MA {latest_breadth[200]:.0f}%",
        f"5D change: {breadth_change_5d[20]:+.1f} pts",
    ],
)

# Row 2 — Internal Strength / Volume Participation / Macro Risk / Sector Rotation
ax_internals = fig_ds.add_subplot(gs_ds[2, 0])
draw_status_card(
    ax_internals, "INTERNAL STRENGTH", internals_status,
    f"TRIN {latest_trin:.2f}  |  McClellan {latest_mcclellan:+.0f}",
    [
        f"New Highs/Lows: {latest_highs} / {latest_lows}",
        f"Advancers/Decliners: {latest_adv} / {latest_dec}",
    ],
)

ax_volp = fig_ds.add_subplot(gs_ds[2, 1])
draw_status_card(
    ax_volp, "VOLUME PARTICIPATION", volume_status,
    f"Up-Volume Share: {latest_upvol_pct:.0f}%",
    [
        f"Up/Down Vol Ratio: {latest_ud_ratio:.2f}",
        f"SPY Relative Volume: {spy_rvol:.2f}x",
    ],
)

ax_macro = fig_ds.add_subplot(gs_ds[2, 2])
yc_line = (
    f"10Y-3M Spread: {yield_curve_spread:+.2f} pts" if not np.isnan(yield_curve_spread) else "10Y-3M Spread: N/A"
)
pc_line = f"Put/Call: {last_putcall:.2f}" if not np.isnan(last_putcall) else "Put/Call: N/A"
draw_status_card(
    ax_macro, "MACRO / RISK", macro_status,
    yield_curve_status,
    [yc_line, pc_line],
)

ax_rot = fig_ds.add_subplot(gs_ds[2, 3])
draw_status_card(
    ax_rot, "SECTOR ROTATION", rotation_status,
    f"{sector_emergence_counts['green']} Leading / {sector_emergence_counts['yellow']} Neutral / "
    f"{sector_emergence_counts['red']} Lagging",
    [
        f"Top: {ranked_for_summary[0][1]['sector_name']}" if ranked_for_summary else "",
        f"Weakest: {ranked_for_summary[-1][1]['sector_name']}" if ranked_for_summary else "",
    ],
)

# Row 3 — sector-by-sector status bars, colored green/yellow/red by emergence
# label rather than plain sign, full width — same bars-per-sector layout as
# the RS charts elsewhere in this script.
ax_secbar = fig_ds.add_subplot(gs_ds[3, :])
sec_labels = [f"{etf} · {stats['sector_name']}" for etf, stats in ranked_for_summary]
sec_values = [stats["emergence_score"] for _, stats in ranked_for_summary]
sec_colors = []
for _etf, stats in ranked_for_summary:
    if stats["emergence"] in ("EMERGING", "LEADING"):
        sec_colors.append(STATUS_FG["green"])
    elif stats["emergence"] in ("LAGGING", "WEAKENING"):
        sec_colors.append(STATUS_FG["red"])
    else:
        sec_colors.append("#c9a600")

y_pos = np.arange(len(sec_labels))
ax_secbar.barh(y_pos, sec_values, color=sec_colors)
ax_secbar.axvline(0, color="black", linewidth=0.8)
ax_secbar.set_yticks(y_pos)
ax_secbar.set_yticklabels(sec_labels, fontsize=10)
ax_secbar.invert_yaxis()
ax_secbar.set_xlabel("Emergence Score (higher = stronger short-term rotation into the sector)")
ax_secbar.set_title(
    "Sector Status — Green = Leading, Yellow = Neutral, Red = Lagging", fontweight="bold", fontsize=13
)
ax_secbar.grid(axis="x", alpha=0.3)
for i, (etf, stats) in enumerate(ranked_for_summary):
    val = stats["emergence_score"]
    ax_secbar.text(
        val + (0.15 if val >= 0 else -0.15),
        i,
        stats["emergence"],
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=8,
        fontweight="bold",
        color=sec_colors[i],
    )

fig_ds.text(
    0.01, 0.005,
    "Green = supportive for a bullish/long-biased decision. Yellow = mixed, wait for confirmation. "
    "Red = caution/defensive. Not investment advice.",
    fontsize=8.5, color="#777777",
)

save_current_figure(fig_ds, "00_decision_summary.png", "Market Decision Summary")

# ============================================================
# DASHBOARD FIGURE
# ============================================================
#
# Row map — ONE merged, centered vertical column, top to bottom:
#   0            banner
#   1-4          MONTHLY block (SPY): Price+Volume, MACD, RSI+Stochastic, MFI+CMF
#   5-8          WEEKLY block (SPY): Price+Volume, MACD, RSI+Stochastic, MFI+CMF
#   9-23         DAILY block (15 panels, merged into one continuous group,
#                shared x-axis throughout):
#                  9-12   the 4 SPY daily panels (Price+Volume, MACD,
#                         RSI+Stochastic, MFI+CMF) — same data/indicators
#                         used for the Monthly/Weekly blocks above, just on
#                         daily bars
#                  13-23  the original 13-chart time-aligned stack, minus 2
#                         removed duplicates (SPY RSI(14) — duplicate of the
#                         Daily block's RSI+Stochastic panel — and Up/Down
#                         Volume Ratio — duplicate of Up-Volume Share):
#                           13 SPY Price & Trend
#                           14 VIX
#                           15 Market Breadth — % Above MA
#                           16 A/D Line
#                           17 52-Wk New Highs vs New Lows
#                           18 Up-Volume Share
#                           19 Growth vs Value (IWF/IWD)
#                           20 Small vs Large Cap (IWM/SPY)
#                           21 Arms Index (TRIN)
#                           22 McClellan Summation Index
#                           23 McClellan Oscillator
#   24           Momentum bars / Risk context / Breadth momentum (categorical, not time)
#   25           Sector emergence scorecard (table)
#   26           Sector breadth bars / RS 5D / RS 20D (categorical, by sector)
#   27           Weekly heatmap (weekly, not daily) / Sector correlation matrix
#   28-30        RRG (daily / weekly / monthly) — axes are RS-Ratio vs RS-Momentum, not time
#   31           Cross-asset table / Market concentration
#   32           Leadership table
# ============================================================

N_MONTHLY_ROWS = 4
N_WEEKLY_ROWS = 4
N_DAILY_NEW_ROWS = 4    # the 4 daily panels that used to be the side column
N_DAILY_ORIG_ROWS = 11  # the original time-aligned stack, minus 2 removed duplicates
                         # (SPY RSI(14) — duplicate of the Daily block's RSI+Stochastic panel —
                         # and Up/Down Volume Ratio — duplicate of Up-Volume Share)
N_DAILY_ROWS = N_DAILY_NEW_ROWS + N_DAILY_ORIG_ROWS  # 15
N_STACK_TOTAL = N_MONTHLY_ROWS + N_WEEKLY_ROWS + N_DAILY_ROWS  # 23

N_ROWS = 1 + N_STACK_TOTAL + 9 + 1  # 1 banner + 23 stacked charts + 9 panel rows + 1 sector regime banner row = 34

# Single column now (no more side-by-side split) — narrower figure width so
# the one continuous vertical line of charts reads as centered rather than
# stretched full-bleed edge to edge. Width trimmed to 70% of the prior 20in.
fig = plt.figure(figsize=(14, 148))
gs = gridspec.GridSpec(
    N_ROWS,
    4,
    height_ratios=(
        [1.1]  # 0 banner
        + [1.35, 1.35, 1.35, 2.0]  # 1-4 Monthly (bottom panel taller — shows date labels)
        + [1.35, 1.35, 1.35, 2.0]  # 5-8 Weekly (bottom panel taller — shows date labels)
        + [1.35] * (N_DAILY_ROWS - 1) + [2.0]  # 9-23 Daily merged (15 rows; only the very last is taller)
        + [2.6, 3.2, 2.6, 4.2, 10.0, 10.0, 10.0, 12.0, 4.2, 10.5]  # 24-33 everything else (33 = new sector regime banner grid, 6 rows tall)
    ),
    hspace=0.32,
    wspace=0.3,
    figure=fig,
    top=0.99,
    bottom=0.01,
    left=0.06,
    right=0.96,  # inset margins on both sides so the single column sits centered
)
# NOTE: "4" columns is only used by the panels below (rows 24-32) that
# explicitly split into sub-columns via GridSpecFromSubplotSpec; every
# time-series chart in rows 0-23 uses the full row width (gs[row, :]).

# ============================================================
# ROW 0 — REGIME BANNER (dashboard title folded in here, at the true top)
# ============================================================

ax = fig.add_subplot(gs[0, :])
ax.axis("off")
ax.text(
    0.01,
    0.86,
    "S&P 500 PROFESSIONAL MARKET & SECTOR ROTATION DASHBOARD",
    fontsize=13,
    fontweight="bold",
    color="#888888",
    transform=ax.transAxes,
    va="center",
)
ax.text(
    0.01,
    0.52,
    f"MARKET REGIME: {regime_label}  |  Score {regime_score:+d}",
    fontsize=21,
    fontweight="bold",
    transform=ax.transAxes,
    va="center",
)
ax.text(
    0.01,
    0.14,
    f"SPY ${last_close:,.2f} | 20D {market_returns['20D']:+.1f}% | "
    f"Above 20/50/200MA: {'Y' if last_close > last_ma20 else 'N'}/"
    f"{'Y' if last_close > last_ma50 else 'N'}/{'Y' if last_close > last_ma200 else 'N'} | "
    f"RSI {last_rsi:.1f} | VIX {last_vix:.1f} | {latest_breadth[20]:.0f}% >20MA",
    fontsize=11.5,
    transform=ax.transAxes,
    va="center",
)

# ============================================================
# ROWS 1-25 — MONTHLY -> WEEKLY -> DAILY (merged), single centered column
# ============================================================

SPY_PANEL_PARAMS = {
    "Daily": {"period": "1y", "interval": "1d"},
    "Weekly": {"period": "6y", "interval": "1wk"},
    "Monthly": {"period": "7y", "interval": "1mo"},
}
SPY_PANEL_BAR_WIDTH_MAP = {"1d": 0.6, "1wk": 5, "1mo": 20}


def fetch_spy_panel_data(period, interval):
    df = clean_download(MARKET_TICKER, period, interval=interval)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["MACD"], df["Signal"] = calculate_macd(df["Close"])
    df["RSI"] = calculate_rsi(df["Close"])
    df = calculate_stochastic(df)
    df["MFI"] = calculate_mfi(df)
    df["CMF"] = calculate_cmf(df)
    return df.dropna()


# Monthly and Weekly only need their own SPY fetch; Daily reuses this too,
# for the 4 new daily panels merged at the top of the Daily section below.
spy_panel_data = {
    tf: fetch_spy_panel_data(SPY_PANEL_PARAMS[tf]["period"], SPY_PANEL_PARAMS[tf]["interval"])
    for tf in ("Monthly", "Weekly", "Daily")
}

spy_date_fmt = mdates.DateFormatter("%b\n%Y")


def plot_spy_4panel_block(df_tf, tf, row_start, shared_ref=None, apply_label_hiding=True):
    """Plots Price+Volume / MACD / RSI+Stochastic / MFI+CMF for one SPY
    timeframe into 4 consecutive full-width rows starting at row_start.
    If shared_ref is given, the block's x-axis links to it instead of
    starting its own group (used to merge the Daily block's 4 new panels
    with the 13 original daily panels below them into one shared x-axis).
    apply_label_hiding=False skips hiding non-bottom tick labels internally,
    for blocks that will be merged into a larger group handled externally.
    Returns (block_axes, first_axis)."""
    interval = SPY_PANEL_PARAMS[tf]["interval"]

    if df_tf is None or df_tf.empty:
        ax_missing = fig.add_subplot(gs[row_start : row_start + 4, :])
        ax_missing.axis("off")
        ax_missing.text(
            0.5, 0.5, f"SPY {tf} data unavailable", ha="center", va="center", transform=ax_missing.transAxes
        )
        return [], None

    block_axes = []

    # --- Price & Volume ---
    ax_p = fig.add_subplot(gs[row_start, :], sharex=shared_ref)
    ax_p.plot(df_tf.index, df_tf["Close"], color="black", linewidth=1.2)
    ax_p.set_ylabel("Price", fontsize=8)
    ax_p.grid(alpha=0.3)
    ax_p2 = ax_p.twinx()
    x_num = np.asarray(mdates.date2num(df_tf.index.to_pydatetime())).flatten()
    y_vol = np.asarray((df_tf["Volume"] / 1_000_000).fillna(0)).flatten()
    closes_np = df_tf["Close"].astype(float).values
    opens_np = df_tf["Open"].astype(float).values
    vol_colors = [
        "green" if (not np.isnan(closes_np[i]) and closes_np[i] > opens_np[i]) else "red"
        for i in range(len(df_tf))
    ]
    ax_p2.bar(x_num, y_vol, width=SPY_PANEL_BAR_WIDTH_MAP.get(interval, 0.5), color=vol_colors, alpha=0.25)
    ax_p2.set_yticks([])
    ax_p.set_title(f"SPY {tf} Price & Volume", fontweight="bold", loc="left")
    block_axes.append(ax_p)

    ref = shared_ref if shared_ref is not None else ax_p

    # --- MACD ---
    ax_m = fig.add_subplot(gs[row_start + 1, :], sharex=ref)
    ax_m.plot(df_tf.index, df_tf["MACD"], color="blue", linewidth=1.1, label="MACD")
    ax_m.plot(df_tf.index, df_tf["Signal"], color="red", linewidth=1.1, label="Signal")
    hist = df_tf["MACD"] - df_tf["Signal"]
    hist_diff = hist.diff()
    # Bar color = momentum direction, not sign: green when this bar is HIGHER
    # than the previous bar (histogram expanding/rising), red when it's LOWER
    # (histogram contracting/falling) — applies the same way whether the
    # histogram itself is above or below zero. The first bar has no prior bar
    # to compare against, so it falls back to its own sign.
    bar_colors = []
    for i, d in enumerate(hist_diff):
        if pd.isna(d):
            bar_colors.append("green" if hist.iloc[i] >= 0 else "red")
        else:
            bar_colors.append("green" if d >= 0 else "red")
    bar_width = pd.Timedelta(days=SPY_PANEL_BAR_WIDTH_MAP.get(interval, 0.6))
    ax_m.bar(df_tf.index, hist, width=bar_width, color=bar_colors, alpha=0.5)
    ax_m.set_title(f"SPY {tf} MACD (bar color = rising vs falling momentum)", loc="left")
    ax_m.legend(fontsize=7, loc="upper left")
    ax_m.grid(alpha=0.3)
    block_axes.append(ax_m)

    # --- RSI + Stochastic ---
    ax_o = fig.add_subplot(gs[row_start + 2, :], sharex=ref)
    ax_o.plot(df_tf.index, df_tf["RSI"], color="teal", linewidth=1.3, label="RSI")
    ax_o.plot(df_tf.index, df_tf["%K"], color="purple", linewidth=1.0, label="Stoch %K")
    ax_o.axhline(70, color="red", linestyle="--", alpha=0.4)
    ax_o.axhline(30, color="green", linestyle="--", alpha=0.4)
    ax_o.set_ylim(0, 100)
    ax_o.set_title(f"SPY {tf} RSI + Stochastic", loc="left")
    ax_o.legend(fontsize=7, loc="upper left")
    ax_o.grid(alpha=0.3)
    block_axes.append(ax_o)

    # --- MFI + CMF ---
    ax_f = fig.add_subplot(gs[row_start + 3, :], sharex=ref)
    ax_f.plot(df_tf.index, df_tf["MFI"], color="darkviolet", linewidth=1.1, label="MFI")
    ax_f.set_ylim(0, 100)
    ax_f.set_ylabel("MFI", color="darkviolet", fontsize=7.5)
    ax_f.tick_params(axis="y", labelcolor="darkviolet", labelsize=7)
    ax_f2 = ax_f.twinx()
    ax_f2.plot(df_tf.index, df_tf["CMF"], color="teal", linewidth=1.1, label="CMF")
    ax_f2.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax_f2.set_ylim(-1, 1)
    ax_f2.set_ylabel("CMF", color="teal", fontsize=7.5)
    ax_f2.tick_params(axis="y", labelcolor="teal", labelsize=7)
    ax_f.set_title(f"SPY {tf} MFI + CMF", loc="left")
    ax_f.grid(alpha=0.3)
    block_axes.append(ax_f)

    for a in block_axes:
        a.xaxis.set_major_formatter(spy_date_fmt)

    if apply_label_hiding:
        for a in block_axes[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(block_axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=7)

    return block_axes, ax_p


# --- MONTHLY block: rows 1-4 (own x-axis group) ---
plot_spy_4panel_block(spy_panel_data.get("Monthly"), "Monthly", row_start=1)

# --- WEEKLY block: rows 5-8 (own x-axis group) ---
plot_spy_4panel_block(spy_panel_data.get("Weekly"), "Weekly", row_start=5)

# --- DAILY block: rows 9-25 (17 panels, ALL sharing one x-axis group) ---
# First the 4 new SPY daily panels (rows 9-12) establish the shared x-axis
# reference; the original 13-chart stack (rows 13-25) then links to it, so
# the whole merged daily section pans/zooms together.
daily_new_axes, daily_ref_ax = plot_spy_4panel_block(
    spy_panel_data.get("Daily"), "Daily", row_start=9, apply_label_hiding=False
)
daily_df_for_range = spy_panel_data.get("Daily")

stack_axes = []  # the 13 original daily-specific charts (rows 13-25), x-linked to daily_ref_ax


def add_stack_ax(row):
    """Full-width axis in the merged Daily group, x-linked to the shared Daily reference."""
    sharex = daily_ref_ax if daily_ref_ax is not None else (stack_axes[0] if stack_axes else None)
    ax_new = fig.add_subplot(gs[row, :], sharex=sharex)
    stack_axes.append(ax_new)
    return ax_new


# --- 13. SPY Price & Trend ---
ax_price = add_stack_ax(13)
plot = spy.tail(TIMESERIES_LOOKBACK_DAYS)
ax_price.plot(plot.index, plot["Close"], color="black", label="SPY")
ax_price.plot(plot.index, plot["MA20"], label="20D MA")
ax_price.plot(plot.index, plot["MA50"], label="50D MA")
ax_price.plot(plot.index, plot["MA200"], label="200D MA")
ax_price.set_title("S&P 500 / SPY — Price & Trend", fontweight="bold", loc="left")
ax_price.legend(fontsize=8, ncol=4, loc="upper left")
ax_price.grid(alpha=0.35)

# --- 14. VIX ---
ax_vix = add_stack_ax(14)
if vix is not None:
    vp = vix.tail(TIMESERIES_LOOKBACK_DAYS)
    ax_vix.plot(vp.index, vp["Close"], linewidth=1.4)
    ax_vix.axhline(20, linestyle="--", alpha=0.6)
    ax_vix.axhline(30, linestyle="--", alpha=0.6)
    ax_vix.set_title(
        f"VIX {last_vix:.1f} | {vix_zone} | 5D {vix_5d_change:+.1f}% | 20D {vix_20d_change:+.1f}%",
        fontweight="bold",
        loc="left",
    )
    ax_vix.grid(alpha=0.35)

# --- 15. Market Breadth — % Above MA ---
ax_b20 = add_stack_ax(15)
for window in [20, 50, 200]:
    ax_b20.plot(breadth[window].index, breadth[window].values, label=f">{window}D MA")
ax_b20.axhline(50, linestyle=":", alpha=0.6)
ax_b20.set_ylim(0, 100)
ax_b20.set_title("Market Breadth — % Above MA", fontweight="bold", loc="left")
ax_b20.legend(fontsize=7.5, ncol=3, loc="upper left")
ax_b20.grid(alpha=0.3)

# --- 16. A/D Line ---
ax_ad = add_stack_ax(16)
ax_ad.plot(ad_line.index, ad_line.values, linewidth=1.5)
ax_ad.axhline(0, linewidth=0.7, alpha=0.5)
ax_ad.set_title(f"A/D Line | Adv {latest_adv} / Dec {latest_dec}", fontweight="bold", loc="left")
ax_ad.grid(alpha=0.3)

# --- 17. New Highs vs New Lows ---
ax_hl = add_stack_ax(17)
ax_hl.plot(new_highs.index, new_highs.values, label="New Highs")
ax_hl.plot(new_lows.index, new_lows.values, label="New Lows")
ax_hl.set_title(f"52-Wk Highs vs Lows — {latest_highs}H / {latest_lows}L", fontweight="bold", loc="left")
ax_hl.legend(fontsize=7.5, ncol=2, loc="upper left")
ax_hl.grid(alpha=0.3)

# --- 18. Up-Volume Share ---
ax_vol = add_stack_ax(18)
vol_plot = up_volume_pct.tail(TIMESERIES_LOOKBACK_DAYS)
ax_vol.plot(vol_plot.index, vol_plot.values, linewidth=1.5)
ax_vol.axhline(50, linestyle=":", alpha=0.6)
ax_vol.axhline(70, linestyle="--", alpha=0.4)
ax_vol.set_ylim(0, 100)
ax_vol.set_title(f"Up-Volume Share — {latest_upvol_pct:.0f}%", fontweight="bold", loc="left")
ax_vol.grid(alpha=0.3)

# --- 19. Growth vs Value ---
ax_gv = add_stack_ax(19)
if growth_value_ratio is not None:
    gv_plot = growth_value_ratio.tail(TIMESERIES_LOOKBACK_DAYS)
    gv_ma = growth_value_ratio.rolling(20).mean().tail(TIMESERIES_LOOKBACK_DAYS)
    ax_gv.plot(gv_plot.index, gv_plot.values, linewidth=1.4, label="IWF/IWD")
    ax_gv.plot(gv_ma.index, gv_ma.values, linewidth=1.0, linestyle="--", alpha=0.7, label="20D MA")
    ax_gv.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax_gv.set_title("Growth vs Value (IWF/IWD) — rising = Growth leading", fontweight="bold", loc="left")
    ax_gv.grid(alpha=0.3)
else:
    ax_gv.text(0.5, 0.5, "Growth/Value data unavailable", ha="center", va="center", transform=ax_gv.transAxes)
    ax_gv.set_title("Growth vs Value", fontweight="bold", loc="left")

# --- 20. Small vs Large Cap ---
ax_sl = add_stack_ax(20)
if small_large_ratio is not None:
    sl_plot = small_large_ratio.tail(TIMESERIES_LOOKBACK_DAYS)
    sl_ma = small_large_ratio.rolling(20).mean().tail(TIMESERIES_LOOKBACK_DAYS)
    ax_sl.plot(sl_plot.index, sl_plot.values, linewidth=1.4, color="#8B4513", label="IWM/SPY")
    ax_sl.plot(
        sl_ma.index,
        sl_ma.values,
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        color="#8B4513",
        label="20D MA",
    )
    ax_sl.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax_sl.set_title(
        "Small vs Large Cap (IWM/SPY) — rising = Small Caps leading", fontweight="bold", loc="left"
    )
    ax_sl.grid(alpha=0.3)
else:
    ax_sl.text(0.5, 0.5, "Small/Large Cap data unavailable", ha="center", va="center", transform=ax_sl.transAxes)
    ax_sl.set_title("Small vs Large Cap", fontweight="bold", loc="left")

# --- 21. Arms Index (TRIN) ---
ax_trin = add_stack_ax(21)
ax_trin.plot(trin_plot.index, trin_plot.values, linewidth=1.4, color="#6A0DAD")
ax_trin.axhline(1, linestyle=":", alpha=0.6, color="black")
ax_trin.set_yscale("log")
ax_trin.set_title(f"Arms Index (TRIN) — {latest_trin:.2f}", fontweight="bold", loc="left")
ax_trin.grid(alpha=0.3)

# --- 22. McClellan Summation Index ---
ax_summ = add_stack_ax(22)
summ_color = (
    "#1a7a1a" if mcclellan_summation_plot.iloc[-1] >= mcclellan_summation_plot.iloc[0] else "#b02020"
)
ax_summ.plot(
    mcclellan_summation_plot.index, mcclellan_summation_plot.values, linewidth=1.6, color=summ_color
)
ax_summ.axhline(0, linewidth=0.7, alpha=0.5, color="black")
ax_summ.fill_between(
    mcclellan_summation_plot.index, mcclellan_summation_plot.values, 0, color=summ_color, alpha=0.15
)
ax_summ.set_title(
    "McClellan Summation Index (relative — see note in code)", fontweight="bold", loc="left"
)
ax_summ.grid(alpha=0.3)

# --- 23. McClellan Oscillator (bottom of the entire merged Daily section — keeps date labels) ---
ax_mcc = add_stack_ax(23)
mc_colors = ["#1a7a1a" if v >= 0 else "#b02020" for v in mcclellan_osc_plot.values]
ax_mcc.bar(mcclellan_osc_plot.index, mcclellan_osc_plot.values, color=mc_colors, width=1.0)
ax_mcc.axhline(0, linewidth=0.8, color="black")
ax_mcc.set_title(f"McClellan Oscillator — {latest_mcclellan:+.0f}", fontweight="bold", loc="left")
ax_mcc.grid(alpha=0.3)

# Lock the whole merged Daily group (4 new + 11 original = 15 panels) to the
# same x-range, put a tick on EVERY month (interval=1) instead of
# matplotlib's automatic locator skipping to every other month, hide tick
# LABELS on all but the very bottom chart of the group, and rotate those.
all_daily_axes = daily_new_axes + stack_axes
if daily_ref_ax is not None and daily_df_for_range is not None and not daily_df_for_range.empty:
    daily_ref_ax.set_xlim(daily_df_for_range.index.min(), daily_df_for_range.index.max())
for a in all_daily_axes:
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
for a in all_daily_axes[:-1]:
    plt.setp(a.get_xticklabels(), visible=False)
plt.setp(all_daily_axes[-1].get_xticklabels(), rotation=45, ha="right")

# ============================================================
# ROW 24 — MOMENTUM / RISK CONTEXT / BREADTH MOMENTUM (categorical, not time-axis)
# ============================================================

row14_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[24, :], wspace=0.28)

ax_returns = fig.add_subplot(row14_gs[0, 0])
days = [1, 5, 10, 20, 60]
vals = [market_returns[f"{d}D"] for d in days]
ax_returns.bar([f"{d}D" for d in days], vals)
ax_returns.axhline(0, linewidth=0.8)
ax_returns.set_title("SPY Price Momentum", fontweight="bold")
ax_returns.set_ylabel("Return %")
ax_returns.grid(axis="y", alpha=0.3)

ax_risk = fig.add_subplot(row14_gs[0, 1])
risk_lines = [
    f"Distance from 52W High: {distance_from_ath:+.1f}%",
    f"20MA slope: {ma_slopes['20MA slope 10D']:+.1f}%",
    f"50MA slope: {ma_slopes['50MA slope 10D']:+.1f}%",
    f"200MA slope: {ma_slopes['200MA slope 20D']:+.1f}%",
    f"SPY Relative Volume: {spy_rvol:.2f}x",
    f"VIX structure: {vol_structure}",
    (
        f"10Y-3M Yield Curve: {yield_curve_spread:+.2f} pts ({yield_curve_status})"
        if not np.isnan(yield_curve_spread)
        else "10Y-3M Yield Curve: N/A"
    ),
    (f"Put/Call Ratio: {last_putcall:.2f}" if not np.isnan(last_putcall) else "Put/Call Ratio: N/A"),
]
ax_risk.axis("off")
ax_risk.set_title("Trend / Risk Context", fontweight="bold")
for i, line in enumerate(risk_lines):
    ax_risk.text(0.02, 0.90 - i * 0.115, line, fontsize=9)

ax_bchange = fig.add_subplot(row14_gs[0, 2])
labels = ["20D", "50D", "200D"]
five = [breadth_change_5d[w] for w in BREADTH_MA_WINDOWS]
twenty = [breadth_change_20d[w] for w in BREADTH_MA_WINDOWS]
x = np.arange(len(labels))
ax_bchange.bar(x - 0.18, five, width=0.36, label="5D change")
ax_bchange.bar(x + 0.18, twenty, width=0.36, label="20D change")
ax_bchange.axhline(0, linewidth=0.8)
ax_bchange.set_xticks(x)
ax_bchange.set_xticklabels(labels)
ax_bchange.set_title("Breadth Momentum", fontweight="bold")
ax_bchange.set_ylabel("Pts change")
ax_bchange.legend(fontsize=7.5)
ax_bchange.grid(axis="y", alpha=0.3)

# ============================================================
# ROW 15 — SECTOR EMERGENCE SCORECARD (full width)
# ============================================================

ax_table = fig.add_subplot(gs[25, :])
ax_table.axis("off")
ranked = sorted(sector_stats.items(), key=lambda kv: kv[1]["emergence_score"], reverse=True)

rows = []
for etf, stats in ranked:
    rows.append(
        [
            etf,
            stats["sector_name"],
            f"{stats['rs']['RS5']:+.1f}%",
            f"{stats['rs']['RS10']:+.1f}%",
            f"{stats['rs']['RS20']:+.1f}%",
            f"{stats['rs']['RS60']:+.1f}%",
            f"{stats['b20']:.0f}%" if not np.isnan(stats["b20"]) else "N/A",
            f"{stats['breadth_accel']:+.1f}",
            f"{stats['rel_vol']:.2f}x",
            f"{stats['emergence_score']:+.1f}",
            stats["emergence"],
        ]
    )

columns = [
    "ETF",
    "Sector",
    "RS 5D",
    "RS 10D",
    "RS 20D",
    "RS 60D",
    "% >20MA",
    "Breadth Δ5D",
    "Rel Vol",
    "Emergence",
    "Status",
]
table = ax_table.table(cellText=rows, colLabels=columns, cellLoc="center", bbox=[0, 0, 1, 0.88])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
ax_table.set_title(
    "SECTOR EMERGENCE SCORECARD — Short-Term Rotation Is Prioritized",
    fontsize=14,
    fontweight="bold",
    pad=22,
)

# ============================================================
# ROW 16 — SECTOR BREADTH (wide) / RS 5D / RS 20D (categorical, by sector)
# ============================================================

row16_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[26, :], wspace=0.3)

ax_sb = fig.add_subplot(row16_gs[0, 0:2])
sector_names = [stats["sector_name"] for _, stats in ranked]
b20_values = [stats["b20"] for _, stats in ranked]
b50_values = [stats["b50"] for _, stats in ranked]
b200_values = [stats["b200"] for _, stats in ranked]
xw = np.arange(len(sector_names))
width = 0.25
ax_sb.bar(xw - width, b20_values, width, label="% >20MA")
ax_sb.bar(xw, b50_values, width, label="% >50MA")
ax_sb.bar(xw + width, b200_values, width, label="% >200MA")
ax_sb.axhline(50, linestyle=":", alpha=0.6)
ax_sb.set_ylim(0, 100)
ax_sb.set_xticks(xw)
ax_sb.set_xticklabels(sector_names, rotation=45, ha="right", fontsize=8)
ax_sb.set_ylabel("% of constituents")
ax_sb.set_title("Sector Breadth — Broad or Concentrated?", fontweight="bold")
ax_sb.legend(fontsize=7.5)
ax_sb.grid(axis="y", alpha=0.3)

labels = [etf for etf, _ in ranked]

ax_rs5 = fig.add_subplot(row16_gs[0, 2])
rs5_values = [stats["rs"]["RS5"] for _, stats in ranked]
ax_rs5.barh(range(len(labels)), rs5_values)
ax_rs5.axvline(0, linewidth=0.8)
ax_rs5.set_yticks(range(len(labels)))
ax_rs5.set_yticklabels(labels, fontsize=8)
ax_rs5.invert_yaxis()
ax_rs5.set_title("5D Sector RS vs SPY", fontweight="bold")
ax_rs5.set_xlabel("Excess return %")
ax_rs5.grid(axis="x", alpha=0.3)

ax_rs20 = fig.add_subplot(row16_gs[0, 3])
rs20_values = [stats["rs"]["RS20"] for _, stats in ranked]
ax_rs20.barh(range(len(labels)), rs20_values)
ax_rs20.axvline(0, linewidth=0.8)
ax_rs20.set_yticks(range(len(labels)))
ax_rs20.set_yticklabels(labels, fontsize=8)
ax_rs20.invert_yaxis()
ax_rs20.set_title("20D Sector RS vs SPY", fontweight="bold")
ax_rs20.set_xlabel("Excess return %")
ax_rs20.grid(axis="x", alpha=0.3)

# ============================================================
# ROW 17 — WEEKLY SECTOR HEATMAP (left) | SECTOR CORRELATION MATRIX (right)
# (weekly buckets, and a correlation matrix — neither is a daily-date x-axis,
# so these stay outside the aligned stack)
# ============================================================

row17_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[27, :], width_ratios=[2.1, 1.2], wspace=0.22
)

ax_heat = fig.add_subplot(row17_gs[0, 0])
recent_score = {
    etf: sector_stats[etf]["rs"]["RS5"] + sector_stats[etf]["rs"]["RS10"] for etf in sector_stats
}
heat_order = sorted(weekly_returns.columns, key=lambda e: recent_score.get(e, -999), reverse=True)
heat_data = weekly_returns[heat_order].tail(13).T.values

if heat_data.size:
    max_abs = np.nanmax(np.abs(heat_data))
else:
    max_abs = 1
if not np.isfinite(max_abs) or max_abs == 0:
    max_abs = 1

image = ax_heat.imshow(heat_data, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
ax_heat.set_yticks(range(len(heat_order)))
ax_heat.set_yticklabels([SECTOR_ETFS[e] for e in heat_order], fontsize=8)
heat_dates = weekly_returns.tail(13).index
ax_heat.set_xticks(range(len(heat_dates)))
ax_heat.set_xticklabels([d.strftime("%m/%d") for d in heat_dates], rotation=45, ha="right", fontsize=8)
ax_heat.set_title("Sector Weekly Return Heatmap", fontweight="bold")

for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        value = heat_data[i, j]
        if np.isfinite(value):
            ax_heat.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=7)

ax_corr = fig.add_subplot(row17_gs[0, 1])
corr_order = [SECTOR_ETFS[e] for e in sector_corr.columns]
corr_matrix = sector_corr.values
im_corr = ax_corr.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax_corr.set_xticks(range(len(corr_order)))
ax_corr.set_xticklabels(corr_order, rotation=45, ha="right", fontsize=8)
ax_corr.set_yticks(range(len(corr_order)))
ax_corr.set_yticklabels(corr_order, fontsize=8)
ax_corr.set_title(
    f"Sector Correlation Matrix (Last {SECTOR_CORR_LOOKBACK_DAYS}D Daily Returns)",
    fontweight="bold",
    fontsize=11,
)
for i in range(len(corr_order)):
    for j in range(len(corr_order)):
        ax_corr.text(
            j,
            i,
            f"{corr_matrix[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
        )
fig.colorbar(im_corr, ax=ax_corr, fraction=0.04, pad=0.02)

# ============================================================
# ROWS 18-20 — RRG, ONE FULL-WIDTH ROW EACH (daily / weekly / monthly)
# NOT part of the time-aligned stack: the RRG x-axis is RS-Ratio, not a date.
# ============================================================


def plot_rrg(ax, paths, title):
    if not paths:
        ax.text(0.5, 0.5, "RRG data unavailable", ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return

    all_x = pd.concat([path["rs_ratio"] for path in paths.values()])
    all_y = pd.concat([path["rs_momentum"] for path in paths.values()])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    x_pad = max(x_range * 0.15, 0.5)
    y_pad = max(y_range * 0.15, 0.5)
    xmin = min(99, all_x.min() - x_pad)
    xmax = max(101, all_x.max() + x_pad)
    ymin = min(99, all_y.min() - y_pad)
    ymax = max(101, all_y.max() + y_pad)

    ax.axhline(100, linewidth=0.9, color="gray")
    ax.axvline(100, linewidth=0.9, color="gray")

    for etf, path in paths.items():
        sector_name = SECTOR_ETFS[etf]
        color = SECTOR_COLORS.get(sector_name, "#333333")
        xs = path["rs_ratio"].values
        ys = path["rs_momentum"].values
        n = len(xs)

        if n >= 2:
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.18, 1.0, n - 1)
            widths = np.linspace(1.0, 3.0, n - 1)
            rgba = [(*plt.matplotlib.colors.to_rgb(color), a) for a in alphas]
            lc = LineCollection(segments, colors=rgba, linewidths=widths)
            ax.add_collection(lc)
            ax.scatter(xs[:-1], ys[:-1], color=color, s=12, alpha=0.25, zorder=3, linewidth=0)

        ax.scatter([xs[-1]], [ys[-1]], color=color, s=130, edgecolor="black", linewidth=1.0, zorder=5)
        ax.annotate(
            etf,
            (xs[-1], ys[-1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("RS-Ratio", fontsize=10)
    ax.set_ylabel("RS-Momentum", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(alpha=0.25)
    ax.text(
        0.98,
        0.96,
        "LEADING",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1a7a1a",
        alpha=0.8,
    )
    ax.text(
        0.02,
        0.96,
        "IMPROVING",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1565c0",
        alpha=0.8,
    )
    ax.text(
        0.02,
        0.04,
        "LAGGING",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#b02020",
        alpha=0.8,
    )
    ax.text(
        0.98,
        0.04,
        "WEAKENING",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#b8860b",
        alpha=0.8,
    )


def plot_rrg_legend(ax, paths):
    """Small side panel listing ETF -> sector -> color. With up to 11
    overlapping trails on one chart, color alone isn't enough to tell them
    apart, so this panel sits next to each RRG as a lookup key."""
    ax.axis("off")
    if not paths:
        return
    items = sorted(paths.keys(), key=lambda e: SECTOR_ETFS[e])
    n = len(items)
    top, bottom = 0.94, 0.06
    step = (top - bottom) / max(n - 1, 1)
    for i, etf in enumerate(items):
        y = top - i * step if n > 1 else (top + bottom) / 2
        color = SECTOR_COLORS.get(SECTOR_ETFS[etf], "#333333")
        ax.plot([0.03, 0.16], [y, y], color=color, linewidth=3.5, transform=ax.transAxes)
        ax.text(
            0.21, y, f"{etf} — {SECTOR_ETFS[etf]}", transform=ax.transAxes, fontsize=9.5, va="center"
        )


rrg_short_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[28, :], width_ratios=[5, 1], wspace=0.03
)
ax_rrg_short = fig.add_subplot(rrg_short_gs[0, 0])
plot_rrg(ax_rrg_short, rrg_short, "Short-Term RRG — Recent Rotation (Daily)")
ax_rrg_short_legend = fig.add_subplot(rrg_short_gs[0, 1])
plot_rrg_legend(ax_rrg_short_legend, rrg_short)

rrg_weekly_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[29, :], width_ratios=[5, 1], wspace=0.03
)
ax_rrg_weekly = fig.add_subplot(rrg_weekly_gs[0, 0])
plot_rrg(ax_rrg_weekly, rrg_weekly, "Weekly RRG — Intermediate Sector Rotation")
ax_rrg_weekly_legend = fig.add_subplot(rrg_weekly_gs[0, 1])
plot_rrg_legend(ax_rrg_weekly_legend, rrg_weekly)

rrg_monthly_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[30, :], width_ratios=[5, 1], wspace=0.03
)
ax_rrg_monthly = fig.add_subplot(rrg_monthly_gs[0, 0])
plot_rrg(ax_rrg_monthly, rrg_monthly, f"Monthly RRG — Last {MONTHLY_RRG_MONTHS} Months")
ax_rrg_monthly_legend = fig.add_subplot(rrg_monthly_gs[0, 1])
plot_rrg_legend(ax_rrg_monthly_legend, rrg_monthly)

# ============================================================
# ROW 31 — CROSS-ASSET DAILY TRENDS / MARKET CONCENTRATION
# Replaces the old "current values" table with actual daily trend lines for
# each cross-asset series (same TIMESERIES_LOOKBACK_DAYS window as the rest
# of the dashboard), each with a faint SPY overlay on a secondary y-axis —
# the point isn't the cross-asset's own level, it's whether it's tracking
# WITH or diverging AGAINST SPY, which is what actually signals money
# rotating into or out of the S&P 500 (e.g. SPY up while DXY/yields also up
# reads differently than SPY up while they fall).
# ============================================================

stock_returns = {}
for ticker in close_wide.columns:
    series = close_wide[ticker].dropna()
    if len(series) > 20:
        stock_returns[ticker] = {"5D": pct_return(series, 5), "20D": pct_return(series, 20)}

stock_df = pd.DataFrame(stock_returns).T.dropna()
top10 = stock_df.sort_values("20D", ascending=False).head(10)
bottom10 = stock_df.sort_values("20D", ascending=True).head(10)
constituent_sector_map = constituents_df.set_index("Ticker")["Sector"].to_dict()

row21_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[31, :], width_ratios=[3.4, 1], wspace=0.2)

# --- Left: 8 cross-asset daily trend charts, stacked vertically (one per
# row) instead of a cramped 2x4 grid — full width per chart, shared x-axis,
# date labels only on the bottom chart. ---
cross_chart_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=row21_gs[0, 0], hspace=0.5)

CROSS_ASSET_ORDER = ["10Y", "3M T-Bill", "DXY", "Oil", "Gold", "HYG", "LQD", "PutCall"]
CROSS_ASSET_VALUE_FMT = {
    "10Y": "{:.2f}",
    "3M T-Bill": "{:.2f}",
    "PutCall": "{:.2f}",
}
spy_overlay = spy["Close"].tail(TIMESERIES_LOOKBACK_DAYS)

cross_asset_axes = []

for idx, name in enumerate(CROSS_ASSET_ORDER):
    sharex = cross_asset_axes[0] if cross_asset_axes else None
    ax_ca = fig.add_subplot(cross_chart_gs[idx, 0], sharex=sharex)
    df_ca = cross_asset.get(name)

    if df_ca is None or df_ca.empty:
        ax_ca.axis("off")
        ax_ca.set_title(f"{name} — unavailable", fontsize=9, loc="left")
        continue

    ca_plot = df_ca["Close"].tail(TIMESERIES_LOOKBACK_DAYS)
    r5 = pct_return(df_ca["Close"], 5)
    r20 = pct_return(df_ca["Close"], 20)
    latest = safe_last(df_ca["Close"])
    fmt = CROSS_ASSET_VALUE_FMT.get(name, "{:,.2f}")
    latest_text = fmt.format(latest) if not np.isnan(latest) else "N/A"

    ax_ca.plot(ca_plot.index, ca_plot.values, color="#1f4e8c", linewidth=1.3)
    ax_ca.set_title(
        f"{name} — {latest_text} | 5D {r5:+.1f}% | 20D {r20:+.1f}%", fontsize=9, loc="left"
    )
    ax_ca.tick_params(axis="y", labelsize=7, labelcolor="#1f4e8c")
    ax_ca.grid(alpha=0.25)
    ax_ca.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_ca.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    ax_spy_overlay = ax_ca.twinx()
    ax_spy_overlay.plot(spy_overlay.index, spy_overlay.values, color="gray", linewidth=0.9, alpha=0.5, linestyle="--")
    ax_spy_overlay.set_yticks([])

    cross_asset_axes.append(ax_ca)

for a in cross_asset_axes[:-1]:
    plt.setp(a.get_xticklabels(), visible=False)
if cross_asset_axes:
    plt.setp(cross_asset_axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=7)

if cross_asset_axes:
    cross_asset_axes[0].set_xlim(spy_overlay.index.min(), spy_overlay.index.max())

print("Cross-asset data availability (rows fetched, most-recent date):")
for name in CROSS_ASSET_ORDER:
    df_ca = cross_asset.get(name)
    if df_ca is None or df_ca.empty:
        print(f"  {name:10s} — no data returned")
    else:
        print(f"  {name:10s} — {len(df_ca)} rows, last date {df_ca.index.max().date()}")

ax_conc = fig.add_subplot(row21_gs[0, 1])
ax_conc.axis("off")
ax_conc.set_title("Market Concentration / Breadth", fontweight="bold", fontsize=13, pad=18)

if not stock_df.empty:
    positive_20 = (stock_df["20D"] > 0).mean() * 100
    positive_5 = (stock_df["5D"] > 0).mean() * 100
    top10_avg = top10["20D"].mean()
    bottom10_avg = bottom10["20D"].mean()
else:
    positive_20 = np.nan
    positive_5 = np.nan
    top10_avg = np.nan
    bottom10_avg = np.nan

concentration_lines = [
    f"% stocks positive 5D: {positive_5:.0f}%",
    f"% stocks positive 20D: {positive_20:.0f}%",
    f"Top 10 avg 20D return: {top10_avg:+.1f}%",
    f"Bottom 10 avg 20D return: {bottom10_avg:+.1f}%",
    f"New highs / lows: {latest_highs} / {latest_lows}",
    f"Up-volume share: {latest_upvol_pct:.0f}%",
]
for i, line in enumerate(concentration_lines):
    ax_conc.text(0.03, 0.82 - i * 0.14, line, fontsize=10)

ax_conc.text(
    0.03,
    0.02,
    "Cross-asset trends are confirmation, not a standalone buy/sell signal.\n"
    "HYG/LQD are credit-risk proxies; DXY, yields, oil and gold give macro context.",
    transform=ax_conc.transAxes,
    fontsize=7.5,
)

# ============================================================
# ROW 22 — MARKET LEADERSHIP (full width — 20 rows, too tall to pair)
# ============================================================

ax_leaders = fig.add_subplot(gs[32, :])
ax_leaders.axis("off")
ax_leaders.set_title("S&P 500 Short-Term Leaders / Laggards", fontweight="bold", fontsize=14, pad=22)

leader_rows = []
for ticker, row in top10.iterrows():
    leader_rows.append(
        [
            ticker,
            constituent_sector_map.get(ticker, "Unknown"),
            f"{row['5D']:+.1f}%",
            f"{row['20D']:+.1f}%",
        ]
    )
for ticker, row in bottom10.iterrows():
    leader_rows.append(
        [
            ticker,
            constituent_sector_map.get(ticker, "Unknown"),
            f"{row['5D']:+.1f}%",
            f"{row['20D']:+.1f}%",
        ]
    )

leader_table = ax_leaders.table(
    cellText=leader_rows,
    colLabels=["Ticker", "Sector", "5D", "20D"],
    cellLoc="center",
    bbox=[0.15, 0, 0.7, 0.88],
)
leader_table.auto_set_font_size(False)
leader_table.set_fontsize(8.5)

# ============================================================
# ROW 33 — SECTOR REGIME BANNERS (all 11 sectors, one mini-banner each)
# Same "SECTOR REGIME: LABEL | Score +N / TICKER $price | 20D ... | RSI ..."
# format as the single-sector drill-down dashboard's top banner, just
# repeated once per sector ETF in a grid so all 11 are visible at once.
# ============================================================

REGIME_BANNER_BG = {
    "BULLISH / CONSTRUCTIVE": "#c8e6c9",
    "NEUTRAL / MIXED": "#f0f0f0",
    "BEARISH / CAUTIONARY": "#ffcdd2",
}
REGIME_BANNER_TEXT = {
    "BULLISH / CONSTRUCTIVE": "#1a7a1a",
    "NEUTRAL / MIXED": "#555555",
    "BEARISH / CAUTIONARY": "#b02020",
}

banner_order = [
    etf for etf, _ in sorted(sector_stats.items(), key=lambda kv: kv[1]["regime_score"], reverse=True)
]

banner_gs = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs[33, :], wspace=0.06, hspace=0.45)

for i, etf in enumerate(banner_order):
    stats = sector_stats[etf]
    row_i, col_i = divmod(i, 2)
    ax_b = fig.add_subplot(banner_gs[row_i, col_i])
    ax_b.axis("off")

    label = stats["regime_label"]
    bg = REGIME_BANNER_BG.get(label, "#f0f0f0")
    fg = REGIME_BANNER_TEXT.get(label, "#333333")
    ax_b.set_facecolor(bg)

    ma_flags = (
        f"{'Y' if stats['above_ma20'] else 'N'}/"
        f"{'Y' if stats['above_ma50'] else 'N'}/"
        f"{'Y' if stats['above_ma200'] else 'N'}"
    )

    ax_b.text(
        0.03, 0.80,
        f"{stats['sector_name'].upper()} ({etf})",
        fontsize=10.5, fontweight="bold", color="#888888", transform=ax_b.transAxes, va="center",
    )
    ax_b.text(
        0.03, 0.50,
        f"REGIME: {label}  |  Score {stats['regime_score']:+d}",
        fontsize=14, fontweight="bold", color=fg, transform=ax_b.transAxes, va="center",
    )
    ax_b.text(
        0.03, 0.20,
        f"{etf} ${stats['last_close']:,.2f} | 20D {stats['returns'].get('20D', float('nan')):+.1f}% | "
        f"MA20/50/200: {ma_flags} | RSI {stats['rsi']:.1f} | VIX {last_vix:.1f} | {stats['b20']:.0f}% >20MA",
        fontsize=10, color="#222222", transform=ax_b.transAxes, va="center",
    )

for j in range(len(banner_order), 12):
    row_i, col_i = divmod(j, 2)
    ax_blank = fig.add_subplot(banner_gs[row_i, col_i])
    ax_blank.axis("off")

# ============================================================
# FOOTER
# ============================================================

fig.text(
    0.01,
    0.001,
    "Data: Yahoo Finance + current S&P 500 constituent list | "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "Technical/market-regime framework only — not investment advice.",
    fontsize=8,
    color="#777777",
)

save_current_figure(fig, "01_dashboard_main.png", "Market + Sector Rotation Dashboard")

# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MARKET SUMMARY")
print("=" * 70)
print(f"Regime: {regime_label} (score {regime_score:+d})")
print(f"SPY: ${last_close:,.2f}")
print(
    f"SPY 5D / 10D / 20D: {market_returns['5D']:+.2f}% / {market_returns['10D']:+.2f}% / {market_returns['20D']:+.2f}%"
)
print(f"Breadth >20MA: {latest_breadth[20]:.1f}% (5D {breadth_change_5d[20]:+.1f} pts)")
print(f"Advancers / Decliners: {latest_adv} / {latest_dec}")
print(f"Up-volume share: {latest_upvol_pct:.1f}%")
print(f"VIX: {last_vix:.2f} ({vix_zone})")
print(f"VIX structure: {vol_structure}")
print("\nTOP SECTOR EMERGENCE:")
print("-" * 70)
for etf, stats in ranked[:5]:
    print(
        f"{etf:<5} {stats['sector_name']:<28} "
        f"RS5 {stats['rs']['RS5']:+.1f}% | RS20 {stats['rs']['RS20']:+.1f}% | {stats['emergence']}"
    )
print("=" * 70)

# ============================================================
# SECTOR STRENGTH CHARTS — YEARLY / QUARTERLY / MONTHLY / WEEKLY / DAILY
# ============================================================
# Five standalone figures (kept separate from the main GridSpec dashboard
# above since that figure's row math is precisely tuned to N_ROWS — adding
# rows there would require re-deriving every row index below it).
#
# All five reuse SECTOR_ETFS / SECTOR_COLORS / clean_download / MARKET_TICKER
# from the main script above so styling stays consistent. One extended-history
# download (11y) covers all five charts — the shorter-lookback charts (weekly,
# daily) just slice a shorter tail off the same underlying daily data.
#
# Visual only — no console table printed for any of these.
#
# Bar labels (all five charts): horizontal, bold, unsigned (magnitude only —
# the bar's position above/below the zero line already conveys direction).
# The SPY benchmark line keeps its own signed labels, unchanged.
# ============================================================

PERIOD_CHART_DOWNLOAD_YEARS = "11y"  # buffer past the longest (10y yearly) lookback

print("\nDownloading extended history for sector strength charts...")

period_sector_data = {}
for etf in SECTOR_ETFS:
    df = clean_download(etf, PERIOD_CHART_DOWNLOAD_YEARS)
    if df is not None:
        period_sector_data[etf] = df

period_spy = clean_download(MARKET_TICKER, PERIOD_CHART_DOWNLOAD_YEARS)

today_ts = pd.Timestamp.now()


def compute_period_returns(close_series, rule):
    """% return per period (indexed by period-end timestamp), using the
    first and last available close within each period. Used for W/M/Q/Y."""
    s = close_series.dropna()
    grouped = s.resample(rule).agg(["first", "last"]).dropna()
    return (grouped["last"] / grouped["first"] - 1) * 100


def compute_daily_returns(close_series):
    """Simple close-to-close % return per trading day. Used for the daily chart —
    resample('D') on already-daily data would trivially return 0% per bar,
    since first==last within a single-row period."""
    s = close_series.dropna()
    return s.pct_change().dropna() * 100


def plot_sector_strength_chart(rule, lookback_offset, label_func, is_current_partial,
                                title, figsize, filename, daily_mode=False):
    """
    rule:               pandas resample rule ('Y', 'Q', 'M', 'W'); ignored if daily_mode=True
    lookback_offset:     pd.DateOffset — how far back from today to show
    label_func:          fn(period_end_timestamp) -> x-axis label (without partial suffix)
    is_current_partial:  fn(period_end_timestamp) -> True if this period is still in progress
    daily_mode:          True for the daily chart (close-to-close returns, no resampling)
    """
    etf_list = list(SECTOR_ETFS.keys())
    cutoff = today_ts - lookback_offset

    all_returns = {}
    for etf in etf_list:
        df = period_sector_data.get(etf)
        if df is None:
            continue
        all_returns[etf] = compute_daily_returns(df["Close"]) if daily_mode else compute_period_returns(df["Close"], rule)

    if period_spy is not None:
        spy_returns = compute_daily_returns(period_spy["Close"]) if daily_mode else compute_period_returns(period_spy["Close"], rule)
    else:
        spy_returns = pd.Series(dtype=float)

    all_index = spy_returns.index
    for s in all_returns.values():
        all_index = all_index.union(s.index)
    periods = sorted(p for p in all_index if p >= cutoff)

    returns_table = pd.DataFrame(index=periods, columns=etf_list + [MARKET_TICKER], dtype=float)
    for etf in etf_list:
        if etf in all_returns:
            returns_table[etf] = all_returns[etf].reindex(periods)
    returns_table[MARKET_TICKER] = spy_returns.reindex(periods)

    labels = [
        (label_func(p) + f" ({PARTIAL_SUFFIX})") if is_current_partial(p) else label_func(p)
        for p in periods
    ]

    fig_p, ax_p = plt.subplots(figsize=figsize)

    n_sectors = len(etf_list)
    n_periods = len(periods)
    group_width = 0.82
    bar_width = group_width / n_sectors
    x = np.arange(n_periods)

    # Shrink bar-label font as bar count grows, so labels stay legible instead of overlapping
    bar_label_fontsize = 6.5 if n_periods * n_sectors > 80 else 7.5

    for i, etf in enumerate(etf_list):
        sector_name = SECTOR_ETFS[etf]
        color = SECTOR_COLORS.get(sector_name, "#333333")
        offsets = x - group_width / 2 + bar_width * i + bar_width / 2
        vals = returns_table[etf].values.astype(float)
        bars = ax_p.bar(offsets, vals, width=bar_width * 0.92, color=color,
                         label=f"{etf} · {sector_name}", edgecolor="white", linewidth=0.3, zorder=3)
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                ax_p.annotate(f"{abs(v):.0f}%", (rect.get_x() + rect.get_width() / 2, v),
                              textcoords="offset points", xytext=(0, 2 if v >= 0 else -10),
                              ha="center", va="bottom" if v >= 0 else "top",
                              fontsize=bar_label_fontsize, fontweight="bold", rotation=0, zorder=6)

    spy_vals = returns_table[MARKET_TICKER].values.astype(float)
    ax_p.plot(x, spy_vals, color="black", marker="D", markersize=8, linewidth=2.0,
              zorder=5, label=f"{MARKET_TICKER} · S&P 500 (benchmark)")
    for xi, yi in zip(x, spy_vals):
        if not np.isnan(yi):
            ax_p.annotate(f"{yi:+.0f}%", (xi, yi), textcoords="offset points",
                          xytext=(0, 14 if yi >= 0 else -20), ha="center",
                          fontsize=9, fontweight="bold", color="black", zorder=7)

    ax_p.axhline(0, color="#333333", linewidth=1, zorder=2)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(labels, fontsize=9 if n_periods <= 12 else 7.5,
                          rotation=0 if n_periods <= 12 else 45, ha="center" if n_periods <= 12 else "right")
    ax_p.set_ylabel("Total Return (%)", fontsize=12)
    ax_p.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax_p.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_p.set_axisbelow(True)
    ax_p.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False, borderaxespad=0)

    plt.tight_layout()
    save_current_figure(fig_p, filename, title.splitlines()[0])


# ------------------------------------------------------------
# CHART 1 — YEARLY, LAST 10 YEARS (incl. YTD)
# ------------------------------------------------------------
PARTIAL_SUFFIX = "YTD"

plot_sector_strength_chart(
    rule="YE",
    lookback_offset=pd.DateOffset(years=10),
    label_func=lambda p: str(p.year),
    is_current_partial=lambda p: p.year == today_ts.year,
    title="S&P 500 Sector Strength by Year — Last 10 Years\n(Sector SPDR ETF returns vs. S&P 500 benchmark)",
    figsize=(18, 9),
    filename="02_sector_yearly.png",
)

# ------------------------------------------------------------
# CHART 2 — QUARTERLY, LAST 3 YEARS (incl. QTD)
# ------------------------------------------------------------
PARTIAL_SUFFIX = "QTD"

plot_sector_strength_chart(
    rule="QE",
    lookback_offset=pd.DateOffset(years=3),
    label_func=lambda p: f"Q{p.quarter} {p.year}",
    is_current_partial=lambda p: (p.year == today_ts.year) and (p.quarter == today_ts.quarter),
    title="S&P 500 Sector Strength by Quarter — Last 3 Years\n(Sector SPDR ETF returns vs. S&P 500 benchmark)",
    figsize=(20, 9),
    filename="03_sector_quarterly.png",
)

# ------------------------------------------------------------
# CHART 3 — MONTHLY, LAST 12 MONTHS (incl. MTD)
# ------------------------------------------------------------
PARTIAL_SUFFIX = "MTD"

plot_sector_strength_chart(
    rule="ME",
    lookback_offset=pd.DateOffset(months=12),
    label_func=lambda p: p.strftime("%b %Y"),
    is_current_partial=lambda p: (p.year == today_ts.year) and (p.month == today_ts.month),
    title="S&P 500 Sector Strength by Month — Last 12 Months\n(Sector SPDR ETF returns vs. S&P 500 benchmark)",
    figsize=(20, 9),
    filename="04_sector_monthly.png",
)

# ------------------------------------------------------------
# CHART 4 — WEEKLY, LAST 2 MONTHS (incl. current partial week)
# ------------------------------------------------------------
PARTIAL_SUFFIX = "WTD"

plot_sector_strength_chart(
    rule="W",
    lookback_offset=pd.DateOffset(months=2),
    label_func=lambda p: p.strftime("%b %d"),
    is_current_partial=lambda p: pd.Period(p, freq="W") == pd.Period(today_ts, freq="W"),
    title="S&P 500 Sector Strength by Week — Last 2 Months\n(Sector SPDR ETF returns vs. S&P 500 benchmark)",
    figsize=(18, 9),
    filename="05_sector_weekly.png",
)

# ------------------------------------------------------------
# CHART 5 — DAILY, LAST 2 WEEKS
# ------------------------------------------------------------
PARTIAL_SUFFIX = "Today"

plot_sector_strength_chart(
    rule=None,
    lookback_offset=pd.DateOffset(weeks=2),
    label_func=lambda p: p.strftime("%a %b %d"),
    is_current_partial=lambda p: p.normalize() == today_ts.normalize(),
    title="S&P 500 Sector Strength by Day — Last 2 Weeks\n(Sector SPDR ETF returns vs. S&P 500 benchmark)",
    figsize=(18, 9),
    filename="06_sector_daily.png",
    daily_mode=True,
)


# ============================================================
# SECTION: TICKER SPOTLIGHT (single-ticker deep dive)
# ============================================================
# Ported from the standalone "Ticker Indicator Graph DWM" notebook: a
# Daily / Weekly / Monthly grid (rows) x Price+Volume / MACD / RSI+Stochastic
# / MFI+CMF (columns) for one ticker. Reuses this script's own
# calculate_rsi / calculate_macd / calculate_stochastic / calculate_mfi /
# calculate_cmf and SPY_PANEL_BAR_WIDTH_MAP, and saves to OUTPUT_DIR instead
# of plt.show() so it becomes its own card on the Pages site.
#
# MACD histogram bars use the same convention as the SPY panels above: green
# when a bar is higher than the PREVIOUS bar (rising momentum), red when
# lower (falling) — not simply green-above-zero / red-below-zero.
# ============================================================

TICKER_SPOTLIGHT = "TSLA"   # change this to switch which single ticker gets its own section
SPOTLIGHT_DISPLAY_TZ = "US/Central"
SPOTLIGHT_MARKET_TZ = "US/Eastern"

SPOTLIGHT_TIMEFRAMES = [
    ("Daily", "1y", "1d"),
    ("Weekly", "5y", "1wk"),
    ("Monthly", "6y", "1mo"),
]


def _download_ohlc_tz(ticker, period, interval, market_tz):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    if data.index.tz is None:
        data.index = data.index.tz_localize(market_tz)
    else:
        data.index = data.index.tz_convert(market_tz)
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
        data = data.between_time("09:30", "16:00")
    return data


def fetch_ticker_timeframe(ticker, period, interval, min_rows=2, market_tz=SPOTLIGHT_MARKET_TZ):
    data = _download_ohlc_tz(ticker, period, interval, market_tz)

    # A fixed lookback (e.g. '6y' monthly) can come back too thin for a ticker
    # with a shorter actual trading history (a recent IPO/spinoff) — retry with
    # 'max' rather than silently dropping the timeframe.
    if (data is None or len(data) < min_rows) and period != "max":
        fallback = _download_ohlc_tz(ticker, "max", interval, market_tz)
        if fallback is not None and len(fallback) >= (len(data) if data is not None else 0):
            data = fallback

    if data is None:
        return None

    macd_line, signal_line = calculate_macd(data["Close"])
    data["MACD"] = macd_line
    data["Signal"] = signal_line
    data["RSI"] = calculate_rsi(data["Close"], period=14)
    data = calculate_stochastic(data, k_period=14, d_period=3, smooth_k=3)
    data["MFI"] = calculate_mfi(data, period=14)
    data["CMF"] = calculate_cmf(data, period=20)

    # Only require a valid Close — dropping every row until ALL indicators are
    # warmed up (CMF needs 20 bars) can wipe out a short-history ticker entirely.
    data = data.dropna(subset=["Close"])
    if len(data) < min_rows:
        return None
    return data


def to_tz(df, display_tz):
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert(display_tz)
    return out


def plot_price_volume_panel(ax, df, interval, title, display_tz):
    df_plot = to_tz(df, display_tz)
    ax.plot(df_plot.index, df_plot["Close"], color="black")
    ax.set_ylabel("Price ($)")
    ax.grid(True)

    ax2 = ax.twinx()
    x = np.asarray(mdates.date2num(df_plot.index.to_pydatetime())).flatten()
    y = np.asarray((df_plot["Volume"] / 1_000_000).fillna(0)).flatten()
    bar_width = SPY_PANEL_BAR_WIDTH_MAP.get(interval, 0.5)

    closes = df_plot["Close"].astype(float).values
    opens = df_plot["Open"].astype(float).values
    colors = [
        "green" if (not np.isnan(closes[i]) and closes[i] > opens[i]) else "red"
        for i in range(len(df_plot))
    ]
    ax2.bar(x, y, width=bar_width, color=colors, alpha=0.3)
    ax2.set_ylabel("Volume (M)")

    max_vol = max(y) if len(y) > 0 else 1
    ax2.set_ylim(0, max_vol * 1.2)
    ax2.set_yticks(np.linspace(0, max_vol * 1.2, 6))
    ax2.set_yticklabels([f"{v:.1f}M" for v in ax2.get_yticks()])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y", tz=pytz.timezone(display_tz)))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_title(title, fontsize=10)


def plot_macd_panel(ax, df_plot, interval, title, display_tz):
    ax.plot(df_plot.index, df_plot["MACD"], label="MACD", color="blue", linewidth=1.2)
    ax.plot(df_plot.index, df_plot["Signal"], label="Signal", color="red", linewidth=1.2)
    hist = df_plot["MACD"] - df_plot["Signal"]
    bar_width = pd.Timedelta(days=SPY_PANEL_BAR_WIDTH_MAP.get(interval, 0.6))

    # Bar color = momentum direction vs. the PREVIOUS bar (rising = green,
    # falling = red) — same convention as the SPY MACD panels earlier in this
    # script, applied here for Daily/Weekly/Monthly alike.
    hist_vals = hist.values
    bar_colors = [
        "green" if (i == 0 or pd.isna(hist_vals[i - 1]) or hist_vals[i] >= hist_vals[i - 1]) else "red"
        for i in range(len(hist_vals))
    ]
    ax.bar(df_plot.index, hist, width=bar_width, color=bar_colors, alpha=0.5, label="Histogram")

    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y", tz=pytz.timezone(display_tz)))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True)
    ax.legend(fontsize=7)


def plot_osc_panel(ax, df_plot, title, display_tz):
    ax.plot(df_plot.index, df_plot["RSI"], label="RSI", color="teal", linewidth=1.4)
    ax.plot(df_plot.index, df_plot["%K"], label="Stoch %K", color="purple", linewidth=1.0)
    ax.plot(df_plot.index, df_plot["%D"], label="Stoch %D", color="gold", linewidth=1.0, alpha=0.15)

    ax.fill_between(df_plot.index, df_plot["RSI"], 50, where=(df_plot["RSI"] > 50),
                     interpolate=True, color="lightgreen", alpha=0.3)
    ax.fill_between(df_plot.index, df_plot["%K"], 60, where=(df_plot["%K"] > 60),
                     interpolate=True, color="violet", alpha=0.3)

    ax.axhline(70, color="red", linestyle="--", alpha=0.4)
    ax.axhline(30, color="green", linestyle="--", alpha=0.4)
    ax.axhline(80, color="red", linestyle=":", alpha=0.4)
    ax.axhline(20, color="green", linestyle=":", alpha=0.4)

    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y", tz=pytz.timezone(display_tz)))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)


def plot_mfi_cmf_panel(ax_mfi, df_plot, title, display_tz):
    ax_mfi.plot(df_plot.index, df_plot["MFI"], label="MFI (14)", color="darkviolet", linewidth=1.2)
    ax_mfi.axhline(80, color="red", linestyle="--", alpha=0.6)
    ax_mfi.axhline(20, color="green", linestyle="--", alpha=0.6)
    ax_mfi.axhline(50, color="gray", linestyle=":", alpha=0.4)
    ax_mfi.set_ylim(0, 100)
    ax_mfi.set_ylabel("MFI", color="darkviolet", fontsize=8)
    ax_mfi.tick_params(axis="y", labelcolor="darkviolet", labelsize=7)

    ax_cmf = ax_mfi.twinx()
    ax_cmf.plot(df_plot.index, df_plot["CMF"], label="CMF (20)", color="teal", linewidth=1.2)
    ax_cmf.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax_cmf.set_ylim(-1, 1)
    ax_cmf.set_ylabel("CMF", color="teal", fontsize=8)
    ax_cmf.tick_params(axis="y", labelcolor="teal", labelsize=7)

    mfi_norm = (df_plot["MFI"] - 50) / 50
    fill_condition = (mfi_norm > 0) & (df_plot["CMF"] > 0)
    ax_cmf.fill_between(df_plot.index, mfi_norm, df_plot["CMF"], where=fill_condition,
                         interpolate=True, color="limegreen", alpha=0.3)

    lines1, labels1 = ax_mfi.get_legend_handles_labels()
    lines2, labels2 = ax_cmf.get_legend_handles_labels()
    ax_mfi.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    ax_mfi.set_title(title, fontsize=10)
    ax_mfi.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y", tz=pytz.timezone(display_tz)))
    ax_mfi.tick_params(axis="x", rotation=45, labelsize=7)
    ax_mfi.grid(True, alpha=0.3)


def generate_ticker_spotlight(ticker):
    print(f"\nGenerating ticker spotlight for {ticker}...")
    datasets = []
    for label, period, interval in SPOTLIGHT_TIMEFRAMES:
        df = fetch_ticker_timeframe(ticker, period, interval)
        if df is None or df.empty:
            print(f"  Note: {label} timeframe has no usable data for {ticker} — skipping.")
            continue
        datasets.append((label, interval, df))

    if not datasets:
        print(f"  No data available for {ticker} — skipping ticker spotlight section.")
        return

    fig_t, axs_t = plt.subplots(len(datasets), 4, figsize=(30, 4.2 * len(datasets)), squeeze=False)

    for i, (label, interval, df) in enumerate(datasets):
        df_plot = to_tz(df, SPOTLIGHT_DISPLAY_TZ)
        plot_price_volume_panel(axs_t[i, 0], df, interval, f"{ticker} {label} Close & Volume", SPOTLIGHT_DISPLAY_TZ)
        plot_macd_panel(axs_t[i, 1], df_plot, interval, f"{label} MACD", SPOTLIGHT_DISPLAY_TZ)
        plot_osc_panel(axs_t[i, 2], df_plot, f"{label} RSI + Stochastic", SPOTLIGHT_DISPLAY_TZ)
        plot_mfi_cmf_panel(axs_t[i, 3], df_plot, f"{label} MFI + CMF", SPOTLIGHT_DISPLAY_TZ)

    fig_t.suptitle(f"{ticker} — Daily / Weekly / Monthly Indicator Spotlight", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_current_figure(fig_t, f"07_ticker_spotlight_{ticker.lower()}.png", f"{ticker} Ticker Spotlight")


generate_ticker_spotlight(TICKER_SPOTLIGHT)


# ============================================================
# SECTION: SECTOR INDICATOR GRID (all 11 sectors x Daily/Weekly/Monthly)
# ============================================================
# Ported from the standalone "Sectors_IND_GRA_DWM" notebook: all 11 sector
# SPDR ETFs as columns (sorted by latest weekly return, strongest first),
# stacked Daily / Weekly / Monthly blocks as rows (4 rows per block —
# Price+Volume, MACD, RSI+Stochastic, MFI+CMF). Reuses SECTOR_ETFS and the
# same panel-plotting helpers as the ticker spotlight above, and the same
# rising/falling MACD histogram coloring convention.
# ============================================================

SECTOR_GRID_TIMEFRAMES = [
    ("Daily", "1y", "1d"),
    ("Weekly", "6y", "1wk"),
    ("Monthly", "7y", "1mo"),
]
SECTOR_GRID_COL_WIDTH_INCHES = 8.0
SECTOR_GRID_ROW_HEIGHT_INCHES = 4.2


def fetch_sector_grid_timeframe(ticker, period, interval, market_tz=SPOTLIGHT_MARKET_TZ):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    if data.index.tz is None:
        data.index = data.index.tz_localize(market_tz)
    else:
        data.index = data.index.tz_convert(market_tz)

    macd_line, signal_line = calculate_macd(data["Close"])
    data["MACD"] = macd_line
    data["Signal"] = signal_line
    data["RSI"] = calculate_rsi(data["Close"], period=14)
    data = calculate_stochastic(data, k_period=14, d_period=3, smooth_k=3)
    data["MFI"] = calculate_mfi(data, period=14)
    data["CMF"] = calculate_cmf(data, period=20)
    return data.dropna()


def generate_sector_indicator_grid():
    print("\nGenerating sector indicator grid (Daily/Weekly/Monthly x all sectors)...")

    all_grid_data = {}  # {timeframe_label: {etf: df}}
    for label, period, interval in SECTOR_GRID_TIMEFRAMES:
        print(f"  Downloading {label} data ({period}, interval={interval}) for {len(SECTOR_ETFS)} sector ETFs...")
        tf_data = {}
        for etf in SECTOR_ETFS:
            df = fetch_sector_grid_timeframe(etf, period, interval)
            if df is not None and not df.empty:
                tf_data[etf] = df
            else:
                print(f"    Skipped {etf} ({label}) — no data returned")
        all_grid_data[label] = tf_data

    # Sort sectors by latest weekly return, largest to smallest — same order
    # applies to all three stacked timeframe blocks
    weekly_returns = {}
    for etf in SECTOR_ETFS:
        df_w = all_grid_data.get("Weekly", {}).get(etf)
        if df_w is not None and len(df_w) >= 2:
            weekly_returns[etf] = float((df_w["Close"].iloc[-1] / df_w["Close"].iloc[-2] - 1) * 100)
        else:
            weekly_returns[etf] = float("-inf")

    sector_order = sorted(SECTOR_ETFS.items(), key=lambda kv: weekly_returns[kv[0]], reverse=True)
    n_sectors = len(sector_order)
    n_tf = len(SECTOR_GRID_TIMEFRAMES)

    if n_sectors == 0:
        print("  No sector data available — skipping sector indicator grid.")
        return

    fig_s, axs_s = plt.subplots(
        4 * n_tf, n_sectors,
        figsize=(SECTOR_GRID_COL_WIDTH_INCHES * n_sectors, SECTOR_GRID_ROW_HEIGHT_INCHES * 4 * n_tf),
        squeeze=False,
    )

    for t, (label, period, interval) in enumerate(SECTOR_GRID_TIMEFRAMES):
        row_price = t * 4 + 0
        row_macd = t * 4 + 1
        row_osc = t * 4 + 2
        row_mfi = t * 4 + 3

        for j, (etf, sector_name) in enumerate(sector_order):
            df = all_grid_data[label].get(etf)
            if df is None:
                for r in (row_price, row_macd, row_osc, row_mfi):
                    axs_s[r, j].set_title(f"{etf} — no data ({label})", fontsize=10)
                    axs_s[r, j].axis("off")
                continue

            df_plot = to_tz(df, SPOTLIGHT_DISPLAY_TZ)
            wr = weekly_returns.get(etf, float("-inf"))
            wr_label = f" [Wk: {wr:+.2f}%]" if wr != float("-inf") else ""

            plot_price_volume_panel(
                axs_s[row_price, j], df, interval, f"{etf} — {sector_name} ({label}){wr_label}", SPOTLIGHT_DISPLAY_TZ
            )
            plot_macd_panel(axs_s[row_macd, j], df_plot, interval, f"{etf} MACD ({label})", SPOTLIGHT_DISPLAY_TZ)
            plot_osc_panel(axs_s[row_osc, j], df_plot, f"{etf} RSI + Stochastic ({label})", SPOTLIGHT_DISPLAY_TZ)
            plot_mfi_cmf_panel(axs_s[row_mfi, j], df_plot, f"{etf} MFI + CMF ({label})", SPOTLIGHT_DISPLAY_TZ)

    fig_s.suptitle("S&P 500 Sector Comparison — Daily / Weekly / Monthly", fontsize=16, fontweight="bold", y=1.002)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_current_figure(fig_s, "08_sector_indicator_grid.png", "Sector Indicator Grid (Daily/Weekly/Monthly)")


generate_sector_indicator_grid()


# ============================================================
# BUILD STATIC HTML PAGE FOR GITHUB PAGES
# ============================================================
# This writes OUTPUT_DIR/index.html referencing every chart saved above,
# in the order they were generated. The workflow deploys OUTPUT_DIR as
# the Pages site on every run, so this file (and the images beside it)
# are what's live at the Pages URL after each scheduled refresh.

_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_cards_html = "\n".join(
    f"""    <section class="card">
      <h2>{_title}</h2>
      <img src="{_filename}" alt="{_title}" loading="lazy">
    </section>"""
    for _filename, _title in GENERATED_IMAGES
)

# ============================================================
# SCANNER SECTIONS (appended below the dashboard charts)
# ============================================================
# Each scanner writes its own interactive report into
# OUTPUT_DIR/scanners/<name>/ together with a small scanner.json
# ({"title", "subtitle", "order", "page"}). Every folder found here is
# added at the bottom of the page, so new scanners need no edits to
# this file - just run them before this script in the workflow.
import glob as _glob
import json as _json

_scanner_sections = []
for _manifest in _glob.glob(os.path.join(OUTPUT_DIR, "scanners", "*", "scanner.json")):
    try:
        with open(_manifest) as _mf:
            _m = _json.load(_mf)
        _rel = os.path.relpath(os.path.dirname(_manifest), OUTPUT_DIR).replace(os.sep, "/")
        _scanner_sections.append((_m.get("order", 100), _m.get("title", _rel),
                                  _m.get("subtitle", ""), f"{_rel}/{_m.get('page', 'index.html')}"))
    except Exception as _e:
        print(f"  Skipping scanner manifest {_manifest}: {_e}")
_scanner_sections.sort(key=lambda x: (x[0], x[1]))

_scanners_html = ""
if _scanner_sections:
    _scanners_html = '  <h1 class="section-title">Scanners</h1>\n' + "\n".join(
        f"""    <section class="card scanner">
      <h2>{_title} <a class="open-link" href="{_src}" target="_blank" rel="noopener">Open full page &#8599;</a></h2>
      <div class="scanner-sub">{_sub}</div>
      <iframe class="scanner-frame" src="{_src}" title="{_title}" loading="lazy"></iframe>
    </section>"""
        for _order, _title, _sub, _src in _scanner_sections
    )
print(f"Found {len(_scanner_sections)} scanner section(s)")

_page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>S&amp;P 500 Market + Sector Rotation Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: #f5f6f8;
    margin: 0;
    padding: 24px;
    color: #222;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .card {{
    background: #fff;
    border: 1px solid #e2e2e2;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .card h2 {{ font-size: 16px; margin: 0 0 12px; }}
  img {{ max-width: 100%; height: auto; display: block; }}
  .section-title {{ margin: 40px 0 16px; padding-top: 16px; border-top: 2px solid #e2e2e2; }}
  .card h2 .open-link {{ font-size: 13px; font-weight: normal; margin-left: 10px; color: #2a78d6; }}
  .scanner-sub {{ color: #666; font-size: 13px; margin: -6px 0 10px; }}
  .scanner-frame {{ width: 100%; height: 1900px; border: 0; display: block; }}
</style>
</head>
<body>
  <h1>S&amp;P 500 Market + Sector Rotation Dashboard</h1>
  <div class="meta">
    Last updated {_generated_at} (server time) &middot;
    regenerated automatically after each scheduled GitHub Actions run.
  </div>
{_cards_html}
{_scanners_html}
<script>
  // scanner reports report their own height so each section fits without inner scrollbars
  window.addEventListener("message", function (e) {{
    var d = e.data;
    if (!d || d.type !== "scanner-height") return;
    document.querySelectorAll("iframe.scanner-frame").forEach(function (f) {{
      if (f.contentWindow === e.source) f.style.height = (d.h + 8) + "px";
    }});
  }});
</script>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as _f:
    _f.write(_page_html)

print(f"\nWrote {len(GENERATED_IMAGES)} chart(s) + index.html to '{OUTPUT_DIR}/'")
