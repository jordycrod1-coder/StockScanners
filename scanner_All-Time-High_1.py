import os
import smtplib
import warnings
from datetime import datetime
from email.mime.text import MIMEText

import yfinance as yf
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

DATA_PERIOD = "max"       # full daily history needed for a TRUE all-time high
LOOKBACK_DAYS = 10        # check for new highs within the last N business days
RVOL_LOOKBACK = 20        # trading days used for the relative-volume average

# ============================================================
# FILTERS (applied after the scan, before printing / emailing)
# ============================================================

FILTERS_ACTIVE = True

FILTER_MAX_DAYS_SINCE_ATH = 3
FILTER_MIN_RVOL = 1.0
FILTER_MIN_TODAY_RETURN = None
FILTER_MIN_RSI = 50
FILTER_MAX_RSI = 80
FILTER_REQUIRE_MACD_BULL = True
FILTER_REQUIRE_STOCH_BULL = True

# GICS Sector -> SPDR Select Sector ETF
SECTOR_ETF_MAP = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

# ============================================================
# EMAIL SETTINGS
# ============================================================

SCANNER_NAME = "ATH Breakout Scanner"   # used in the email subject line

EMAIL_USER = os.environ.get("EMAIL_USER")   # sending gmail address
EMAIL_PASS = os.environ.get("EMAIL_PASS")   # gmail app password
ALERT_TO = os.environ.get("ALERT_TO")       # recipient address


def send_email(subject: str, body: str):
    if not (EMAIL_USER and EMAIL_PASS and ALERT_TO):
        print("Email credentials not set (EMAIL_USER / EMAIL_PASS / ALERT_TO) - skipping send.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ALERT_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, [ALERT_TO], msg.as_string())

    print(f"Email sent to {ALERT_TO}")


# ============================================================
# S&P 500 LOADER
# ============================================================

def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]

    ticker_col = name_col = sector_col = None
    for c in df.columns:
        if "Symbol" in c or "Ticker" in c:
            ticker_col = c
        if "Security" in c or ("Name" in c and "Sub" not in c) or "Company" in c:
            name_col = c
        if "GICS Sector" in c:
            sector_col = c

    sp500 = df[[ticker_col, name_col, sector_col]].copy()
    sp500.columns = ["Ticker", "Company", "Sector"]
    sp500["Ticker"] = sp500["Ticker"].str.replace(".", "-", regex=False)
    sp500["Sector_ETF"] = sp500["Sector"].map(SECTOR_ETF_MAP).fillna("N/A")

    return sp500


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal_period=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line


def stochastic(high, low, close, k_period=14, smooth_k=3, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()

    return k, d


# ============================================================
# MAIN
# ============================================================

def main():
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()

    print(f"Loaded {len(sp500)} S&P 500 tickers")
    print("Downloading full price history (period='max')...")

    data = yf.download(
        tickers,
        period=DATA_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    results = []

    for _, row in sp500.iterrows():
        t = row["Ticker"]
        company = row["Company"]
        sector_etf = row["Sector_ETF"]

        try:
            full_df = data[t].dropna()
            close = full_df["Close"]

            if len(close) <= LOOKBACK_DAYS:
                continue

            recent = close.iloc[-LOOKBACK_DAYS:]
            prior_ath = close.iloc[:-LOOKBACK_DAYS].max()

            recent_high = recent.max()

            if recent_high > prior_ath:
                recent_high_date = recent.idxmax()
                recent_high_pos = close.index.get_loc(recent_high_date)
                trading_days_since_ath = (len(close) - 1) - recent_high_pos

                high = full_df["High"]
                low = full_df["Low"]
                vol = full_df["Volume"]

                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                today_return_pct = ((last_close - prev_close) / prev_close) * 100

                last_vol = float(vol.iloc[-1])
                vol_window = vol.iloc[-(RVOL_LOOKBACK + 1):-1]
                avg_vol = vol_window.mean()
                rvol = (last_vol / avg_vol) if (pd.notna(avg_vol) and avg_vol > 0) else float("nan")

                rsi_v = rsi(close).iloc[-1]

                macd_line, signal_line = macd(close)
                macd_v = macd_line.iloc[-1]
                signal_v = signal_line.iloc[-1]
                macd_bull = bool(macd_v > signal_v)

                stoch_k, stoch_d = stochastic(high, low, close)
                k_v = stoch_k.iloc[-1]
                d_v = stoch_d.iloc[-1]
                stoch_bull = bool(k_v > d_v)

                results.append({
                    "Ticker": t,
                    "Company": company,
                    "Sector_ETF": sector_etf,
                    f"{LOOKBACK_DAYS}D_High_Close": round(float(recent_high), 2),
                    "Days_Since_ATH": trading_days_since_ath,
                    "Today_Return%": round(today_return_pct, 2),
                    "Volume": int(last_vol),
                    "RVOL": round(rvol, 2) if pd.notna(rvol) else None,
                    "RSI": round(float(rsi_v), 1) if pd.notna(rsi_v) else None,
                    "MACD": round(float(macd_v), 3) if pd.notna(macd_v) else None,
                    "MACD_Signal": round(float(signal_v), 3) if pd.notna(signal_v) else None,
                    "MACD_Bull": macd_bull,
                    "Stoch_%K": round(float(k_v), 1) if pd.notna(k_v) else None,
                    "Stoch_%D": round(float(d_v), 1) if pd.notna(d_v) else None,
                    "Stoch_Bull": stoch_bull,
                })

        except Exception:
            continue

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["Days_Since_ATH", "Sector_ETF", "Company"], ascending=[True, True, True])

    total_before_filters = len(out)

    if FILTERS_ACTIVE and not out.empty:
        if FILTER_MAX_DAYS_SINCE_ATH is not None:
            out = out[out["Days_Since_ATH"] <= FILTER_MAX_DAYS_SINCE_ATH]
        if FILTER_MIN_RVOL is not None:
            out = out[out["RVOL"] >= FILTER_MIN_RVOL]
        if FILTER_MIN_TODAY_RETURN is not None:
            out = out[out["Today_Return%"] >= FILTER_MIN_TODAY_RETURN]
        if FILTER_MIN_RSI is not None:
            out = out[out["RSI"] >= FILTER_MIN_RSI]
        if FILTER_MAX_RSI is not None:
            out = out[out["RSI"] <= FILTER_MAX_RSI]
        if FILTER_REQUIRE_MACD_BULL is not None:
            out = out[out["MACD_Bull"] == FILTER_REQUIRE_MACD_BULL]
        if FILTER_REQUIRE_STOCH_BULL is not None:
            out = out[out["Stoch_Bull"] == FILTER_REQUIRE_STOCH_BULL]

    # ------------------------------------------------------------
    # Build console + email output
    # ------------------------------------------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append(f"NEW ALL-TIME CLOSING HIGHS - LAST {LOOKBACK_DAYS} BUSINESS DAYS ({today_str})")
    if FILTERS_ACTIVE:
        lines.append(f"({len(out)} of {total_before_filters} results passed the active filters)")
    lines.append("=" * 70)

    if out.empty:
        if total_before_filters > 0 and FILTERS_ACTIVE:
            lines.append(f"No tickers passed the active filters ({total_before_filters} made a new ATH before filtering).")
        else:
            lines.append("No tickers made a new all-time closing high in this window.")
    else:
        for _, r in out.iterrows():
            lines.append(
                f"{r['Ticker']} ({r['Company']}, {r['Sector_ETF']}) - "
                f"High: {r[f'{LOOKBACK_DAYS}D_High_Close']}, "
                f"Days Since ATH: {r['Days_Since_ATH']}, "
                f"Return: {r['Today_Return%']}%, "
                f"RVOL: {r['RVOL']}, RSI: {r['RSI']}, "
                f"MACD Bull: {r['MACD_Bull']}, Stoch Bull: {r['Stoch_Bull']}"
            )

    body = "\n".join(lines)
    print(body)

    ticker_count = len(out)
    if ticker_count > 0:
        ticker_list = ", ".join(out["Ticker"].tolist())
        subject = f"{SCANNER_NAME}: {ticker_count} hit(s) - {ticker_list} - {today_str}"
    else:
        subject = f"{SCANNER_NAME}: No hits - {today_str}"

    send_email(subject, body)


if __name__ == "__main__":
    main()
