"""
Order Flow Indicator — Flask Web App (v2)
Enhanced with:
  - Full period/interval options (matches TradingView)
  - MA-20, MA-50, MA-200 overlays
  - Bollinger Bands
  - FII/DII proxy (institutional vs retail pressure via volume decile)
  - Richer JSON API for frontend
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from order_flow_indicator import (
    fetch_data, candle_features, calc_pressure,
    detect_liquidity_zones, detect_order_blocks, CONFIG,
    rolling_sma
)

app = Flask(__name__)

# ─────────────────────────────────────────
# SYMBOL LIST
# ─────────────────────────────────────────
NSE_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "NESTLEIND.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "ITC.NS",
    "NIFTYBEES.NS", "BANKBEES.NS", "ADANIENT.NS", "LTIM.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ONGC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "GRASIM.NS", "BPCL.NS", "HEROMOTOCO.NS",
    "M&M.NS", "TITAN.NS", "HCLTECH.NS", "TECHM.NS", "COALINDIA.NS",
    "NTPC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "BAJAJFINSV.NS",
]

# ─────────────────────────────────────────
# VALID PERIOD / INTERVAL COMBOS
# Mirrors what yfinance + TradingView support
# ─────────────────────────────────────────
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

INTERVALS = ["1m", "5m", "15m", "30m", "60m", "90m", "1h", "4h", "1d", "5d", "1wk", "1mo", "3mo"]

# yfinance interval constraints:
# 1m  → max 7d of data
# 5m/15m/30m/90m → max 60d
# 60m/1h → max 730d
# 4h  → max 60d (via 1h aggregation)
# 1d+ → unlimited
INTERVAL_MAX_PERIOD = {
    "1m":  "7d",   "5m":  "60d",  "15m": "60d",
    "30m": "60d",  "60m": "730d", "90m": "60d",
    "1h":  "730d", "4h":  "60d",  "1d":  "max",
    "5d":  "max",  "1wk": "max",  "1mo": "max",  "3mo": "max",
}

PERIOD_LABELS = {
    "1d": "1D", "5d": "5D", "1mo": "1M", "3mo": "3M",
    "6mo": "6M", "1y": "1Y", "2y": "2Y", "5y": "5Y", "max": "MAX",
}

INTERVAL_LABELS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "90m": "90m", "1h": "1h", "4h": "4h",
    "1d": "1D", "5d": "5D", "1wk": "1W", "1mo": "1M", "3mo": "3M",
}


# ─────────────────────────────────────────
# TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────

def calc_ma(close: pd.Series, period: int) -> list:
    return close.rolling(period, min_periods=1).mean().round(2).tolist()

def calc_ema(close: pd.Series, period: int) -> list:
    return close.ewm(span=period, adjust=False).mean().round(2).tolist()

def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    ma   = close.rolling(period, min_periods=1).mean()
    std  = close.rolling(period, min_periods=1).std().fillna(0)
    return {
        "mid":   ma.round(2).tolist(),
        "upper": (ma + std_dev * std).round(2).tolist(),
        "lower": (ma - std_dev * std).round(2).tolist(),
    }

def calc_rsi(close: pd.Series, period: int = 14) -> list:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, 1e-9)
    rsi    = (100 - 100 / (1 + rs)).round(2)
    return rsi.fillna(50).tolist()

def calc_macd(close: pd.Series) -> dict:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    return {
        "macd":      macd.round(4).tolist(),
        "signal":    sig.round(4).tolist(),
        "histogram": hist.round(4).tolist(),
    }

def calc_atr(df: pd.DataFrame, period: int = 14) -> list:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().round(2).fillna(0).tolist()

def calc_vwap(df: pd.DataFrame) -> list:
    """Session VWAP — resets daily when using intraday data."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].fillna(0).astype(float)
    cum_tp_vol = (typical * vol).cumsum()
    cum_vol    = vol.cumsum().replace(0, 1e-9)
    return (cum_tp_vol / cum_vol).round(2).tolist()


# ─────────────────────────────────────────
# FII / DII PROXY
# ─────────────────────────────────────────
# Real FII/DII data requires NSE/BSE API (paid).
# We proxy it using volume decile analysis:
#   - Large candles with high volume above 80th pct → FII (institutional)
#   - Smaller candles with mid-range volume         → DII (domestic)
#   - Low volume candles                            → Retail
# Confidence score 0–100 per bar.

def calc_fii_dii_proxy(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    vol    = df["Volume"].fillna(0).astype(float)
    body   = (df["Close"] - df["Open"]).abs()
    rng    = (df["High"] - df["Low"]).replace(0, 1e-9)

    vol_p80  = vol.quantile(0.80)
    vol_p50  = vol.quantile(0.50)
    vol_p20  = vol.quantile(0.20)
    body_p60 = body.quantile(0.60)

    fii_buy, fii_sell = [], []
    dii_buy, dii_sell = [], []
    fii_conf, dii_conf = [], []

    for i in range(len(df)):
        v  = vol.iloc[i]
        b  = body.iloc[i]
        r  = rng.iloc[i]
        is_bull = df["Close"].iloc[i] >= df["Open"].iloc[i]
        body_ratio = b / r

        # FII proxy: high volume + large body (decisive institutional moves)
        if v >= vol_p80 and b >= body_p60:
            conf = min(100, int(((v / vol_p80) * 50) + (body_ratio * 50)))
            fii_conf.append(conf)
            if is_bull:
                fii_buy.append(conf);  fii_sell.append(0)
                dii_buy.append(0);     dii_sell.append(0)
            else:
                fii_sell.append(conf); fii_buy.append(0)
                dii_buy.append(0);     dii_sell.append(0)

        # DII proxy: medium volume + moderate body
        elif vol_p20 <= v < vol_p80 and body_ratio > 0.35:
            conf = min(100, int(((v / vol_p50) * 40) + (body_ratio * 60)))
            dii_conf.append(conf)
            if is_bull:
                dii_buy.append(conf);  dii_sell.append(0)
                fii_buy.append(0);     fii_sell.append(0)
            else:
                dii_sell.append(conf); dii_buy.append(0)
                fii_buy.append(0);     fii_sell.append(0)
        else:
            fii_buy.append(0);  fii_sell.append(0)
            dii_buy.append(0);  dii_sell.append(0)

    # Rolling 5-bar cumulative FII/DII scores
    fii_net = [b - s for b, s in zip(fii_buy, fii_sell)]
    dii_net = [b - s for b, s in zip(dii_buy, dii_sell)]

    def roll5(lst):
        s = pd.Series(lst, dtype=float)
        return s.rolling(5, min_periods=1).mean().round(1).tolist()

    return {
        "fii_buy":   fii_buy,
        "fii_sell":  fii_sell,
        "dii_buy":   dii_buy,
        "dii_sell":  dii_sell,
        "fii_net":   fii_net,
        "dii_net":   dii_net,
        "fii_roll":  roll5(fii_net),
        "dii_roll":  roll5(dii_net),
    }


# ─────────────────────────────────────────
# 4H AGGREGATION
# ─────────────────────────────────────────

def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h OHLCV into 4h bars."""
    df4 = df.resample("4h").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
    return df4


# ─────────────────────────────────────────
# MAIN DATA BUILDER
# ─────────────────────────────────────────

def build_indicator_data(symbol: str, period: str, interval: str) -> dict:

    # Handle 4h specially — fetch 1h and resample
    fetch_interval = interval
    if interval == "4h":
        fetch_interval = "1h"
        raw = fetch_data(symbol, period=min(period, "60d") if period not in ("max","5y","2y","1y") else "60d", interval="1h")
        df  = resample_4h(raw)
    else:
        df = fetch_data(symbol, period=period, interval=fetch_interval)

    df = candle_features(df)
    df = calc_pressure(df, CONFIG)
    df = detect_liquidity_zones(df, CONFIG)
    bull_obs, bear_obs = detect_order_blocks(df, CONFIG)

    # Date format: include time for intraday
    intraday = interval in ("1m","5m","15m","30m","60m","90m","1h","4h")
    if intraday:
        dates = df.index.strftime("%Y-%m-%d %H:%M").tolist()
    else:
        dates = df.index.strftime("%Y-%m-%d").tolist()
    n = len(df)

    # ── Candles ──────────────────────────────────────────────
    candles = {
        "dates":    dates,
        "open":     df["Open"].round(2).tolist(),
        "high":     df["High"].round(2).tolist(),
        "low":      df["Low"].round(2).tolist(),
        "close":    df["Close"].round(2).tolist(),
        "volume":   df["Volume"].fillna(0).astype(int).tolist(),
        "intraday": intraday,
    }

    # ── Pressure ─────────────────────────────────────────────
    pressure = {
        "buy_pressure":   (df["buy_pressure"]  * 100).round(1).tolist(),
        "sell_pressure":  (df["sell_pressure"] * 100).round(1).tolist(),
        "net_delta":      df["net_delta"].round(4).tolist(),
        "delta_ma":       df["delta_ma"].round(4).tolist(),
        "vol_factor":     df["vol_factor"].round(2).tolist(),
        "pressure_label": df["pressure_label"].tolist(),
        "bull":           df["bull"].tolist(),
        "bear":           df["bear"].tolist(),
        "body":           df["body"].round(2).tolist(),
        "upper_wick":     df["upper_wick"].round(2).tolist(),
        "lower_wick":     df["lower_wick"].round(2).tolist(),
        "range":          df["range"].round(2).tolist(),
    }

    # ── Technical overlays ───────────────────────────────────
    close = df["Close"]
    bb    = calc_bollinger(close, 20, 2.0)
    macd  = calc_macd(close)
    technicals = {
        "ma20":   calc_ma(close, 20),
        "ma50":   calc_ma(close, 50),
        "ma200":  calc_ma(close, 200),
        "ema9":   calc_ema(close, 9),
        "ema21":  calc_ema(close, 21),
        "bb_mid":   bb["mid"],
        "bb_upper": bb["upper"],
        "bb_lower": bb["lower"],
        "rsi":    calc_rsi(close, 14),
        "macd":   macd["macd"],
        "macd_signal": macd["signal"],
        "macd_hist":   macd["histogram"],
        "atr":    calc_atr(df, 14),
        "vwap":   calc_vwap(df),
    }

    # ── FII / DII proxy ──────────────────────────────────────
    fii_dii = calc_fii_dii_proxy(df)

    # ── Order Blocks ─────────────────────────────────────────
    def ob_list(obs):
        return [{
            "bar_idx": o["bar_idx"],
            "date":    dates[o["bar_idx"]],
            "top":     round(float(o["top"]), 2),
            "bot":     round(float(o["bot"]), 2),
            "label":   o["label"],
            "range":   round(float(o["top"]) - float(o["bot"]), 2),
        } for o in obs]

    order_blocks = {
        "bullish": ob_list(bull_obs),
        "bearish": ob_list(bear_obs),
    }

    # ── Liquidity Zones ──────────────────────────────────────
    liq_zones = {"ssl": [], "bsl": []}
    seen_hi, seen_lo = set(), set()
    liq_len = CONFIG["liq_len"]

    for i, row in df.iterrows():
        xi = df.index.get_loc(i)
        if row["is_eq_high"]:
            price = round(float(row["highest_in_range"]), 2)
            if price not in seen_hi:
                liq_zones["ssl"].append({
                    "price":   price,
                    "x_start": dates[max(0, xi - liq_len)],
                    "x_end":   dates[min(n - 1, xi + 5)],
                    "bar":     xi, "date": dates[xi],
                })
                seen_hi.add(price)
        if row["is_eq_low"]:
            price = round(float(row["lowest_in_range"]), 2)
            if price not in seen_lo:
                liq_zones["bsl"].append({
                    "price":   price,
                    "x_start": dates[max(0, xi - liq_len)],
                    "x_end":   dates[min(n - 1, xi + 5)],
                    "bar":     xi, "date": dates[xi],
                })
                seen_lo.add(price)

    # ── Dashboard ────────────────────────────────────────────
    last  = df.iloc[-1]
    dist  = df["pressure_label"].value_counts().to_dict()
    last_rsi  = technicals["rsi"][-1]
    last_macd = technicals["macd"][-1]
    last_atr  = technicals["atr"][-1]

    dashboard = {
        "symbol": symbol, "period": period, "interval": interval,
        "bars": n, "date_from": dates[0], "date_to": dates[-1],
        "last_close":  round(float(last["Close"]), 2),
        "last_open":   round(float(last["Open"]),  2),
        "last_high":   round(float(last["High"]),  2),
        "last_low":    round(float(last["Low"]),   2),
        "last_volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
        "pressure_bias": str(last["pressure_label"]),
        "buy_pct":     round(float(last["buy_pressure"])  * 100, 1),
        "sell_pct":    round(float(last["sell_pressure"]) * 100, 1),
        "net_delta":   round(float(last["net_delta"]), 4),
        "vol_ratio":   round(float(last["vol_factor"]), 2),
        "liq_above":   bool(last["is_eq_high"]),
        "liq_below":   bool(last["is_eq_low"]),
        "bull_obs_count": len(bull_obs),
        "bear_obs_count": len(bear_obs),
        "ssl_count": len(liq_zones["ssl"]),
        "bsl_count": len(liq_zones["bsl"]),
        "dist_buy":  dist.get("▲ BUY",  0),
        "dist_sell": dist.get("▼ SELL", 0),
        "dist_neut": dist.get("◆ NEUT", 0),
        "rsi":  round(last_rsi,  2),
        "macd": round(last_macd, 4),
        "atr":  round(last_atr,  2),
        "ma20":  round(technicals["ma20"][-1],  2),
        "ma50":  round(technicals["ma50"][-1],  2),
        "ma200": round(technicals["ma200"][-1], 2),
        "bb_upper": round(technicals["bb_upper"][-1], 2),
        "bb_lower": round(technicals["bb_lower"][-1], 2),
        "fii_net_last": fii_dii["fii_net"][-1],
        "dii_net_last": fii_dii["dii_net"][-1],
    }

    return {
        "candles":      candles,
        "pressure":     pressure,
        "technicals":   technicals,
        "fii_dii":      fii_dii,
        "order_blocks": order_blocks,
        "liq_zones":    liq_zones,
        "dashboard":    dashboard,
        "config":       CONFIG,
    }


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        symbols=NSE_SYMBOLS,
        periods=PERIODS,
        intervals=INTERVALS,
        period_labels=PERIOD_LABELS,
        interval_labels=INTERVAL_LABELS,
        default_symbol="RELIANCE.NS",
        default_period="3mo",
        default_interval="1d",
    )


@app.route("/api/data")
def api_data():
    symbol   = request.args.get("symbol",   "RELIANCE.NS")
    period   = request.args.get("period",   "3mo")
    interval = request.args.get("interval", "1d")

    if symbol not in NSE_SYMBOLS:
        return jsonify({"error": f"Unknown symbol: {symbol}"}), 400
    if period not in PERIODS:
        return jsonify({"error": f"Unknown period: {period}"}), 400
    if interval not in INTERVALS:
        return jsonify({"error": f"Unknown interval: {interval}"}), 400

    try:
        data = build_indicator_data(symbol, period, interval)
        return jsonify(data)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/symbols")
def api_symbols():
    return jsonify(NSE_SYMBOLS)


if __name__ == "__main__":
    print("\n  Order Flow Indicator v2")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
