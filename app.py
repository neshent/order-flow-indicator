"""
Order Flow Indicator — Flask Web App
Serves interactive Plotly chart with:
  - Candlestick chart (pan / zoom / hover)
  - Order Block boxes
  - Liquidity Zone lines
  - Buy/Sell Pressure labels
  - Net Delta bar chart
  - Rich hover tooltips on every element
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Re-use all indicator logic from order_flow_indicator.py
from order_flow_indicator import (
    fetch_data, candle_features, calc_pressure,
    detect_liquidity_zones, detect_order_blocks, CONFIG
)

app = Flask(__name__)

# ─────────────────────────────────────────
# SUPPORTED NSE SYMBOLS
# ─────────────────────────────────────────
NSE_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "NESTLEIND.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "ITC.NS",
    "NIFTYBEES.NS", "BANKBEES.NS", "ADANIENT.NS", "LTIM.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ONGC.NS", "POWERGRID.NS",
]

PERIODS   = ["1mo", "3mo", "6mo", "1y", "2y"]
INTERVALS = ["1d", "1wk"]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def build_indicator_data(symbol: str, period: str, interval: str) -> dict:
    """
    Fetch + calculate all indicator values.
    Returns a dict ready to be JSON-serialised for the frontend.
    """
    df = fetch_data(symbol, period=period, interval=interval)
    df = candle_features(df)
    df = calc_pressure(df, CONFIG)
    df = detect_liquidity_zones(df, CONFIG)
    bull_obs, bear_obs = detect_order_blocks(df, CONFIG)

    dates  = df.index.strftime("%Y-%m-%d").tolist()
    n      = len(df)

    # ── Candle data ──────────────────────────────────────────
    candles = {
        "dates":        dates,
        "open":         df["Open"].round(2).tolist(),
        "high":         df["High"].round(2).tolist(),
        "low":          df["Low"].round(2).tolist(),
        "close":        df["Close"].round(2).tolist(),
        "volume":       df["Volume"].fillna(0).astype(int).tolist(),
    }

    # ── Pressure / delta ─────────────────────────────────────
    pressure = {
        "buy_pressure":    (df["buy_pressure"]  * 100).round(1).tolist(),
        "sell_pressure":   (df["sell_pressure"] * 100).round(1).tolist(),
        "net_delta":       df["net_delta"].round(4).tolist(),
        "delta_ma":        df["delta_ma"].round(4).tolist(),
        "vol_factor":      df["vol_factor"].round(2).tolist(),
        "pressure_label":  df["pressure_label"].tolist(),
        "bull":            df["bull"].tolist(),
        "bear":            df["bear"].tolist(),
        "body":            df["body"].round(2).tolist(),
        "upper_wick":      df["upper_wick"].round(2).tolist(),
        "lower_wick":      df["lower_wick"].round(2).tolist(),
        "range":           df["range"].round(2).tolist(),
    }

    # ── Order blocks ─────────────────────────────────────────
    def ob_list(obs):
        return [
            {
                "bar_idx": o["bar_idx"],
                "date":    dates[o["bar_idx"]],
                "top":     round(float(o["top"]), 2),
                "bot":     round(float(o["bot"]), 2),
                "label":   o["label"],
                "range":   round(float(o["top"]) - float(o["bot"]), 2),
            }
            for o in obs
        ]

    order_blocks = {
        "bullish": ob_list(bull_obs),
        "bearish": ob_list(bear_obs),
    }

    # ── Liquidity zones ──────────────────────────────────────
    liq_zones = {
        "ssl": [],   # sell-side liquidity (equal highs)
        "bsl": [],   # buy-side  liquidity (equal lows)
    }
    seen_hi, seen_lo = set(), set()
    liq_len = CONFIG["liq_len"]

    for i, row in df.iterrows():
        xi = df.index.get_loc(i)
        if row["is_eq_high"]:
            price = round(float(row["highest_in_range"]), 2)
            if price not in seen_hi:
                liq_zones["ssl"].append({
                    "price":    price,
                    "x_start":  dates[max(0, xi - liq_len)],
                    "x_end":    dates[min(n - 1, xi + 5)],
                    "bar":      xi,
                    "date":     dates[xi],
                })
                seen_hi.add(price)
        if row["is_eq_low"]:
            price = round(float(row["lowest_in_range"]), 2)
            if price not in seen_lo:
                liq_zones["bsl"].append({
                    "price":    price,
                    "x_start":  dates[max(0, xi - liq_len)],
                    "x_end":    dates[min(n - 1, xi + 5)],
                    "bar":      xi,
                    "date":     dates[xi],
                })
                seen_lo.add(price)

    # ── Dashboard summary ────────────────────────────────────
    last = df.iloc[-1]
    dist = df["pressure_label"].value_counts().to_dict()

    dashboard = {
        "symbol":          symbol,
        "period":          period,
        "interval":        interval,
        "bars":            n,
        "date_from":       dates[0],
        "date_to":         dates[-1],
        "last_close":      round(float(last["Close"]), 2),
        "last_open":       round(float(last["Open"]), 2),
        "last_high":       round(float(last["High"]), 2),
        "last_low":        round(float(last["Low"]), 2),
        "pressure_bias":   str(last["pressure_label"]),
        "buy_pct":         round(float(last["buy_pressure"]) * 100, 1),
        "sell_pct":        round(float(last["sell_pressure"]) * 100, 1),
        "net_delta":       round(float(last["net_delta"]), 4),
        "vol_ratio":       round(float(last["vol_factor"]), 2),
        "liq_above":       bool(last["is_eq_high"]),
        "liq_below":       bool(last["is_eq_low"]),
        "bull_obs_count":  len(bull_obs),
        "bear_obs_count":  len(bear_obs),
        "ssl_count":       len(liq_zones["ssl"]),
        "bsl_count":       len(liq_zones["bsl"]),
        "dist_buy":        dist.get("▲ BUY",  0),
        "dist_sell":       dist.get("▼ SELL", 0),
        "dist_neut":       dist.get("◆ NEUT", 0),
    }

    return {
        "candles":      candles,
        "pressure":     pressure,
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
        default_symbol="RELIANCE.NS",
        default_period="3mo",
        default_interval="1d",
    )


@app.route("/api/data")
def api_data():
    symbol   = request.args.get("symbol",   "RELIANCE.NS")
    period   = request.args.get("period",   "3mo")
    interval = request.args.get("interval", "1d")

    # Basic validation
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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Order Flow Indicator Web App")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
