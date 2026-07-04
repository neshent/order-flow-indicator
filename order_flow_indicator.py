"""
============================================================
Order Flow Indicator — Python (converted from Pine Script v6)
Detects: Order Blocks | Liquidity Zones | Buy/Sell Pressure

Default data source: NSE stocks via yfinance (TICKER.NS format)
Charts:              matplotlib
============================================================

Usage:
    python order_flow_indicator.py

    Change SYMBOL at the bottom to any NSE ticker, e.g.:
        RELIANCE.NS  TCS.NS  INFY.NS  HDFCBANK.NS  SBIN.NS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG  (mirrors Pine Script inputs)
# ─────────────────────────────────────────
CONFIG = {
    # Order Blocks
    "ob_len":       10,     # Swing length for OB detection
    "ob_extend":    20,     # How many bars to extend OB box to the right (visual only)
    "ob_show":      True,

    # Liquidity Zones
    "liq_len":      20,     # Lookback window for equal H/L detection
    "liq_thresh":   0.1,    # Threshold % — how close two highs/lows must be to be "equal"
    "liq_show":     True,

    # Buy/Sell Pressure
    "pres_vol":     True,   # Weight pressure by volume
    "delta_bars":   14,     # Rolling window for MA / volume normalisation

    # Dashboard
    "dash_show":    True,
}

# ─────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────

def fetch_data(symbol: str = "RELIANCE.NS", period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance.

    NSE stocks  → use TICKER.NS   e.g. "RELIANCE.NS", "TCS.NS", "INFY.NS"
    BSE stocks  → use TICKER.BO   e.g. "RELIANCE.BO"
    Crypto      → use TICKER-USD  e.g. "BTC-USD"      (add later if needed)
    Forex       → use PAIR=X      e.g. "USDINR=X"     (add later if needed)

    The returned DataFrame must have columns:
        Open, High, Low, Close, Volume
    with a DatetimeIndex.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def rolling_highest(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()

def rolling_lowest(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()

def rolling_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Returns the pivot high value at bar i if high[i] is the maximum
    over [i-left .. i+right], else NaN.
    """
    result = pd.Series(np.nan, index=high.index)
    arr = high.values
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.max():
            result.iloc[i] = arr[i]
    return result

def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """
    Returns the pivot low value at bar i if low[i] is the minimum
    over [i-left .. i+right], else NaN.
    """
    result = pd.Series(np.nan, index=low.index)
    arr = low.values
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.min():
            result.iloc[i] = arr[i]
    return result


# ─────────────────────────────────────────
# CANDLE HELPERS
# ─────────────────────────────────────────

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle analysis columns to the DataFrame."""
    df = df.copy()
    df["bull"]       = df["Close"] > df["Open"]
    df["bear"]       = df["Close"] < df["Open"]
    df["doji"]       = (df["Close"] - df["Open"]).abs() <= (df["High"] - df["Low"]) * 0.1
    df["body"]       = (df["Close"] - df["Open"]).abs()
    df["range"]      = (df["High"] - df["Low"]).replace(0, 1e-9)

    # Upper wick: from top of body to high
    df["upper_wick"] = np.where(df["bull"], df["High"] - df["Close"], df["High"] - df["Open"])
    # Lower wick: from bottom of body to low
    df["lower_wick"] = np.where(df["bull"], df["Open"] - df["Low"],   df["Close"] - df["Low"])

    return df


# ─────────────────────────────────────────
# BUY / SELL PRESSURE
# ─────────────────────────────────────────

def calc_pressure(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Mirrors Pine Script delta proxy logic.
    Returns df with columns: buy_pressure, sell_pressure, net_delta,
                              delta_ma, vol_factor, pressure_label
    """
    df = df.copy()
    delta_bars = cfg["delta_bars"]
    use_vol    = cfg["pres_vol"]

    # Raw pressure (0 – 1.5 range)
    df["buy_raw"]  = (df["body"] / df["range"]) * df["bull"].astype(float) \
                   + (df["lower_wick"] / df["range"]) * 0.5
    df["sell_raw"] = (df["body"] / df["range"]) * df["bear"].astype(float) \
                   + (df["upper_wick"] / df["range"]) * 0.5

    # Volume factor: current bar volume / rolling mean volume
    if use_vol and "Volume" in df.columns and df["Volume"].sum() > 0:
        vol_ma = rolling_sma(df["Volume"].astype(float), delta_bars).replace(0, 1e-9)
        df["vol_factor"] = df["Volume"].astype(float) / vol_ma
    else:
        df["vol_factor"] = 1.0

    df["buy_pressure"]  = df["buy_raw"]  * df["vol_factor"]
    df["sell_pressure"] = df["sell_raw"] * df["vol_factor"]
    df["net_delta"]     = df["buy_pressure"] - df["sell_pressure"]
    df["delta_ma"]      = rolling_sma(df["net_delta"], delta_bars)

    # Pressure label  (mirrors Pine ternary logic)
    def label(row):
        if row["net_delta"] > row["delta_ma"] * 1.5:
            return "▲ BUY"
        elif row["net_delta"] < row["delta_ma"] * -1.5:
            return "▼ SELL"
        return "◆ NEUT"

    df["pressure_label"] = df.apply(label, axis=1)
    return df


# ─────────────────────────────────────────
# ORDER BLOCK DETECTION
# ─────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> tuple[list, list]:
    """
    Returns two lists of dicts — bullish_obs and bearish_obs.
    Each dict: {bar_idx, top, bot, label}
    """
    ob_len  = cfg["ob_len"]
    bull_obs, bear_obs = [], []

    highs = df["High"].values
    lows  = df["Low"].values
    closes = df["Close"].values
    opens  = df["Open"].values

    for i in range(ob_len + 1, len(df)):
        # Impulse: close breaks above the highest high of the last ob_len bars
        bull_impulse = closes[i] > np.max(highs[i - ob_len: i])
        # Impulse: close breaks below the lowest low of the last ob_len bars
        bear_impulse = closes[i] < np.min(lows[i - ob_len: i])

        prev_bear = opens[i - 1] > closes[i - 1]   # previous candle was bearish
        prev_bull = closes[i - 1] > opens[i - 1]   # previous candle was bullish

        if bull_impulse and prev_bear:
            bull_obs.append({
                "bar_idx": i - 1,
                "top":     highs[i - 1],
                "bot":     lows[i - 1],
                "label":   "Bull OB",
            })

        if bear_impulse and prev_bull:
            bear_obs.append({
                "bar_idx": i - 1,
                "top":     highs[i - 1],
                "bot":     lows[i - 1],
                "label":   "Bear OB",
            })

    return bull_obs, bear_obs


# ─────────────────────────────────────────
# LIQUIDITY ZONE DETECTION
# ─────────────────────────────────────────

def detect_liquidity_zones(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Mirrors Pine Script equal-highs / equal-lows detection.
    Returns df with boolean columns: is_eq_high, is_eq_low,
    and price columns: highest_in_range, lowest_in_range.
    """
    df = df.copy()
    liq_len    = cfg["liq_len"]
    liq_thresh = cfg["liq_thresh"]   # percentage

    highs = df["High"].values
    lows  = df["Low"].values
    closes = df["Close"].values

    eq_high = np.zeros(len(df), dtype=bool)
    eq_low  = np.zeros(len(df), dtype=bool)
    hi_vals = np.full(len(df), np.nan)
    lo_vals = np.full(len(df), np.nan)

    for i in range(liq_len, len(df)):
        window_h = highs[i - liq_len: i]
        window_l = lows[i - liq_len:  i]
        thresh   = closes[i] * (liq_thresh / 100)

        highest  = window_h.max()
        lowest   = window_l.min()

        eq_h_count = np.sum(np.abs(window_h - highest) <= thresh)
        eq_l_count = np.sum(np.abs(window_l - lowest)  <= thresh)

        eq_high[i] = eq_h_count >= 2
        eq_low[i]  = eq_l_count >= 2
        hi_vals[i] = highest
        lo_vals[i] = lowest

    df["is_eq_high"]      = eq_high
    df["is_eq_low"]       = eq_low
    df["highest_in_range"] = hi_vals
    df["lowest_in_range"]  = lo_vals

    return df


# ─────────────────────────────────────────
# DASHBOARD (text output)
# ─────────────────────────────────────────

def print_dashboard(df: pd.DataFrame, cfg: dict):
    """Print the last bar's order flow summary — mirrors Pine Script table."""
    if not cfg["dash_show"]:
        return

    last = df.iloc[-1]

    print("\n" + "═" * 42)
    print(f"{'ORDER FLOW PANEL':^42}")
    print("═" * 42)

    rows = [
        ("Pressure Bias",   last.get("pressure_label", "N/A")),
        ("Buy Pressure",    f"{last.get('buy_pressure', 0) * 100:.1f}%"),
        ("Sell Pressure",   f"{last.get('sell_pressure', 0) * 100:.1f}%"),
        ("Net Delta",       f"{last.get('net_delta', 0):.4f}"),
        ("Liquidity Above", "SSL Zone ⚠" if last.get("is_eq_high") else "—"),
        ("Liquidity Below", "BSL Zone ⚠" if last.get("is_eq_low")  else "—"),
        ("Volume Ratio",    f"{last.get('vol_factor', 1):.2f}x"),
    ]
    for label, value in rows:
        print(f"  {label:<20} {value:>18}")
    print("═" * 42 + "\n")


# ─────────────────────────────────────────
# CHART
# ─────────────────────────────────────────

def plot_chart(df: pd.DataFrame, bull_obs: list, bear_obs: list, cfg: dict, symbol: str = ""):
    """
    Draw candlestick chart with:
      - Order block boxes
      - Liquidity zone dashed lines
      - Pressure labels above/below candles
      - Net delta bar chart in lower panel
    """
    n = len(df)
    idx = np.arange(n)         # integer x-axis
    dates = df.index

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0d1117"
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ── Candlesticks ──────────────────────────────────────────
    for i, (idx_val, row) in enumerate(df.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if c >= o else "#ef5350"  # TradingView green/red
        # Body
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 1e-9)
        ax1.add_patch(plt.Rectangle(
            (i - 0.4, body_bottom), 0.8, body_height,
            color=color, zorder=3
        ))
        # Wicks
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)

    # ── Order Block Boxes ────────────────────────────────────
    if cfg["ob_show"]:
        ob_extend = cfg["ob_extend"]
        for ob in bull_obs[-10:]:   # show last 10
            x_start = ob["bar_idx"]
            x_end   = min(ob["bar_idx"] + ob_extend, n - 1)
            rect = mpatches.FancyBboxPatch(
                (x_start - 0.4, ob["bot"]),
                (x_end - x_start + 0.8), (ob["top"] - ob["bot"]),
                boxstyle="square,pad=0",
                linewidth=1, edgecolor="#26a69a",
                facecolor="#26a69a22", zorder=1
            )
            ax1.add_patch(rect)
            ax1.text(x_start, ob["top"], " Bull OB", fontsize=6,
                     color="#26a69a", va="bottom", zorder=5)

        for ob in bear_obs[-10:]:
            x_start = ob["bar_idx"]
            x_end   = min(ob["bar_idx"] + ob_extend, n - 1)
            rect = mpatches.FancyBboxPatch(
                (x_start - 0.4, ob["bot"]),
                (x_end - x_start + 0.8), (ob["top"] - ob["bot"]),
                boxstyle="square,pad=0",
                linewidth=1, edgecolor="#ef5350",
                facecolor="#ef535022", zorder=1
            )
            ax1.add_patch(rect)
            ax1.text(x_start, ob["bot"], " Bear OB", fontsize=6,
                     color="#ef5350", va="top", zorder=5)

    # ── Liquidity Zone Lines ──────────────────────────────────
    if cfg["liq_show"]:
        liq_len = cfg["liq_len"]
        plotted_hi, plotted_lo = set(), set()   # avoid duplicate lines at same price

        for i, row in df.iterrows():
            xi = df.index.get_loc(i)
            if row.get("is_eq_high") and round(row["highest_in_range"], 6) not in plotted_hi:
                y = row["highest_in_range"]
                ax1.hlines(y, xi - liq_len, xi + 5,
                           colors="#ff9800", linewidths=1.2, linestyles="--", zorder=4)
                ax1.text(xi + 5, y, " SSL 🔴", fontsize=6, color="#ff9800",
                         va="center", zorder=5)
                plotted_hi.add(round(y, 6))

            if row.get("is_eq_low") and round(row["lowest_in_range"], 6) not in plotted_lo:
                y = row["lowest_in_range"]
                ax1.hlines(y, xi - liq_len, xi + 5,
                           colors="#00bcd4", linewidths=1.2, linestyles="--", zorder=4)
                ax1.text(xi + 5, y, " BSL 🟢", fontsize=6, color="#00bcd4",
                         va="center", zorder=5)
                plotted_lo.add(round(y, 6))

    # ── Pressure Labels ────────────────────────────────────────
    for i, (idx_val, row) in enumerate(df.iterrows()):
        lbl = row.get("pressure_label", "")
        if lbl == "▲ BUY":
            ax1.text(i, row["High"] + row["range"] * 0.3, lbl,
                     fontsize=5, color="#26a69a", ha="center", va="bottom", zorder=6)
        elif lbl == "▼ SELL":
            ax1.text(i, row["Low"] - row["range"] * 0.3, lbl,
                     fontsize=5, color="#ef5350", ha="center", va="top", zorder=6)

    # ── Net Delta Bar Chart ────────────────────────────────────
    delta = df["net_delta"].values
    colors_delta = ["#26a69a" if d >= 0 else "#ef5350" for d in delta]
    ax2.bar(idx, delta, color=colors_delta, width=0.8, zorder=3)
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_ylabel("Net Delta", color="gray", fontsize=8)

    # ── Axes formatting ───────────────────────────────────────
    tick_step = max(1, n // 12)
    tick_idx  = list(range(0, n, tick_step))
    ax1.set_xlim(-1, n + cfg["ob_extend"])
    ax2.set_xlim(-1, n + cfg["ob_extend"])
    ax1.set_xticks([])
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(
        [dates[i].strftime("%b %d") for i in tick_idx],
        rotation=30, fontsize=7, color="gray"
    )

    ax1.yaxis.set_tick_params(labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)

    title = f"Order Flow Indicator — {symbol}" if symbol else "Order Flow Indicator"
    ax1.set_title(title, color="white", fontsize=11, pad=8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#26a69a33", edgecolor="#26a69a", label="Bullish OB"),
        mpatches.Patch(facecolor="#ef535033", edgecolor="#ef5350", label="Bearish OB"),
        mlines.Line2D([], [], color="#ff9800", linestyle="--", label="SSL (Sell-Side Liq.)"),
        mlines.Line2D([], [], color="#00bcd4", linestyle="--", label="BSL (Buy-Side Liq.)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left",
               facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="white", fontsize=7)

    plt.tight_layout(h_pad=0.3)
    plt.show()


# ─────────────────────────────────────────
# FULL ANALYSIS FUNCTION
# ─────────────────────────────────────────

def run_analysis(df: pd.DataFrame, cfg: dict = CONFIG, symbol: str = "") -> pd.DataFrame:
    """
    Run all indicator calculations on a raw OHLCV DataFrame.
    Returns the enriched DataFrame with all indicator columns.
    """
    # 1. Candle features
    df = candle_features(df)

    # 2. Buy/Sell pressure + delta
    df = calc_pressure(df, cfg)

    # 3. Liquidity zones
    df = detect_liquidity_zones(df, cfg)

    # 4. Order blocks (separate lists)
    bull_obs, bear_obs = detect_order_blocks(df, cfg)

    # 5. Console dashboard
    print_dashboard(df, cfg)

    # 6. Summary of detected zones
    print(f"  Bullish Order Blocks detected : {len(bull_obs)}")
    print(f"  Bearish Order Blocks detected : {len(bear_obs)}")
    if bull_obs:
        latest_bull = bull_obs[-1]
        print(f"  Latest Bull OB  → bar {latest_bull['bar_idx']:>4}  "
              f"zone {latest_bull['bot']:.4f} – {latest_bull['top']:.4f}")
    if bear_obs:
        latest_bear = bear_obs[-1]
        print(f"  Latest Bear OB  → bar {latest_bear['bar_idx']:>4}  "
              f"zone {latest_bear['bot']:.4f} – {latest_bear['top']:.4f}")
    print()

    # 7. Chart
    plot_chart(df, bull_obs, bear_obs, cfg, symbol=symbol)

    return df


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    # ── NSE Stock Symbols (yfinance format = TICKER.NS) ──────────────
    # Large-cap examples:
    #   RELIANCE.NS  TCS.NS       INFY.NS      HDFCBANK.NS  ICICIBANK.NS
    #   SBIN.NS      WIPRO.NS     BAJFINANCE.NS ADANIENT.NS  AXISBANK.NS
    #   MARUTI.NS    TATAMOTORS.NS SUNPHARMA.NS LTIM.NS      NESTLEIND.NS
    #
    # Index ETFs:
    #   NIFTYBEES.NS  BANKBEES.NS  JUNIORBEES.NS
    #
    # NOTE: Forex (USDINR=X etc.) can be added later — just change SYMBOL.
    # ─────────────────────────────────────────────────────────────────

    SYMBOL   = "RELIANCE.NS"   # NSE stock — change to any TICKER.NS above
    PERIOD   = "3mo"           # 1d | 5d | 1mo | 3mo | 6mo | 1y | 2y | 5y
    INTERVAL = "1d"            # 1m | 5m | 15m | 30m | 1h | 1d | 1wk

    print(f"\nFetching {SYMBOL} — {PERIOD} @ {INTERVAL} ...")
    df = fetch_data(SYMBOL, period=PERIOD, interval=INTERVAL)
    print(f"Loaded {len(df)} bars.\n")

    result = run_analysis(df, cfg=CONFIG, symbol=SYMBOL)

    # Optional: print last 5 rows of enriched data
    cols = ["Open", "High", "Low", "Close",
            "buy_pressure", "sell_pressure", "net_delta",
            "pressure_label", "is_eq_high", "is_eq_low"]
    print(result[cols].tail(5).to_string())
