# Order Flow Indicator

A trading indicator that detects **Order Blocks**, **Liquidity Zones**, and **Buy/Sell Pressure** — built in both Pine Script v6 (TradingView) and Python.

## Files

| File | Description |
|---|---|
| `order_flow_indicator.pine` | TradingView Pine Script v6 indicator |
| `order_flow_indicator.py`   | Python version — runs locally with charts |
| `requirements.txt`          | Python dependencies |

## Features

- **Order Blocks** — Bullish & bearish OB detection using impulse moves
- **Liquidity Zones** — Equal highs (SSL) and equal lows (BSL) stop clusters
- **Buy/Sell Pressure** — Delta proxy using candle body, wicks, and volume
- **Dashboard** — Console summary of last bar's order flow state
- **Chart** — Dark-themed candlestick chart with all zones overlaid

## Quick Start

```bash
pip install -r requirements.txt
python order_flow_indicator.py
```

Default symbol is `RELIANCE.NS` (NSE). Change `SYMBOL` at the bottom of `order_flow_indicator.py` to any NSE ticker:

```python
SYMBOL   = "TCS.NS"      # or INFY.NS, HDFCBANK.NS, SBIN.NS ...
PERIOD   = "3mo"         # 1d | 5d | 1mo | 3mo | 6mo | 1y | 2y
INTERVAL = "1d"          # 1m | 5m | 15m | 30m | 1h | 1d | 1wk
```

## NSE Ticker Reference

| Sector | Tickers |
|---|---|
| Large-cap | `RELIANCE.NS` `TCS.NS` `INFY.NS` `HDFCBANK.NS` `ICICIBANK.NS` |
| Banking | `SBIN.NS` `AXISBANK.NS` `KOTAKBANK.NS` |
| Auto | `MARUTI.NS` `TATAMOTORS.NS` `BAJAJ-AUTO.NS` |
| Pharma | `SUNPHARMA.NS` `DRREDDY.NS` `CIPLA.NS` |
| FMCG | `NESTLEIND.NS` `HINDUNILVR.NS` `BRITANNIA.NS` |
| Index ETF | `NIFTYBEES.NS` `BANKBEES.NS` |

> **Note:** Pine Script cannot access live order book data. Both versions use price action + volume as a proxy for pending order activity. Forex support coming later.

## Requirements

- Python 3.12+
- numpy, pandas, matplotlib, yfinance
