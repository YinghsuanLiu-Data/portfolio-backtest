"""Data loading: download real daily adjusted prices (yfinance) with a local CSV cache."""
from pathlib import Path

import pandas as pd


def load_prices(tickers, start, end, cache_path=None, refresh=False):
    """Return a DataFrame of daily adjusted close prices (rows = dates, columns = tickers).

    Prices are adjusted for splits and dividends (``auto_adjust=True``), so the
    daily % change is the asset's *total* return.

    If ``cache_path`` is given, the data is saved there the first time and re-used
    afterwards so the notebook gives the same numbers every run.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not refresh:
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return prices[list(tickers)]

    import yfinance as yf  # imported here so the cache works without yfinance

    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"][list(tickers)].dropna(how="all")
    prices = prices.ffill().dropna()  # fill rare single-day gaps, drop leading NaNs

    if prices.empty:
        raise RuntimeError("No price data downloaded - check your internet connection.")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def load_risk_free_rate(start, end, default=0.04):
    """Average annual risk-free rate from the 13-week US T-bill yield (^IRX).

    Falls back to ``default`` if the download fails.
    """
    try:
        import yfinance as yf
        irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"]
        value = float(irx.squeeze().mean()) / 100  # quoted in percent
        return value if value == value else default  # NaN check
    except Exception:
        return default
