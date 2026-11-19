"""Performance metrics for a series of daily returns."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annual_return(daily_returns):
    """Compound annual growth rate (CAGR)."""
    growth = (1 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS
    return growth ** (1 / years) - 1


def annual_volatility(daily_returns):
    """Annualised standard deviation of daily returns."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns, rf=0.0):
    """Annualised Sharpe ratio: (mean excess return) / volatility."""
    excess = daily_returns - rf / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def drawdown_series(daily_returns):
    """% below the previous peak at each date (0 = at a new high)."""
    wealth = (1 + daily_returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(daily_returns):
    """Worst peak-to-trough loss."""
    return drawdown_series(daily_returns).min()


def summary_table(returns_df, rf=0.0):
    """One row of metrics per column (strategy) of a daily-returns DataFrame."""
    rows = {}
    for name, r in returns_df.items():
        r = r.dropna()
        rows[name] = {
            "Annual Return": annual_return(r),
            "Annual Volatility": annual_volatility(r),
            "Sharpe Ratio": sharpe_ratio(r, rf),
            "Max Drawdown": max_drawdown(r),
            "Total Return": (1 + r).prod() - 1,
        }
    return pd.DataFrame(rows).T
