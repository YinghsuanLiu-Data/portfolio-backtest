"""Simple portfolio backtester with periodic rebalancing."""
import pandas as pd


def backtest_rebalanced(prices, target_weights, rebalance="ME", cost_bps=0.0):
    """Daily returns of a portfolio rebalanced back to ``target_weights``.

    Between rebalance dates the weights drift with prices (just like a real
    account). At the end of each period (``"ME"`` = month end) we trade back to
    the targets, paying ``cost_bps`` basis points on the traded amount.
    """
    target = pd.Series(target_weights, dtype=float).reindex(prices.columns).fillna(0)
    target = target / target.sum()
    asset_returns = prices.pct_change().dropna()

    # last trading day of each period -> rebalance at that day's close
    period_ends = set(asset_returns.groupby(asset_returns.index.to_period(rebalance[0]))
                      .tail(1).index)

    weights = target.copy()
    port_returns = []
    for date, r in asset_returns.iterrows():
        day_return = float((weights * r).sum())
        weights = weights * (1 + r) / (1 + day_return)        # drift with prices
        if date in period_ends:
            turnover = (weights - target).abs().sum()
            day_return -= turnover * cost_bps / 10_000
            weights = target.copy()
        port_returns.append(day_return)
    return pd.Series(port_returns, index=asset_returns.index, name="portfolio")


def buy_and_hold(prices, ticker):
    """Daily returns of holding one asset (e.g. the SPY benchmark)."""
    return prices[ticker].pct_change().dropna().rename(ticker)
