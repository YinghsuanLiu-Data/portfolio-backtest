"""Mean-variance (Markowitz) optimisation with scipy."""
import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252


def portfolio_stats(weights, mean_returns, cov_matrix):
    """Annualised expected return and volatility of a weight vector."""
    ret = float(np.dot(weights, mean_returns)) * TRADING_DAYS
    vol = float(np.sqrt(weights @ cov_matrix @ weights)) * np.sqrt(TRADING_DAYS)
    return ret, vol


def _constraints_and_bounds(n, max_weight):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # fully invested
    bounds = [(0.0, max_weight)] * n                                 # long-only, capped
    return constraints, bounds


def max_sharpe_weights(daily_returns, rf=0.0, max_weight=1.0):
    """Weights that maximise the Sharpe ratio on the given (training) returns."""
    mean, cov = daily_returns.mean().values, daily_returns.cov().values
    n = len(mean)

    def neg_sharpe(w):
        ret, vol = portfolio_stats(w, mean, cov)
        return -(ret - rf) / vol

    constraints, bounds = _constraints_and_bounds(n, max_weight)
    result = minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x.clip(0) / result.x.clip(0).sum()


def min_volatility_for_return(daily_returns, target_return, max_weight=1.0):
    """Lowest-volatility weights that reach an annual ``target_return``."""
    mean, cov = daily_returns.mean().values, daily_returns.cov().values
    n = len(mean)
    constraints, bounds = _constraints_and_bounds(n, max_weight)
    constraints.append({"type": "eq",
                        "fun": lambda w: portfolio_stats(w, mean, cov)[0] - target_return})
    result = minimize(lambda w: portfolio_stats(w, mean, cov)[1], np.full(n, 1 / n),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None


def efficient_frontier(daily_returns, n_points=40, max_weight=1.0):
    """List of (volatility, return) points on the efficient frontier."""
    mean = daily_returns.mean().values * TRADING_DAYS
    cov = daily_returns.cov().values
    # with a weight cap the max reachable return is below the best single asset
    lo, hi = mean.min(), mean.max()
    points = []
    for target in np.linspace(lo, hi, n_points):
        w = min_volatility_for_return(daily_returns, target, max_weight)
        if w is not None:
            ret, vol = portfolio_stats(w, daily_returns.mean().values, cov)
            points.append((vol, ret))
    return points


def random_portfolios(daily_returns, n=5000, seed=42):
    """Random long-only portfolios, for the frontier scatter plot."""
    rng = np.random.default_rng(seed)
    mean, cov = daily_returns.mean().values, daily_returns.cov().values
    weights = rng.dirichlet(np.ones(len(mean)), size=n)
    stats = np.array([portfolio_stats(w, mean, cov) for w in weights])
    return weights, stats[:, 0], stats[:, 1]
