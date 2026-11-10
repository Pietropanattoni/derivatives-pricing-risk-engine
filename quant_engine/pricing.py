"""European option pricing under constant-volatility Black--Scholes assumptions.

No dividends; continuously compounded risk-free rate. Prices and Greeks may be
numpy arrays, except implied-volatility solver, which accepts scalar inputs.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _validate(S, K, tau, sigma):
    S, K, tau, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float), np.asarray(K, dtype=float),
        np.asarray(tau, dtype=float), np.asarray(sigma, dtype=float))
    if np.any(~np.isfinite(S)) or np.any(~np.isfinite(K)) or np.any(~np.isfinite(tau)) or np.any(~np.isfinite(sigma)):
        raise ValueError('Inputs must be finite')
    if np.any(S <= 0) or np.any(K <= 0) or np.any(tau < 0) or np.any(sigma < 0):
        raise ValueError('S and K must be positive; tau and sigma nonnegative')
    return S, K, tau, sigma


def black_scholes(S, K, tau, r, sigma, kind='call'):
    """European call/put price. Handles zero maturity and zero volatility."""
    if kind not in ('call', 'put'):
        raise ValueError("kind must be 'call' or 'put'")
    S, K, tau, sigma = _validate(S, K, tau, sigma)
    if not np.isfinite(r):
        raise ValueError('r must be finite')
    forward_discounted = S - K * np.exp(-r * tau)
    deterministic = np.maximum(forward_discounted, 0) if kind == 'call' else np.maximum(-forward_discounted, 0)
    active = (tau > 0) & (sigma > 0)
    # Safe placeholders avoid numerical warnings for inactive elements.
    t = np.where(active, tau, 1.0)
    v = np.where(active, sigma, 1.0)
    d1 = (np.log(S / K) + (r + 0.5 * v*v) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if kind == 'call':
        regular = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:
        regular = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    result = np.where(active, regular, deterministic)
    return float(result) if result.ndim == 0 else result


def greeks(S, K, tau, r, sigma, kind='call'):
    """Analytic Greeks for tau>0 and sigma>0. Vega is per unit volatility (1.0).
    Theta is calendar-time derivative (per year), not time-to-maturity derivative.
    """
    if kind not in ('call','put'):
        raise ValueError('Invalid option kind')
    S, K, tau, sigma = _validate(S,K,tau,sigma)
    if np.any(tau <= 0) or np.any(sigma <= 0):
        raise ValueError('Greeks require positive remaining maturity and volatility')
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    pdf = norm.pdf(d1)
    gamma = pdf/(S*sigma*np.sqrt(tau))
    vega = S*pdf*np.sqrt(tau)
    common_theta = -S*pdf*sigma/(2*np.sqrt(tau))
    if kind == 'call':
        delta = norm.cdf(d1)
        theta = common_theta - r*K*np.exp(-r*tau)*norm.cdf(d2)
        rho = K*tau*np.exp(-r*tau)*norm.cdf(d2)
    else:
        delta = norm.cdf(d1)-1
        theta = common_theta + r*K*np.exp(-r*tau)*norm.cdf(-d2)
        rho = -K*tau*np.exp(-r*tau)*norm.cdf(-d2)
    values = dict(delta=delta,gamma=gamma,vega=vega,theta=theta,rho=rho)
    return {k: float(x) if x.ndim == 0 else x for k,x in values.items()}


def implied_volatility(market_price, S, K, tau, r, kind='call', max_sigma=10.0):
    """Invert BS with Brent. Strict no-arbitrage bounds required (finite positive IV)."""
    S,K,tau,market_price = map(float,(S,K,tau,market_price))
    if kind not in ('call','put') or S<=0 or K<=0 or tau<=0 or not all(map(np.isfinite,(S,K,tau,r,market_price))):
        raise ValueError('Invalid implied volatility inputs')
    discounted = K*np.exp(-r*tau)
    lo = max(S-discounted,0.0) if kind=='call' else max(discounted-S,0.0)
    hi = S if kind=='call' else discounted
    if not lo < market_price < hi:
        raise ValueError(f'Price must be strictly inside no-arbitrage bounds ({lo:.6f}, {hi:.6f})')
    objective = lambda vol: black_scholes(S,K,tau,r,vol,kind)-market_price
    if objective(max_sigma) < 0:
        raise ValueError('IV exceeds configured search range')
    return float(brentq(objective, 0.0, max_sigma, xtol=1e-12))


def monte_carlo_european(S, K, tau, r, sigma, kind='call', n=100_000, seed=42, antithetic=True):
    """Risk-neutral terminal-price simulation with normal 95% confidence interval.

    With antithetic variates, each observation is a *pair-averaged discounted
    payoff*, and the standard error is computed over independent pairs.
    """
    if not isinstance(n,int) or n < 100:
        raise ValueError('n must be an integer >= 100')
    if tau<=0 or sigma<=0 or S<=0 or K<=0 or kind not in ('call','put'):
        raise ValueError('Positive S/K/tau/sigma and valid kind required')
    rng = np.random.default_rng(seed)
    if antithetic:
        pairs = (n+1)//2
        z = rng.standard_normal(pairs)
        def discounted_payoff(z_vals):
            st = S*np.exp((r-0.5*sigma*sigma)*tau + sigma*np.sqrt(tau)*z_vals)
            return np.exp(-r*tau)*(np.maximum(st-K,0) if kind=='call' else np.maximum(K-st,0))
        x = (discounted_payoff(z)+discounted_payoff(-z))/2
        effective_paths = pairs*2
    else:
        z = rng.standard_normal(n)
        st = S*np.exp((r-0.5*sigma*sigma)*tau + sigma*np.sqrt(tau)*z)
        x = np.exp(-r*tau)*(np.maximum(st-K,0) if kind=='call' else np.maximum(K-st,0))
        effective_paths = n
    price = float(np.mean(x))
    se = float(np.std(x,ddof=1)/np.sqrt(x.size))
    return {'price':price,'standard_error':se,'ci_lower':price-1.96*se,
            'ci_upper':price+1.96*se,'n_paths':effective_paths,
            'independent_observations':int(x.size),'antithetic':antithetic}
