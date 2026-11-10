"""Single-underlying mixed stock/European-option portfolio risk demo.

Simulated real-world GBM scenarios with a user-supplied drift and volatility.
Options are revalued with Black--Scholes; flat volatility scenarios are a
scenario assumption, NOT a calibrated market volatility surface.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .pricing import black_scholes
from .hedging import tail_risk

@dataclass(frozen=True)
class Position:
    instrument: str   # stock, call or put
    quantity: float    # positive is long, negative is short
    strike: float = 105.0
    maturity: float = 1.0  # years remaining at time 0

    def __post_init__(self):
        if self.instrument not in ('stock','call','put'):
            raise ValueError('instrument must be stock, call or put')
        if not np.isfinite(self.quantity) or (self.instrument!='stock' and (self.strike<=0 or self.maturity<=0)):
            raise ValueError('Invalid position')


def portfolio_value(positions, spot, r=0.03, sigma=0.20, elapsed=0):
    """Mark-to-model portfolio value at a specified spot, elapsed time and vol."""
    spot=np.asarray(spot,dtype=float)
    if np.any(spot<=0) or sigma<0 or elapsed<0:
        raise ValueError('Invalid market scenarios')
    total=np.zeros_like(spot,dtype=float)
    for pos in positions:
        if pos.instrument=='stock':
            component=spot
        else:
            remaining=max(pos.maturity-elapsed,0.0)
            component=black_scholes(spot,pos.strike,remaining,r,sigma,pos.instrument)
        total=total+pos.quantity*component
    return float(total) if total.ndim==0 else total


def scenario_pnl(positions, spot0=100, horizon=1/12, mu=0.08,
                 realized_vol=0.20, pricing_vol=0.20, r=0.03,
                 n=20_000, seed=42, vol_shock=0.0):
    """Simulated horizon P&L under physical P; assumes same flat repricing vol for all options.

    No rebalancing, funding, fees or dividends. Returns raw scenario P&L.
    """
    if n<100 or horizon<=0 or realized_vol<0 or pricing_vol+vol_shock<0:
        raise ValueError('Invalid scenario parameters')
    rng=np.random.default_rng(seed)
    z=rng.standard_normal(n)
    future=spot0*np.exp((mu-0.5*realized_vol**2)*horizon+realized_vol*np.sqrt(horizon)*z)
    base=portfolio_value(positions,spot0,r,pricing_vol,0)
    future_value=portfolio_value(positions,future,r,pricing_vol+vol_shock,horizon)
    return {'pnl':future_value-base,'future_spot':future,'initial_value':base,
            'future_values':future_value}


def portfolio_risk(pnl, confidence=0.95):
    pnl=np.asarray(pnl,dtype=float)
    losses=-pnl
    risk=tail_risk(losses,confidence)
    return {'mean_pnl':float(np.mean(pnl)),'std_pnl':float(np.std(pnl,ddof=1)),
            'loss_var':risk['var'],'loss_es':risk['es'],
            'loss_probability':float(np.mean(pnl<0))}


def stress_grid(positions, spot0=100, r=0.03, pricing_vol=0.20,
                horizon=1/12, spot_shocks=(-0.20,-0.10,0,0.10,0.20),
                vol_shocks=(-0.05,0,0.05,0.10)):
    """Deterministic spot/volatility scenarios; amounts are P&L vs model value now."""
    start=portfolio_value(positions,spot0,r,pricing_vol,0)
    rows=[]
    for dv in vol_shocks:
        for ds in spot_shocks:
            vol=pricing_vol+dv
            if vol<0:
                raise ValueError('Negative stressed volatility')
            new=portfolio_value(positions,spot0*(1+ds),r,vol,horizon)
            rows.append({'Spot shock':ds,'Vol shock (pp)':dv*100,
                         'P&L':float(new-start)})
    return pd.DataFrame(rows)
