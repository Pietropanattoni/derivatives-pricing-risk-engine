"""Complete reproducible numerical demo, with charts written under outputs/."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from quant_engine import (black_scholes,greeks,implied_volatility,monte_carlo_european,
                          gbm_paths,compare_hedges,Position,scenario_pnl,portfolio_risk,stress_grid)
from quant_engine.hedging import hedging_sensitivity

out=Path('outputs'); out.mkdir(exist_ok=True)
S,K,T,r,sigma=100,105,1,0.03,0.20
call=black_scholes(S,K,T,r,sigma)
put=black_scholes(S,K,T,r,sigma,'put')
mc=monte_carlo_european(S,K,T,r,sigma,n=100_000,seed=42)
market_price=9.50  # Synthetic quotation used for demonstration
iv=implied_volatility(market_price,S,K,T,r)
print('\n=== OPTION PRICING (SYNTHETIC INPUTS) ===')
print(f'Call BS {call:.4f} | Put BS {put:.4f} | MC {mc["price"]:.4f} ± {1.96*mc["standard_error"]:.4f} (95% half-width)')
print(f'Greeks {greeks(S,K,T,r,sigma)}')
print(f'Synthetic market quote {market_price:.4f} -> implied vol {iv:.4%}')

paths=gbm_paths(S,r,sigma,T,252,5000,seed=42)
df=compare_hedges(paths,K,T,r,sigma)
df.to_csv(out/'hedging_comparison.csv',index=False)
print('\n=== HEDGING COMPARISON ===')
print(df.round(4).to_string(index=False))
print('\n=== BEST REBALANCING STRATEGY (COST vs RISK WEIGHT) ===')
print(hedging_sensitivity(df).to_string())
fig,ax=plt.subplots(figsize=(8,5))
for name,group in df.groupby('Strategy'):
    ax.plot(group['Cost (bps)'],group['Objective'],marker='o',label=name)
ax.set(xlabel='Cost (bps)',ylabel='Mean loss + ES95 (lambda=1)',title='Discrete delta hedging: risk-cost trade-off')
ax.grid(alpha=.3); ax.legend();fig.tight_layout();fig.savefig(out/'hedging_tradeoff.png',dpi=160);plt.close(fig)

# One underlying, multiple positions; an intentionally illustrative portfolio.
positions=[Position('stock',5),Position('call',-3,105,1),Position('put',2,95,1)]
scenario=scenario_pnl(positions,spot0=S,horizon=1/12,mu=.08,
                       realized_vol=.20,pricing_vol=.20,n=20000,seed=42)
metrics=portfolio_risk(scenario['pnl'])
print('\n=== PORTFOLIO RISK (PHYSICAL-DRIFT SCENARIOS; ILLUSTRATIVE) ===')
print(f'Initial portfolio value: {scenario["initial_value"]:.4f}')
print({k:round(v,4) for k,v in metrics.items()})
fig,ax=plt.subplots(figsize=(8,5));ax.hist(scenario['pnl'],bins=70,alpha=.8)
ax.axvline(-metrics['loss_var'],linestyle='--',label='5% P&L percentile')
ax.set(xlabel='One-month portfolio P&L',ylabel='Count',title='Portfolio model-scenario distribution')
ax.legend();fig.tight_layout();fig.savefig(out/'portfolio_pnl.png',dpi=160);plt.close(fig)
stress=stress_grid(positions,spot0=S,pricing_vol=.20,horizon=1/12)
stress.to_csv(out/'stress_scenarios.csv',index=False)
print('\n=== SPOT / VOLATILITY STRESS TEST ===')
print(stress.pivot(index='Vol shock (pp)',columns='Spot shock',values='P&L').round(2).to_string())
print('\nCharts and CSV reports saved to outputs/.')
