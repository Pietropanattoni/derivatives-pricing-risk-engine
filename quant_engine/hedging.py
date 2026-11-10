"""Self-financing (before costs) discrete delta hedge of a short European call.

Premium is sold at BS fair value, not marked up for transaction costs.
Transaction cost is a proportional fraction of traded stock notional;
we include initial stock purchase, rebalances and terminal stock liquidation.
"""
import numpy as np
import pandas as pd
from .pricing import black_scholes, greeks
from .simulation import gbm_paths


def delta_hedge(stock_paths, K=105, T=1.0, r=0.03, sigma=0.20,
                rebalance_every=1, transaction_cost=0.0):
    stock_paths=np.asarray(stock_paths,dtype=float)
    if stock_paths.ndim != 2 or stock_paths.shape[0]<2 or np.any(stock_paths<=0):
        raise ValueError('stock_paths must be a positive (steps+1, paths) matrix')
    if not isinstance(rebalance_every,int) or rebalance_every<1 or transaction_cost<0:
        raise ValueError('Invalid hedge frequency/cost')
    steps=stock_paths.shape[0]-1
    dt=T/steps
    S0=stock_paths[0]
    premium=black_scholes(S0,K,T,r,sigma,'call')
    shares=greeks(S0,K,T,r,sigma,'call')['delta']
    cash=premium-shares*S0-transaction_cost*np.abs(shares)*S0
    cumulative_cost=transaction_cost*np.abs(shares)*S0
    for i in range(1,steps+1):
        cash=cash*np.exp(r*dt)
        if i==steps or i % rebalance_every != 0:
            continue
        remaining=T-i*dt
        target=greeks(stock_paths[i],K,remaining,r,sigma,'call')['delta']
        trade=target-shares
        cost=transaction_cost*np.abs(trade)*stock_paths[i]
        cash=cash-trade*stock_paths[i]-cost
        cumulative_cost=cumulative_cost+cost
        shares=target
    terminal_cost=transaction_cost*np.abs(shares)*stock_paths[-1]
    cumulative_cost=cumulative_cost+terminal_cost
    payoff=np.maximum(stock_paths[-1]-K,0)
    pnl=cash+shares*stock_paths[-1]-terminal_cost-payoff
    return {'pnl':pnl,'transaction_costs_paid':cumulative_cost}


def tail_risk(losses, confidence=0.95):
    """Empirical VaR and ES; ES uses the upper (1-confidence) fraction of sorted losses.

    Fractional tail weights ensure exactly the requested tail fraction,
    even if sample size is not divisible by the tail probability.
    Losses may be negative if every scenario makes a profit.
    """
    x=np.sort(np.asarray(losses,dtype=float))[::-1]
    if x.ndim!=1 or x.size<2 or not 0<confidence<1 or not np.all(np.isfinite(x)):
        raise ValueError('Invalid losses or confidence')
    var=float(np.quantile(x,confidence))
    k=(1-confidence)*x.size
    whole=int(np.floor(k))
    fraction=k-whole
    es=(np.sum(x[:whole])+(fraction*x[whole] if fraction>1e-12 else 0.0))/k
    return {'var':var,'es':float(es)}


def compare_hedges(stock_paths, K=105, T=1, r=0.03, sigma=0.20,
                   costs=(0,0.0005,0.001,0.002),
                   frequencies=None, risk_weight=1.0):
    if frequencies is None:
        frequencies={'Monthly':21,'Weekly':5,'Daily':1}
    rows=[]
    for c in costs:
        for label,days in frequencies.items():
            result=delta_hedge(stock_paths,K,T,r,sigma,days,c)
            pnl=result['pnl']
            risk=tail_risk(-pnl)
            mean_loss=float(-np.mean(pnl))
            rows.append({'Cost (bps)':c*10000,'Strategy':label,'Rebalance (steps)':days,
                         'Mean P&L':float(np.mean(pnl)),'P&L Std':float(np.std(pnl,ddof=1)),
                         'Mean abs P&L':float(np.mean(np.abs(pnl))),
                         'VaR 95% loss':risk['var'],'ES 95% loss':risk['es'],
                         'Mean transaction cost':float(np.mean(result['transaction_costs_paid'])),
                         'Objective':mean_loss+risk_weight*risk['es']})
    return pd.DataFrame(rows)


def hedging_sensitivity(df, risk_weights=(0,0.25,0.5,1,2,5)):
    rows=[]
    for weight in risk_weights:
        for _, row in df.iterrows():
            rows.append({'Lambda':weight,'Cost (bps)':row['Cost (bps)'],
                         'Strategy':row['Strategy'],
                         'Objective':-row['Mean P&L']+weight*row['ES 95% loss']})
    all_scores=pd.DataFrame(rows)
    indices=all_scores.groupby(['Lambda','Cost (bps)'])['Objective'].idxmin()
    best=all_scores.loc[indices]
    return best.pivot(index='Lambda',columns='Cost (bps)',values='Strategy')
