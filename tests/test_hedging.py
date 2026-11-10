import numpy as np
from quant_engine.simulation import gbm_paths
from quant_engine.hedging import delta_hedge, compare_hedges, hedging_sensitivity, tail_risk

def test_hedge_cost_reduces_every_path_pnl():
    paths=gbm_paths(paths=300,steps=63,seed=123)
    no_cost=delta_hedge(paths,rebalance_every=5,transaction_cost=0)
    with_cost=delta_hedge(paths,rebalance_every=5,transaction_cost=.001)
    assert np.all(with_cost['pnl']<=no_cost['pnl']+1e-10)
    assert np.all(with_cost['transaction_costs_paid']>0)

def test_mean_zero_without_cost_in_risk_neutral_model():
    paths=gbm_paths(paths=10000,steps=252,seed=42)
    result=delta_hedge(paths,rebalance_every=1)['pnl']
    assert abs(np.mean(result))<.08

def test_hedging_frequency_dispersion():
    paths=gbm_paths(paths=5000,steps=252,seed=42)
    daily=delta_hedge(paths,rebalance_every=1)['pnl']
    monthly=delta_hedge(paths,rebalance_every=21)['pnl']
    assert np.std(daily)<np.std(monthly)

def test_compare_and_sensitivity_shape():
    paths=gbm_paths(paths=300,steps=63,seed=9)
    df=compare_hedges(paths,costs=[0,.001],frequencies={'Daily':1,'Monthly':21})
    table=hedging_sensitivity(df,risk_weights=[0,1])
    assert len(df)==4
    assert table.shape==(2,2)

def test_tail_risk_with_fractional_tail():
    losses=np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
    v=tail_risk(losses,.8)
    assert np.isclose(v['es'],8.5)
    assert v['es']>=v['var']
