import numpy as np
from quant_engine.portfolio import Position,portfolio_value,scenario_pnl,portfolio_risk,stress_grid

positions=[Position('stock',5),Position('call',-3,105,1),Position('put',2,95,1)]

def test_portfolio_value_vectorization():
    s=np.array([80,100,120.])
    values=portfolio_value(positions,s)
    assert values.shape==(3,)
    assert np.all(np.isfinite(values))

def test_risk_metrics_on_simulated_scenarios():
    output=scenario_pnl(positions,n=3000,seed=12)
    risk=portfolio_risk(output['pnl'])
    assert risk['loss_es']>=risk['loss_var']
    assert 0<=risk['loss_probability']<=1
    assert output['pnl'].shape==(3000,)

def test_stress_grid_shape_and_nonempty():
    df=stress_grid(positions)
    assert len(df)==20
    assert np.all(np.isfinite(df['P&L']))

def test_stock_only_value():
    assert portfolio_value([Position('stock',3)],100)==300
