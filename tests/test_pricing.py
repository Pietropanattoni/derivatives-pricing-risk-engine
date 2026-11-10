import numpy as np
import pytest
from quant_engine.pricing import black_scholes, greeks, implied_volatility, monte_carlo_european

S,K,T,r,v=100,105,1,.03,.20

def test_put_call_parity():
    assert np.isclose(black_scholes(S,K,T,r,v)-black_scholes(S,K,T,r,v,'put'),S-K*np.exp(-r*T),atol=1e-10)

def test_known_call_value():
    assert np.isclose(black_scholes(S,K,T,r,v),7.1281,atol=1e-4)

def test_expiry_and_deterministic_vol():
    assert black_scholes(120,105,0,r,.2)==15
    assert np.isclose(black_scholes(100,105,T,r,0),max(100-105*np.exp(-r*T),0))

def test_greeks_numerical():
    h=.001; g=greeks(S,K,T,r,v)
    c=lambda s,vol: black_scholes(s,K,T,r,vol)
    numeric_delta=(c(S+h,v)-c(S-h,v))/(2*h)
    numeric_gamma=(c(S+h,v)-2*c(S,v)+c(S-h,v))/h**2
    numeric_vega=(c(S,v+h)-c(S,v-h))/(2*h)
    assert np.isclose(g['delta'],numeric_delta,atol=1e-6)
    assert np.isclose(g['gamma'],numeric_gamma,atol=1e-5)
    assert np.isclose(g['vega'],numeric_vega,atol=1e-3)

def test_implied_volatility_inversion():
    for kind in ('call','put'):
        value=black_scholes(S,K,T,r,.25,kind)
        assert np.isclose(implied_volatility(value,S,K,T,r,kind),.25,atol=1e-9)

def test_bad_market_quote_rejected():
    with pytest.raises(ValueError):
        implied_volatility(500,S,K,T,r)

def test_mc_analytical_agreement():
    for kind in ('call','put'):
        result=monte_carlo_european(S,K,T,r,v,kind,n=80_000,seed=42)
        analytical=black_scholes(S,K,T,r,v,kind)
        assert abs(result['price']-analytical)<4*result['standard_error']
        assert result['ci_lower']<result['ci_upper']

def test_mc_antithetic_independent_pair_count():
    result=monte_carlo_european(S,K,T,r,v,n=10_000,antithetic=True)
    assert result['independent_observations']==5000
    assert result['n_paths']==10000

def test_vectorized_pricing():
    prices=black_scholes(np.array([90.,100.,110.]),K,T,r,v)
    assert prices.shape==(3,)
    assert np.all(np.diff(prices)>0)
