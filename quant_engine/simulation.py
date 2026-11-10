"""Exact-grid simulation of a geometric Brownian motion."""
import numpy as np


def gbm_paths(S0=100.0, mu=0.03, sigma=0.20, T=1.0, steps=252, paths=5000, seed=42):
    if S0<=0 or T<=0 or sigma<0 or steps<1 or paths<1:
        raise ValueError('Invalid GBM parameters')
    dt=T/steps
    rng=np.random.default_rng(seed)
    z=rng.standard_normal((steps,paths))
    log_increments=(mu-0.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*z
    out=np.empty((steps+1,paths))
    out[0]=S0
    out[1:]=S0*np.exp(np.cumsum(log_increments,axis=0))
    return out
