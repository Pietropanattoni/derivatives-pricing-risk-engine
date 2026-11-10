"""Streamlit dashboard for synthetic quant finance research examples."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from quant_engine import (black_scholes,greeks,implied_volatility,monte_carlo_european,
                          gbm_paths,compare_hedges,Position,scenario_pnl,portfolio_risk,stress_grid)
from quant_engine.hedging import hedging_sensitivity

st.set_page_config(page_title='Quant Finance Engine',page_icon='📈',layout='wide')
st.title('Derivatives Pricing & Portfolio Risk Engine')
st.caption('Research demonstration • All data are simulated or hypothetical • No live market connection')
with st.sidebar:
    st.header('Model inputs')
    S=st.number_input('Underlying spot',min_value=1.,value=100.,step=5.)
    K=st.number_input('Strike',min_value=1.,value=105.,step=5.)
    T=st.slider('Maturity (years)',0.1,3.0,1.0,0.1)
    r=st.slider('Risk-free rate (%)',-2.,12.,3.,0.5)/100
    sigma=st.slider('Pricing volatility (%)',1.,90.,20.,1.)/100
    seed=st.number_input('Random seed',min_value=0,value=42,step=1)

pricing,hedging,portfolio=st.tabs(['Option pricing','Delta hedging','Portfolio risk'])

with pricing:
    st.subheader('Black–Scholes and Monte Carlo')
    c1,c2,c3=st.columns(3)
    kind=c1.selectbox('Option type',['call','put'])
    market=c2.number_input('Hypothetical option market quote',min_value=0.01,value=9.50,step=0.25)
    n=c3.select_slider('Monte Carlo requested paths',[1000,10000,50000,100000],value=50000)
    bs=black_scholes(S,K,T,r,sigma,kind)
    mc=monte_carlo_european(S,K,T,r,sigma,kind,n=n,seed=int(seed))
    g=greeks(S,K,T,r,sigma,kind)
    a,b,c=st.columns(3)
    a.metric('BS model price',f'{bs:.4f}')
    b.metric('MC estimate',f'{mc["price"]:.4f}')
    c.metric('MC standard error',f'{mc["standard_error"]:.4f}')
    st.write(f'95% Monte Carlo CI: [{mc["ci_lower"]:.4f}, {mc["ci_upper"]:.4f}]')
    st.dataframe(pd.DataFrame([g]).T.rename(columns={0:'Greek value'}),use_container_width=True)
    try:
        iv=implied_volatility(market,S,K,T,r,kind)
        st.success(f'Implied volatility from the hypothetical quote: {iv:.2%}')
    except ValueError as exc:
        st.warning(f'IV not available: {exc}')
    grid=np.linspace(max(1.,S*.5),S*1.5,100)
    curve=black_scholes(grid,K,T,r,sigma,kind)
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(grid,curve);ax.set(xlabel='Underlying price',ylabel='Option model price',title='Option sensitivity to spot')
    ax.grid(alpha=.25);st.pyplot(fig);plt.close(fig)

with hedging:
    st.subheader('Discrete hedge of one short European call')
    st.caption('All frequencies use the same GBM market paths. Pricing and simulated volatility match in this experiment.')
    m=st.select_slider('Simulated paths',[1000,3000,5000],value=3000)
    weight=st.slider('Tail risk weight λ',0.,5.,1.,0.25)
    costs=st.multiselect('Stock trading cost (bps)',[0,5,10,20,40],default=[0,5,10,20])
    if costs:
        with st.spinner('Simulating hedging P&L...'):
            paths=gbm_paths(S,r,sigma,T,252,m,seed=int(seed))
            frame=compare_hedges(paths,K,T,r,sigma,costs=[x/10000 for x in sorted(costs)],risk_weight=weight)
        st.dataframe(frame.round(4),use_container_width=True,hide_index=True)
        fig,ax=plt.subplots(figsize=(8,4))
        for name,gp in frame.groupby('Strategy'):
            ax.plot(gp['Cost (bps)'],gp['Objective'],marker='o',label=name)
        ax.set(xlabel='Trading cost (bps)',ylabel='Mean loss + λ ES95',title='Hedging trade-off')
        ax.grid(alpha=.3);ax.legend();st.pyplot(fig);plt.close(fig)
        st.write('Preferred strategy by cost level (minimum chosen objective):')
        idx=frame.groupby('Cost (bps)')['Objective'].idxmin()
        st.dataframe(frame.loc[idx,['Cost (bps)','Strategy','Objective']].round(4),hide_index=True)
        st.write('Risk-weight sensitivity:')
        st.dataframe(hedging_sensitivity(frame),use_container_width=True)

with portfolio:
    st.subheader('Portfolio scenario risk and stress testing')
    st.caption('Single underlying; fixed example positions. Model values, not observed market prices.')
    st.write('Example positions: +5 stocks, −3 calls (strike 105), +2 puts (strike 95), each with 1 year remaining.')
    positions=[Position('stock',5),Position('call',-3,105,1),Position('put',2,95,1)]
    horizon=st.slider('Risk horizon (months)',1,6,1)/12
    physical_mu=st.slider('Physical-measure drift μ (%)',-20.,25.,8.,1.)/100
    realized_vol=st.slider('Scenario realized volatility (%)',1.,80.,20.,1.)/100
    vol_change=st.slider('Repricing vol shock (percentage points)',-10,20,0)/100
    if sigma+vol_change<=0:
        st.warning('Resulting repricing volatility must be positive.')
    else:
        scenario=scenario_pnl(positions,spot0=S,horizon=horizon,mu=physical_mu,
                              realized_vol=realized_vol,pricing_vol=sigma,r=r,n=20000,
                              seed=int(seed),vol_shock=vol_change)
        metrics=portfolio_risk(scenario['pnl'])
        c1,c2,c3=st.columns(3)
        c1.metric('Initial model value',f'{scenario["initial_value"]:.2f}')
        c2.metric('95% loss VaR',f'{metrics["loss_var"]:.2f}')
        c3.metric('95% loss ES',f'{metrics["loss_es"]:.2f}')
        st.write(f'Average scenario P&L {metrics["mean_pnl"]:.2f} | Scenario loss frequency {metrics["loss_probability"]:.1%}')
        fig,ax=plt.subplots(figsize=(8,3.6))
        ax.hist(scenario['pnl'],bins=70,alpha=.8)
        ax.axvline(-metrics['loss_var'],linestyle='--',label='5% P&L percentile')
        ax.set(xlabel='Horizon P&L',ylabel='Number of scenarios',title='Scenario-based P&L distribution')
        ax.legend();st.pyplot(fig);plt.close(fig)
        stress=stress_grid(positions,spot0=S,r=r,pricing_vol=sigma,horizon=horizon,
                           vol_shocks=tuple(x for x in (-.05,0,.05,.10) if sigma+x>=0))
        table=stress.pivot(index='Vol shock (pp)',columns='Spot shock',values='P&L')
        table.columns=[f'{x:+.0%}' for x in table.columns]
        st.write('Spot / volatility stress P&L')
        st.dataframe(table.round(2),use_container_width=True)

st.caption('Limitations: Black–Scholes constant volatility, no dividends, synthetic GBM scenarios, no multi-asset correlations, no liquidity model. See README.')
