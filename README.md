# Derivatives Pricing & Portfolio Risk Engine

A **reproducible, educational Python research toolkit** for derivatives pricing and scenario-based risk analysis. Built around the Black--Scholes model, European option valuation, implied volatility, Monte Carlo, a discrete delta hedge, and a single-underlying stock/options portfolio.

> **Data integrity:** All prices, option quotes and stochastic scenarios shipped here are **synthetic**. The sample quote of 9.50 is illustrative, not a real market quote. No live market-data connection is claimed. This is a model demonstration, not an investment strategy or trading advice.

## Features

| Component | Functionality |
|---|---|
| `quant_engine.pricing` | Black--Scholes call/put, all five standard Greeks, implied volatility via Brent, antithetic Monte Carlo with confidence interval |
| `quant_engine.simulation` | Exact-step GBM simulation |
| `quant_engine.hedging` | Discrete delta hedge of a short call with financing and proportional trading costs; compare monthly/weekly/daily hedging |
| `quant_engine.portfolio` | Mixed long/short stocks, calls and puts; one-month scenario P&L, loss VaR, Expected Shortfall and spot/vol stress grid |
| `app.py` | Optional Streamlit interactive analytics dashboard |
| `tests/` | Automated correctness, numeric sensitivity and hedging checks |

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.run_analysis
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`). Run commands **from the repository root**. The scripts generate CSV files and PNG charts in `outputs/`.

## Financial methodology

**Pricing.** The no-dividend stock follows `dS = r S dt + sigma S dW_Q` under the risk-neutral measure. The European call price is `S N(d1) - K exp(-r T) N(d2)`. Implied volatility numerically inverts the call/put pricing function given a market quotation.

**Monte Carlo.** Terminal prices are simulated exactly for the GBM. Antithetic normals are paired and *each pair* is treated as one independent observation for the standard error of the mean payoff. A normal-approximation 95% confidence interval is reported. Black--Scholes is an analytic benchmark under the same model.

**Hedging.** One short call is sold at its BS theoretical price and delta-hedged with long stock, financed by a cash account at rate `r`. Shares are adjusted at chosen discrete trading intervals, with proportional trading costs on initial purchases, rebalances and terminal liquidation. Final P&L includes the option payoff. Comparisons use **identical simulated paths** for all strategies. `Objective = expected loss + lambda * ES95(loss)`. This criterion is deliberately illustrative and does not constitute a universal trading optimum.

**Portfolio risk.** Under the **physical** GBM, a configurable `mu` and realized volatility produce horizon stock-price scenarios. Options are revalued under Black--Scholes with a flat volatility assumption (plus optional fixed volatility shock). `P&L = horizon model value - initial model value`. VaR95 and ES95 refer to the **loss distribution**: the 95th loss percentile and mean of the worst 5% of losses. The stress grid explicitly shocks spot and implied volatility.

## Assumptions and limitations

- European options on a **single, non-dividend-paying stock**, constant rates/volatilities; no stochastic volatility, jumps or discrete dividends.
- The simplified GBM may underestimate real-world tails; stress tests are scenarios, not calibrated forecasts.
- Transaction costs are proportional to stock notional; no spread dynamics, option execution fees, market impact, constraints or margin.
- One underlying and a flat repricing volatility shared by all options. Not a multi-asset risk system or a fitted volatility surface.
- Risk simulations omit financing charges for the unhedged portfolio and include no portfolio rebalancing over the risk horizon.
- Five-seed robustness is an exploratory check and no claim of universal strategy superiority should be inferred.

## Reproducibility and validation

`python -m pytest -q` checks put-call parity, numerical Greeks, IV inversion, Monte Carlo statistical agreement with the analytic price, deterministic limits, risk metrics, and the monotonic effect of trading costs for identical hedges. Reports are generated using fixed seeds. The numbers reported in the examples are **simulation outputs** and should not be presented as realized trading performance.

## Project structure

```text
quant-finance-engine/
  quant_engine/
    __init__.py
    pricing.py
    simulation.py
    hedging.py
    portfolio.py
  examples/
    __init__.py
    run_analysis.py
  tests/
    test_pricing.py
    test_hedging.py
    test_portfolio.py
  app.py
  requirements.txt
  README.md
  .gitignore
  outputs/           # generated reports
```

## Suggested next extensions

Calibration to timestamped bid/ask option chains, a dividend-aware model, historical vs implied volatility comparison, volatility surfaces, multi-asset correlated scenarios, realized market frictions and CI automation.

## Attribution and ownership

This repository was assembled with AI assistance. Review and understand the methodology, inspect the tests, and adapt the work before describing it as independent implementation in an interview. No proprietary or real trading data are included.
