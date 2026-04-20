# Report Upgrade TODO

## Completed in this implementation
- [x] Reframed the generated report into an academic paper structure with abstract, introduction, methodology, discussion, conclusion, and references.
- [x] Added a computed sample split summary so the report states the chronological train, validation, and backtest design explicitly.
- [x] Generated a synchronized TeX paper and repository TODO artifact from the Python pipeline outputs.
- [x] Expanded the narrative around model failures, stress-test interpretation, and limitations for university-style reporting.

## Recommended next extensions
- [ ] Add Christoffersen conditional coverage tests to complement the existing Kupiec coverage test.
- [ ] Extend the neural models to direct multi-step forecasting so the 24-hour comparison includes a conditional recurrent benchmark.
- [ ] Introduce liquidity-adjusted VaR or slippage-aware stress testing for execution-sensitive capital analysis.
- [ ] Replace the proxy benchmark with a fiat BTC/USD series or an explicit basis adjustment workflow.
