# Bitcoin Market Risk Analysis: Comparative Value-at-Risk Evidence from Hourly Data

## Abstract

This report evaluates whether a Gaussian Value at Risk benchmark is credible for Bitcoin when the asset is observed at hourly frequency from 2021-01-01 00:00:00 UTC to 2026-04-13 06:00:00 UTC. The empirical design compares a rolling Gaussian benchmark with four richer model families: a parametric Student-t specification, a jump-diffusion Monte Carlo engine, a feature-conditioned LSTM, and a variational autoencoder used for latent scenario generation. The evidence rejects the Gaussian baseline on both tail shape and backtest diagnostics, supports heavy-tailed and state-dependent modeling, and shows that LSTM cond. VaR provides the strongest one-hour conditional coverage result in the final-year backtest, while jump-diffusion remains the more conservative scenario engine under stress.

## Introduction

The central research question is whether a simple Gaussian VaR framework can describe Bitcoin downside risk at an hourly horizon, or whether the data require heavier tails, jump risk, and conditional machine-learning overlays. That question matters academically because Bitcoin combines large volatility clusters, event-driven discontinuities, and rapid regime changes that can break the assumptions behind textbook variance-scaling rules and thin-tailed parametric benchmarks (Jorion, 2007; McNeil, Frey, and Embrechts, 2015). The report therefore evaluates model credibility along three dimensions: distributional fit, out-of-sample tail calibration, and stress-period warning quality.

## Data

The dataset contains 46,279 cleaned hourly observations built from Binance BTC/USDT spot candles between 2021-01-01 00:00:00 UTC and 2026-04-13 06:00:00 UTC. Data cleaning filled 14 missing hours, removed 0 duplicate rows, and repaired 0 candidate flash-crash artefacts. BTC/USDT is used as a liquid USD proxy, which is operationally convenient for uninterrupted hourly sampling but should still be interpreted as a proxy benchmark rather than a pure fiat BTC/USD series.

| original_rows | cleaned_rows | missing_hours_filled | duplicate_rows_removed | flash_crash_repairs | start | end |
| --- | --- | --- | --- | --- | --- | --- |
| 46265 | 46279 | 14 | 0 | 0 | 2021-01-01 00:00:00 UTC | 2026-04-13 06:00:00 UTC |

![Price and volatility](figures/price_and_volatility.png)

## Methodology

The benchmark layer contains two rolling parametric baselines: a Gaussian VaR benchmark and a Student-t VaR benchmark, both estimated on the same recent return window so that the distributional assumption is the main moving part (Jorion, 2007; McNeil, Frey, and Embrechts, 2015). The structural scenario layer adds jump-diffusion Monte Carlo with 10,000 simulation paths and a tuned jump threshold in a Merton-style discontinuous-return setting (Merton, 1976). The neural layer adds a one-step LSTM with a Gaussian output head and a VAE that learns latent return-window structure for synthetic scenario generation (Hochreiter and Schmidhuber, 1997; Kingma and Welling, 2014). Feature engineering supplies log returns, rolling 24-hour and 168-hour volatility, RSI(14), MACD with signal line, and a rolling volume z-score to the conditional models. Chronological splitting is enforced throughout. The final-year backtest holdout spans 2025-04-13 06:00:00 UTC to 2026-04-13 06:00:00 UTC with 8,761 hourly observations. Student-t and jump-diffusion hyperparameters are tuned on a 30-day pre-backtest slice sampled every 6 hours, which yields 120 validation checkpoints. For the neural models, the final 20% of the pre-backtest sequences or rolling windows are reserved for validation, giving 7,455 LSTM validation sequences and 7,498 VAE validation windows. The LSTM is trained with AdamW, learning rate 0.001, batch size 256, and 8 epochs, with the final checkpoint chosen by minimum validation Gaussian negative log-likelihood. The VAE is trained with AdamW, learning rate 0.001, batch size 512, 20 epochs, and a reconstruction-plus-KL loss on rolling 24-hour windows. Validation scores in the tuning table therefore represent mean pinball loss for the parametric models, validation Gaussian NLL for the LSTM, and validation ELBO-style loss for the VAE.

### Method Overview

| Method | Model | Primary role | Main trade-off |
| --- | --- | --- | --- |
| A | Gaussian VaR | Thin-tailed rolling benchmark used to test whether a Gaussian assumption is adequate. | Transparent baseline, but structurally weak under heavy tails and volatility clustering. |
| B | Student-t VaR | Parametric heavy-tail benchmark using a rolling return window. | Transparent and fast, but only weakly conditional on current market state. |
| C | Jump-diffusion Monte Carlo | Scenario engine that combines continuous diffusion with discrete jump risk. | More realistic under stress, but sensitive to jump-calibration choices. |
| D | LSTM Gaussian head | Feature-conditioned one-step forecast of mean and volatility. | Adaptive to regime shifts, but harder to validate and explain. |
| E | VAE latent sampling | Generative model for synthetic return scenarios and latent tail structure. | Useful for scenario enrichment, but weaker as a standalone production VaR engine. |

### Tuned Model Settings

| model | design | best_parameters | selection_metric | validation_score |
| --- | --- | --- | --- | --- |
| Gaussian VaR | Rolling normal distribution benchmark | 90-day rolling window | Baseline reference |  |
| Student-t VaR | Parametric heavy-tail baseline | 90-day rolling window; df estimated from sample kurtosis | Mean pinball loss | 0.0003 |
| Jump-diffusion Monte Carlo | Diffusion plus calibrated jump scenario engine | 90-day rolling window; jump threshold 2.5 sigma | Mean pinball loss | 0.0003 |
| LSTM cond. VaR | Feature-conditioned recurrent forecaster with validation-calibrated tail | 72-hour look-back; 2 layers; 48 hidden units; dropout 0.10; calibrated tail alpha 0.0125; calibrated tail z -2.662 | Conditional-coverage-targeted fine-tune | 0.9624 |
| VAE latent VaR | Latent generative return model on rolling windows | 24-hour window; latent dimension 8 | Best validation ELBO | 0.5177 |

### Sample Split Summary

| block | start | end | observations | design_note |
| --- | --- | --- | --- | --- |
| Full cleaned sample | 2021-01-01 00:00:00 UTC | 2026-04-13 06:00:00 UTC | 46279 | Hourly cleaned candles used for descriptive statistics, figures, and feature construction. |
| Pre-backtest return history | 2021-01-01 01:00:00 UTC | 2025-04-13 05:00:00 UTC | 37517 | History available for model estimation before the final one-year holdout. |
| Tuning validation slice | 2025-03-14 06:00:00 UTC | 2025-04-13 00:00:00 UTC | 120 | Chronological validation checkpoints every 6 hours for Student-t and jump-diffusion tuning. |
| Backtest holdout | 2025-04-13 06:00:00 UTC | 2026-04-13 06:00:00 UTC | 8761 | Final one-year out-of-sample window used for Kupiec coverage testing. |
| LSTM fit sequences | 2021-01-11 00:00:00 UTC | 2024-06-06 14:00:00 UTC | 29823 | Chronological 72-hour feature sequences used for recurrent model fitting. |
| LSTM validation sequences | 2024-06-06 15:00:00 UTC | 2025-04-13 05:00:00 UTC | 7455 | Final 20% of pre-backtest sequence targets reserved for neural validation. |
| LSTM backtest sequences | 2025-04-13 06:00:00 UTC | 2026-04-13 06:00:00 UTC | 8761 | Out-of-sample sequence targets used for the one-step conditional backtest. |
| VAE train windows | 2021-01-01 01:00:00 UTC | 2024-06-04 19:00:00 UTC | 29996 | Rolling 24-hour windows used to estimate the latent generative model. |
| VAE validation windows | 2024-06-03 21:00:00 UTC | 2025-04-13 05:00:00 UTC | 7498 | Final 20% of pre-backtest rolling windows used for chronological latent-model validation. |

The downside tail metrics used in the report follow the standard left-tail definitions:

$$
CVaR_{\alpha} = \frac{1}{1-\alpha} \int_{\alpha}^{1} VaR_u \, du
$$

For unconditional coverage backtesting, the Kupiec proportion-of-failures statistic is:

$$
LR_{POF} = -2 \ln \left( \frac{(1-\alpha)^{T-x} \alpha^x}{(1-\hat{p})^{T-x} \hat{p}^x} \right), \qquad \hat{p} = \frac{x}{T}
$$

where $T$ is the number of forecasts, $x$ is the number of VaR violations, and $\hat{p}$ is the observed violation rate. The report complements this with Christoffersen-style independence and conditional-coverage diagnostics (Kupiec, 1995; Christoffersen, 1998).

## Statistical Diagnostics

The descriptive evidence rejects Gaussianity decisively. The Jarque-Bera statistic reaches 491079.15 with a near-zero p-value, while skewness is -0.207 and kurtosis is 18.955. Hourly returns range from -9.38% to 11.61%, and the ADF statistic of -30.208 rejects a unit root in the return series. Taken together, these diagnostics support heavy-tailed and conditional models rather than a thin-tailed homoskedastic baseline.

| observations | mean | std | skewness | kurtosis | jarque_bera_stat | jarque_bera_pvalue | adf_stat | adf_pvalue | adf_critical_1pct | adf_critical_5pct | adf_critical_10pct | min | max | p01 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46278 | 0 | 0.0064 | -0.2066 | 18.9551 | 491079.1521 | <0.0001 | -30.2084 | <0.0001 | -3.4305 | -2.8616 | -2.5668 | -0.0938 | 0.1161 | -0.0193 | 0.0186 |

![Return distribution](figures/return_distribution.png)

![Distribution diagnostics](figures/distribution_diagnostics.png)

## Model Results

Over the full sample, Bitcoin delivered a cumulative price change of 144.96%. In the current forecast snapshot, the one-hour VaR rankings differ materially by methodology, with the most conservative 99% estimate coming from Jump-diffusion MC at 1.60%. The Gaussian benchmark currently reports a one-hour VaR loss of 1.31%, versus 1.47% for the Student-t benchmark estimated on the same rolling history. The Student-t model remains the most interpretable heavy-tail baseline, the jump-diffusion engine provides the most stress-aware structural scenarios, and the LSTM contributes the most explicitly conditional one-step signal. Because the recurrent model is estimated as a one-step conditional forecaster, the 24-hour comparison table includes only models that directly produce multi-period distributions through aggregation or simulation.

| model | horizon_hours | VaR_return | CVaR_return | VaR_loss | CVaR_loss | mean_return | volatility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian VaR | 1 | -0.0131 | -0.015 | 0.0131 | 0.015 | -0.0001 | 0.0056 |
| Jump-diffusion MC | 1 | -0.0163 | -0.0281 | 0.0163 | 0.0281 | -0.0003 | 0.0066 |
| LSTM | 1 | -0.0116 | -0.0153 | 0.0116 | 0.0153 | -0.0009 | 0.0043 |
| Student-t param. | 1 | -0.0147 | -0.0194 | 0.0147 | 0.0194 | -0.0001 | 0.0043 |
| VAE | 1 | -0.0117 | -0.0144 | 0.0117 | 0.0144 | 0 | 0.0042 |
| Gaussian VaR | 24 | -0.0665 | -0.0757 | 0.0665 | 0.0757 | -0.0029 | 0.0273 |
| Jump-diffusion MC | 24 | -0.0829 | -0.096 | 0.0829 | 0.096 | -0.0046 | 0.0323 |
| Student-t param. | 24 | -0.0672 | -0.0779 | 0.0672 | 0.0779 | -0.0029 | 0.0272 |
| VAE | 24 | -0.0419 | -0.0486 | 0.0419 | 0.0486 | -0.0015 | 0.0174 |

![Monte Carlo paths](figures/monte_carlo_paths.png)

## Sensitivity Analysis

Sensitivity analysis confirms the expected widening of downside risk as confidence moves from 95% to 99%. At 99%, the largest one-hour VaR loss remains 1.60%. The Gaussian row in the confidence table provides the thin-tailed benchmark, the Student-t parameter sweep shows that lower degrees of freedom amplify tail losses in a nonlinear way, and the LSTM look-back sweep shows a trade-off between responsiveness and stability.

### Confidence-Level Comparison

| confidence_level_pct | model | VaR_loss | CVaR_loss | volatility |
| --- | --- | --- | --- | --- |
| 95 | Jump-diffusion MC | 0.0097 | 0.0143 | 0.0067 |
| 95 | Gaussian VaR | 0.0093 | 0.0116 | 0.0056 |
| 95 | Student-t param. | 0.0088 | 0.0126 | 0.0043 |
| 95 | LSTM | 0.008 | 0.0098 | 0.0043 |
| 95 | VAE | 0.0074 | 0.0101 | 0.0042 |
| 97.5 | Jump-diffusion MC | 0.0119 | 0.0179 | 0.0065 |
| 97.5 | Student-t param. | 0.0112 | 0.0153 | 0.0043 |
| 97.5 | Gaussian VaR | 0.0111 | 0.0132 | 0.0056 |
| 97.5 | LSTM | 0.0093 | 0.011 | 0.0043 |
| 97.5 | VAE | 0.0093 | 0.012 | 0.0042 |
| 99 | Jump-diffusion MC | 0.016 | 0.0284 | 0.0067 |
| 99 | Student-t param. | 0.0147 | 0.0194 | 0.0043 |
| 99 | Gaussian VaR | 0.0131 | 0.015 | 0.0056 |
| 99 | VAE | 0.0117 | 0.0144 | 0.0042 |
| 99 | LSTM | 0.0109 | 0.0124 | 0.0043 |

### Student-t Degrees-of-Freedom Impact

| degrees_of_freedom | VaR_return | VaR_loss | tail_ratio_to_sigma | is_estimated_setting |
| --- | --- | --- | --- | --- |
| 4.5 | -0.0148 | 0.0148 | 2.6506 | 0 |
| 5 | -0.0147 | 0.0147 | 2.6281 | 0 |
| 6 | -0.0144 | 0.0144 | 2.5876 | 0 |
| 8 | -0.0141 | 0.0141 | 2.5301 | 0 |
| 12 | -0.0138 | 0.0138 | 2.4691 | 0 |
| 20 | -0.0135 | 0.0135 | 2.4199 | 0 |

### LSTM Look-back Window Impact

| lookback_hours | best_validation_loss | sigma_tracking_mae | current_VaR_loss | current_sigma | validation_violation_rate | validation_kupiec_pvalue | validation_ind_pvalue | validation_cc_pvalue | tail_quantile_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | -4.6994 | 0.0038 | 0.014 | 0.0054 | 0.01 | 0.9675 | 0.0473 | 0.1397 | -2.2808 |
| 48 | -4.7665 | 0.0033 | 0.0123 | 0.0049 | 0.0101 | 0.9629 | 0.7852 | 0.9625 | -2.5731 |
| 72 | -4.7694 | 0.003 | 0.0124 | 0.0043 | 0.0101 | 0.9583 | 0.7857 | 0.9624 | -2.662 |

## Stress Testing

The stress section is designed as a two-window case-study exercise rather than an exhaustive event screen. Two pre-specified event windows are defined in code, and the event timestamp inside each window is selected algorithmically by either the worst one-hour loss or the largest absolute one-hour move; warnings are then evaluated over the preceding 24-hour horizon. Within that design, the FTX collapse window peaks at 2022-11-08 17:00:00 UTC with a realized one-hour loss of 5.18%, and the ETF whipsaw peaks at 2024-01-11 15:00:00 UTC with a realized absolute move of 3.75%. In both cases, the jump-diffusion engine issues the strongest warning profile and lands closest to the realized loss. At the same time, every one-hour VaR model understates the realized event loss by a wide margin, so the stress section should be interpreted as a warning-comparison exercise rather than a claim of exact crisis prediction.

| event | event_timestamp | selection_start | selection_end | selection_mode | warning_horizon_hours | event_scope | actual_return | actual_loss | student_t_VaR_loss | mc_VaR_loss | student_t_peak_warning_loss | mc_peak_warning_loss | student_t_peak_warning_hours_before_event | mc_peak_warning_hours_before_event | strongest_warning_model | closest_model_to_realized_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FTX Collapse | 2022-11-08 17:00:00 UTC | 2022-11-07T00:00:00+00:00 | 2022-11-10T23:00:00+00:00 | loss | 24 | pre-specified case study | -0.0518 | 0.0518 | 0.0145 | 0.0163 | 0.0144 | 0.0171 | 5 | 23 | Jump-diffusion MC | Jump-diffusion MC |
| ETF whipsaw | 2024-01-11 15:00:00 UTC | 2024-01-10T00:00:00+00:00 | 2024-01-12T23:00:00+00:00 | absolute | 24 | pre-specified case study | -0.0375 | 0.0375 | 0.0123 | 0.0129 | 0.0122 | 0.014 | 2 | 23 | Jump-diffusion MC | Jump-diffusion MC |

![Stress warning profiles](figures/stress_test_warnings.png)

## Time-Scale Scaling

The square-root-of-time rule is only partially reliable for Bitcoin. The scaled one-day 99% VaR is 9.43% versus an empirical one-day VaR of 8.74%, while the scaled ten-day VaR is 29.83% against an empirical ten-day VaR of 27.66%. Weak linear autocorrelation in raw returns coexists with persistent autocorrelation in absolute returns, which implies that volatility clustering rather than mean predictability is the main reason naive variance scaling can misstate long-horizon risk.

| horizon | sqrt_time_scaled_VaR | empirical_VaR | scaling_bias_pct |
| --- | --- | --- | --- |
| 1 day | 0.0943 | 0.0874 | 7.9691 |
| 10 days | 0.2983 | 0.2766 | 7.8213 |

| return_autocorr_lag1 | return_autocorr_lag24 | abs_return_autocorr_lag1 | abs_return_autocorr_lag24 | ljung_box_return_pvalue_lag24 | ljung_box_abs_return_pvalue_lag24 |
| --- | --- | --- | --- | --- | --- |
| -0.0047 | -0.0262 | 0.2847 | 0.1904 | <0.0001 | <0.0001 |

## Practical Implications

For a hypothetical $1 million BTC spot position, the largest 24-hour VaR capital estimate in the current snapshot is $82,946.43. These are market-risk numbers rather than liquidity-adjusted exit costs. For committee reporting, the jump-diffusion estimates are the more conservative baseline, while the LSTM is more appropriate as an intraday surveillance overlay.

| position_notional_usd | model | horizon_hours | VaR_capital_usd | CVaR_capital_usd |
| --- | --- | --- | --- | --- |
| 1000000 | Jump-diffusion MC | 1 | 16276.37 | 28112.64 |
| 1000000 | Student-t param. | 1 | 14655.86 | 19350.76 |
| 1000000 | Gaussian VaR | 1 | 13094.56 | 14984.39 |
| 1000000 | VAE | 1 | 11732.93 | 14385.96 |
| 1000000 | LSTM | 1 | 11576.58 | 15282.24 |
| 1000000 | Jump-diffusion MC | 24 | 82946.43 | 96006.01 |
| 1000000 | Student-t param. | 24 | 67183.15 | 77946.75 |
| 1000000 | Gaussian VaR | 24 | 66455.78 | 75714.02 |
| 1000000 | VAE | 24 | 41882.46 | 48609.76 |

## Backtesting

The last-year backtest evaluates 8,761 one-hour forecasts against a 1% tail target using both Kupiec's unconditional coverage test and Christoffersen's independence and conditional-coverage diagnostics (Kupiec, 1995; Christoffersen, 1998). LSTM cond. VaR delivers the strongest joint result with a conditional-coverage p-value of 0.078, while the Gaussian benchmark posts a Kupiec p-value of 0.000 and a conditional-coverage p-value of 0.000. At the 5% significance level, Gaussian VaR, Student-t VaR, Jump-diffusion MC, and VAE latent VaR are rejected by the conditional-coverage test, whereas LSTM cond. VaR is not rejected. That pattern suggests that conditional adaptation improves tail calibration on this holdout window, but it is still evidence about this sample path rather than proof that one specification dominates in every regime.

| model | observations | violations | expected_violation_rate | observed_violation_rate | kupiec_pvalue | christoffersen_ind_pvalue | christoffersen_cc_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian VaR | 8761 | 184 | 0.01 | 0.021 | <0.0001 | <0.0001 | <0.0001 |
| Student-t VaR | 8761 | 129 | 0.01 | 0.0147 | <0.0001 | 0.0001 | <0.0001 |
| Jump-diffusion MC | 8761 | 110 | 0.01 | 0.0126 | 0.0208 | 0.0005 | 0.0002 |
| LSTM cond. VaR | 8761 | 72 | 0.01 | 0.0082 | 0.0837 | 0.1463 | 0.078 |
| VAE latent VaR | 8761 | 133 | 0.01 | 0.0152 | <0.0001 | 0.0051 | <0.0001 |

![VaR backtest](figures/var_backtest.png)

![LSTM volatility forecast](figures/lstm_volatility_forecast.png)

![VAE generated returns](figures/vae_generated_returns.png)

## Limitations

The project has several explicit limitations. The benchmark is a proxy BTC/USDT series rather than a fiat BTC/USD market, the report relies on hourly candles rather than intrahour depth, and the backtesting section still evaluates only one realized holdout year even after adding conditional-coverage diagnostics. The neural models improve adaptability but remain harder to interpret than the parametric baseline, and the validation design is chronological rather than full rolling-origin cross-validation across multiple non-overlapping regimes. Finally, the reported VaR and CVaR numbers are market-risk statistics rather than liquidity-adjusted liquidation losses.

## Discussion

Two empirical themes dominate the evidence. First, Bitcoin exhibits persistent volatility clustering, so state dependence matters even when raw returns themselves are weakly autocorrelated (Engle, 1982). Second, the downside tail is too heavy for a Gaussian benchmark to be credible: the Gaussian row understates tail risk relative to the matched Student-t benchmark and is weaker than Student-t VaR on the final-year backtest. Those facts explain why the project benefits from layering models instead of selecting a single winner for every use case.

## Conclusion

The main conclusion is that Gaussian hourly VaR is not an adequate benchmark for Bitcoin. Heavy-tailed parametric modeling, jump-aware simulation, and conditional neural forecasting all add economically meaningful information relative to a thin-tailed variance-only baseline. In this project, the jump-diffusion engine is the strongest conservative scenario benchmark, while the LSTM is the strongest one-hour conditional overlay on the final holdout year under the current conditional-coverage diagnostics.

## References

1. Jorion, Philippe. 2007. Value at Risk: The New Benchmark for Managing Financial Risk. 3rd ed. New York: McGraw-Hill.
2. McNeil, Alexander J., Rudiger Frey, and Paul Embrechts. 2015. Quantitative Risk Management: Concepts, Techniques and Tools. Revised ed. Princeton: Princeton University Press.
3. Kupiec, Paul H. 1995. Techniques for Verifying the Accuracy of Risk Measurement Models. Journal of Derivatives 3(2): 73-84.
4. Christoffersen, Peter F. 1998. Evaluating Interval Forecasts. International Economic Review 39(4): 841-862.
5. Merton, Robert C. 1976. Option Pricing When Underlying Stock Returns Are Discontinuous. Journal of Financial Economics 3(1-2): 125-144.
6. Engle, Robert F. 1982. Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica 50(4): 987-1007.
7. Hochreiter, Sepp, and Jurgen Schmidhuber. 1997. Long Short-Term Memory. Neural Computation 9(8): 1735-1780.
8. Kingma, Diederik P., and Max Welling. 2014. Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations.

## Appendix

The appendix figures support the report's main claims visually. The volatility-clustering scatter plot shows persistence in absolute returns, the normal Q-Q panel shows substantial tail deviation from Gaussian behavior, the backtest figure shows the relative responsiveness of each VaR series, and the generated-return panels illustrate how the neural models differ in purpose from the parametric baselines.
