## Bitcoin Market Risk Analysis

This project implements an end-to-end Bitcoin market risk workflow in Python using `uv`.

It covers:

- Hourly BTC/USDT data engineering from 2021 onward.
- Statistical diagnostics for non-normality and stationarity.
- Parametric Student-t VaR.
- 10,000-path jump diffusion Monte Carlo VaR.
- PyTorch LSTM conditional volatility forecasting.
- PyTorch VAE latent distribution sampling.
- Intel Core Ultra NPU integration attempts for compatible AI inference modules.
- Kupiec POF backtesting and a generated English Markdown report.

Run the full pipeline with:

```powershell
uv run market-risk-analysis --refresh-data
```

Generated artifacts are written under `outputs/`.

Compile the generated LaTeX report to PDF from Windows via WSL with:

```powershell
wsl bash -lc "cd /mnt/c/Users/mio.zhu/Documents/Market-Risk-Analysis/outputs; xelatex -interaction=nonstopmode -halt-on-error -jobname=report report-wrapper.tex; xelatex -interaction=nonstopmode -halt-on-error -jobname=report report-wrapper.tex"
```

This command reads `outputs/report.tex` through the wrapper file and writes `outputs/report.pdf`. Running XeLaTeX twice helps stabilize longtable widths and PDF outlines.
