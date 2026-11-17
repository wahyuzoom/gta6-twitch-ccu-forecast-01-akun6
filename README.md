# GTA VI launch-day Twitch CCU forecast — akun 6 rebuild

Competition: `gta6-twitch-ccu-forecast-01` (make.fun). Forecast five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of Twitch concurrent viewers
for the "Grand Theft Auto VI" category, hours 2–24 of release day
(2026-11-19 UTC), given the hour-1 print.

## Model

Per-hour **ensemble of Theil-Sen and recency-weighted OLS** on
`log(viewers_h)` vs `log(viewers_1)` across the rolling-midnight
comparables, with a **sharp spread (×0.75)**:

- Theil-Sen: median of pairwise slopes — robust to outlying launches.
- WOLS: weights `1.0 + 0.1*i` in list order — newer launches count more.
- Center: average of the two fits. Spread: ensemble residual sd × 0.75,
  clamped to [0.20, 1.00].

Leave-one-out backtest over the 12 rolling comparables: **mean pinball
loss 0.2401** — better than Theil-Sen alone (0.2413) and WOLS alone
(0.2434), the two best single estimators.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Stdlib only, deterministic, no network. Reads every `<id>.in`, writes
`<id>.out` with 23 rows (hours 2–24), five quantiles each.

## Rebuild note

This replaces the account's first entry (a blend of comparable shapes,
practice loss 0.196, rank 1.404). Same repo, new history: `makefun
init` root (freshness anchor) + LICENSE + this work.
