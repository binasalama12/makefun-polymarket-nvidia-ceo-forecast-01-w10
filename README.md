# polymarket-nvidia-ceo-forecast-01 entry — Quasar (w10)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

The calm hand of the family:

- **Shrunk centre:** 0.9 × last price + 0.1 × the 120-minute mean.
  A spike has to survive two hours of trading before this centre
  fully believes it.
- **Median-based sigma:** 1.4826 × the median absolute one-minute
  change over the last 1440 minutes (floor 0.0008, cap 0.003).
- **Spread = an even 50/50 blend** of the empirical quantiles of
  h-minute changes (anchors 1, 3, 5, 15, 30, 60, 120, 240,
  interpolated in sqrt(h)) and the Gaussian walk around the
  shrunk centre.
- **No regime rescaling**, deliberately: with the shrink in place,
  regime scaling would chase the same recent moves twice.
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.369 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
