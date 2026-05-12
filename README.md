# Hypothesis-HMM-with-supertrend
conducted hypothesis research with HMM model on supertrend indicator and xauusd.


What the Script Tests & Why
The core hypothesis is modeled exactly like your EA: every ST direction flip is an entry, the ST band is the SL, and the next opposing flip is the exit. A trade "rode the trend" if it exits in profit, "hit SL" if it exits at a loss. Since the ST band trails in your favor, this maps exactly to how your EA behaves.
Statistical validation built in:

Binomial test — is the win rate meaningfully above 50%? This tells you if you're seeing signal or luck.
Bootstrap CI — 10,000 resamples give you the honest range your win rate could be at.
One-sample t-test — is the mean R-multiple significantly different from zero? Win rate alone is misleading if losers are huge.
Bonferroni correction on alpha filters — prevents false positives when testing many combinations.
Chi-squared test on HMM — tells you statistically whether market regime actually predicts your trade outcome.

The seven breakdowns that reveal alpha:

Directional — does gold trend better long or short?
Hour of day — London open vs New York session vs Asian hours
ATR quartile — low vol environment vs high vol (your KER is a proxy for this)
Prior trend duration — does a long preceding trend predict a successful continuation?
Day of week — Monday reversals vs Thursday trends
HMM regime — BULL/BEAR/CHOP state vs trade outcome (the key KER upgrade)
Two-way grid search — combinations like direction=long & session=London with Bonferroni-corrected p-values
