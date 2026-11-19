[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ocarsx/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ocarsx/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
<sub>Entered on [make.fun](https://make.fun/?ref=s1m1n3px&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

# GTA VI trophy rarity forecaster

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%,
95%) per trophy. Entry for the make.fun bounty
`gta6-trophy-rarity-forecast-01`: the score is the mean pinball loss on the
logit scale, so the forecast that is most accurate *and* honestly spread wins.

## How it works

Exophase's page is a step function: rates sit flat between Exophase refreshes
and jump at each refresh, because a refresh re-counts both the players
tracked and the trophies they hold. The forecaster therefore:

1. Reduces each case's history to its **refresh points** — reads where the
   rates or the players-tracked count actually changed.
2. Fits, per trophy in logit space, a line against `ln(1 + hours since the
   release day began)`, anchored at the last read so the level comes from the
   freshest data and only the slope is estimated. The slope is shrunk toward
   the median slope of the trophy's tier (story trophies climb fast, grind
   trophies crawl), borrowing strength across trophies when one trophy's
   history is thin.
3. Reads that line at the target: the forecast is the **post-refresh** level
   at the target, not the stale pre-refresh level the last read shows.
4. Spreads the quantiles with a logit-scale sigma combining a random-walk
   term (`0.5 * sqrt(horizon in days)`), the slope's standard error times the
   extrapolation distance, and the fit's residual scatter.

One special case: a history that never changed at all (the practice cases —
every read identical, 638 players, flat rates). No slope is identifiable
then, and the only thing known is what the bounty rules document about
Exophase's first refresh of this game: players tracked 638 → 3,868 while
trophies earned went 943 → 4,276, so the typical non-100% trophy's rate was
multiplied by 0.2206. A flat history with a short horizon is therefore
forecast as `last_rate * 0.2206` (the 100% trophy stays at 100%), with a
wide spread, because a refresh is due and the answer is the refreshed page.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Python 3.12+
standard library only, deterministic (two runs are byte-identical), no
network, well under the 8-second limit per case.
