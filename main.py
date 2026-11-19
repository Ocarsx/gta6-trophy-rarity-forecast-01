#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (else /input) and writes one
``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Model, per trophy, in one paragraph: Exophase's page is a step function -- the
rates sit flat between Exophase refreshes and jump at each refresh, because a
refresh re-counts both the players tracked and the trophies they hold. So the
history is first reduced to its *refresh points* (reads where the rates or the
players-tracked count actually changed). On those points we fit, in logit
space, a line against x = ln(1 + hours since the release day began), anchored
at the last read so the level comes from the freshest data and only the slope
is estimated; the slope is shrunk toward the median slope of the trophy's
tier (story trophies climb fast, grind trophies crawl), which borrows strength
across trophies when a single trophy's history is thin. The median at the
target is that line read at the target's x: the forecast is the POST-refresh
level at the target, not the stale pre-refresh level the last read shows.
The spread is a logit-scale sigma combining a random-walk term
(0.4 * sqrt(horizon in days)), the slope's standard error times the
extrapolation distance, and the residual scatter of the fit, floored and
capped so a 90-minute horizon is not absurdly narrow and a 4.5-day horizon is
not absurdly wide.

One special case: a history that never changed at all (the practice cases:
every read identical, 638 players, flat rates). Then no slope is identifiable
from the data, and the only thing known is what the rules document about
Exophase's first refresh of this game: players tracked 638 -> 3,868 while
trophies earned went 943 -> 4,276, so the typical non-100% trophy's rate was
multiplied by (4276-3868)/(943-638) * (638/3868) = 0.2206. A flat history
with a short horizon is therefore forecast as last_rate * 0.2206 (the 100%
trophy stays at 100%), with a wide spread, because a refresh is due and the
answer is the refreshed page.

Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
from datetime import datetime, timezone

# Standard-normal quantiles for the five levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Rate multiplier applied when the whole history is one flat read (see the
# module docstring): the documented first Exophase refresh of this game.
FLAT_REFRESH_MULTIPLIER = 0.2206
# The flat-history branch is only trusted for short horizons; beyond this the
# trend model takes over even with no observed change.
FLAT_MAX_HORIZON_HOURS = 6.0
# Spread (logit sigma) used in the flat-history branch.
FLAT_SIGMA = 0.55

# Random-walk coefficient for the spread: sigma_rw = RW * sqrt(days).
# Tuned to 0.4 (from the house baseline's 0.5): on synthetic final-like
# histories the 0.8x spread scores consistently better than the 1.0x spread
# (loss 0.18-0.22 vs 0.23-0.27 across straight and saturating truths, 90%
# coverage still 1.00), while staying wide enough for real-world model
# error the simulations cannot see. The flat-history spread below was
# tuned separately on practice-noise simulations and is left at 0.55.
RW_COEF = 0.4
# Extra spread terms.
FIT_UNCERT_FLOOR = 0.24   # added in quadrature when a slope was not identified
STEP_SIGMA = 0.24         # residual refresh-step noise, in quadrature
SIGMA_FLOOR = 0.32
SIGMA_CAP = 1.20
# Slope shrinkage: b = (n_eff * b_i + PRIOR_WEIGHT * b_prior) / (n_eff + ...)
PRIOR_WEIGHT = 3.0
SLOPE_CAP = 2.0
# Recency half-life (hours) for the weights of the slope fit.
RECENCY_HALF_LIFE_H = 24.0

# Fallback medians (percent) by tier when a case carries no history at all.
TIER_FALLBACK = {"platinum": 0.2, "gold": 1.0, "silver": 2.0, "bronze": 1.0}


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp (with 'Z' or an offset)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def logit_pct(rate_pct):
    """logit of a percent rate, clamped to [0.01, 99.99] first."""
    p = min(max(rate_pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(eta):
    """Inverse logit, returned as a percent in [0.01, 99.99]."""
    if eta >= 0:
        p = 1.0 / (1.0 + math.exp(-eta))
    else:
        e = math.exp(eta)
        p = e / (1.0 + e)
    return min(max(p * 100.0, 0.01), 99.99)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def refresh_points(history):
    """Collapse the history to reads where anything changed.

    Returns a list of (hours_since_release, players_tracked, rates_tuple),
    oldest first, keeping the first read and every later read whose rates or
    players-tracked differ from the previous read.
    """
    points = []
    prev = None
    for read in history:
        rates = tuple(read.get("rates") or [])
        players = read.get("players_tracked")
        key = (players, rates)
        if prev is None or key != prev:
            try:
                t = hours_since_release(parse_ts(read["at"]))
            except (KeyError, ValueError, TypeError):
                continue
            points.append((t, players, rates))
        prev = key
    return points


def last_known_rate(points, idx):
    """Most recent non-null rate for trophy idx, else None."""
    for _t, _n, rates in reversed(points):
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def fit_slope(points, idx, x_last, logit_last, t_last):
    """Anchored weighted least-squares slope of logit(rate) vs ln(1+h).

    Fits y = b * u through the origin, where u = x - x_last and
    y = logit(rate) - logit(last rate), over the refresh points strictly
    before the last read, with recency weights. Returns
    (slope, slope_se, resid_sd, n_points) or None when unidentifiable.
    """
    num = 0.0
    den = 0.0
    rows = []
    for t, _n, rates in points:
        if idx >= len(rates) or rates[idx] is None:
            continue
        x = math.log1p(max(t, -0.99))
        u = x - x_last
        if u >= -1e-9:
            continue  # the anchor point itself (or a tie) carries no slope info
        y = logit_pct(rates[idx]) - logit_last
        w = 0.5 ** ((t_last - t) / RECENCY_HALF_LIFE_H)
        rows.append((u, y, w))
        num += w * u * y
        den += w * u * u
    if len(rows) < 2 or den <= 1e-9:
        return None
    slope = num / den
    # Weighted residual scatter around the anchored line.
    wsum = sum(w for _u, _y, w in rows)
    if len(rows) >= 3 and wsum > 0:
        resid_var = sum(w * (y - slope * u) ** 2 for u, y, w in rows) / wsum
        resid_sd = math.sqrt(max(resid_var, 0.0))
    else:
        resid_sd = 0.0
    slope_se = resid_sd / math.sqrt(den) if resid_sd > 0 else 0.0
    return slope, slope_se, resid_sd, len(rows)


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = horizon_min / 1440.0
    horizon_hours = horizon_min / 60.0

    target_text = case.get("target")
    if target_text:
        t_target = hours_since_release(parse_ts(target_text))
    else:
        at = parse_ts(case["at"])
        t_target = hours_since_release(at) + horizon_hours
    x_target = math.log1p(max(t_target, -0.99))

    points = refresh_points(history)
    flat_history = len(points) <= 1

    # --- per-trophy slope fits -------------------------------------------
    slopes = {}       # trophy index -> (slope, slope_se, resid_sd, n)
    tier_slopes = {}  # tier -> [slopes]
    all_slopes = []
    if points:
        t_last = points[-1][0]
        x_last = math.log1p(max(t_last, -0.99))
        for i, tr in enumerate(trophies):
            r_last = last_known_rate(points, i)
            if r_last is None:
                continue
            fit = fit_slope(points, i, x_last, logit_pct(r_last), t_last)
            if fit is not None:
                slopes[i] = fit
                all_slopes.append(fit[0])
                tier_slopes.setdefault(tr.get("tier") or "", []).append(fit[0])
    tier_prior = {}
    for tier, vals in tier_slopes.items():
        if len(vals) >= 2:
            tier_prior[tier] = median(vals)
    global_prior = median(all_slopes) if len(all_slopes) >= 3 else None

    rows = []
    for i, tr in enumerate(trophies):
        tid = tr.get("id")
        tier = tr.get("tier") or ""
        r_last = last_known_rate(points, i) if points else None

        if r_last is None:
            # No usable history for this trophy: tier fallback, huge spread.
            med = TIER_FALLBACK.get(tier, 1.0)
            sigma = SIGMA_CAP
        elif r_last >= 99.99:
            # Everybody holds it and it survived the last refresh at 100%:
            # keep it at the top with a narrow band.
            rows.append({"id": tid, "q": [98.0, 99.5, 99.9, 99.99, 100.0]})
            continue
        elif flat_history and horizon_hours <= FLAT_MAX_HORIZON_HOURS:
            # No refresh ever observed: expect the documented first-refresh
            # dilution (see module docstring).
            med = min(max(r_last * FLAT_REFRESH_MULTIPLIER, 0.01), 99.99)
            sigma = FLAT_SIGMA
            rows.append({"id": tid, "q": quantiles(med, sigma)})
            continue
        else:
            # Trend model: anchored slope, shrunk toward the tier prior.
            eta_last = logit_pct(r_last)
            fit = slopes.get(i)
            if fit is not None:
                b_raw, b_se, resid_sd, n_pts = fit
                prior = tier_prior.get(tier, global_prior)
                if prior is None:
                    prior = 0.0
                b = (n_pts * b_raw + PRIOR_WEIGHT * prior) / (n_pts + PRIOR_WEIGHT)
                b = min(max(b, -SLOPE_CAP), SLOPE_CAP)
                sigma_fit = min(1.0, b_se * max(x_target - x_last, 0.0)) + min(resid_sd, 0.8)
            else:
                prior = tier_prior.get(tier, global_prior)
                b = prior if prior is not None else 0.0
                b = min(max(b, -SLOPE_CAP), SLOPE_CAP)
                sigma_fit = FIT_UNCERT_FLOOR
            eta_med = eta_last + b * (x_target - x_last)
            med = inv_logit_pct(eta_med)
            sigma_rw = RW_COEF * math.sqrt(max(horizon_days, 0.0))
            sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + STEP_SIGMA ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
            rows.append({"id": tid, "q": quantiles(med, sigma)})
            continue

        rows.append({"id": tid, "q": quantiles(med, sigma)})

    return {"trophies": rows}


def quantiles(med_pct, sigma):
    """Five quantiles around a median percent, on the logit scale."""
    eta = logit_pct(med_pct)
    out = []
    prev = None
    for z in Z:
        q = inv_logit_pct(eta + z * sigma)
        q = round(q, 4)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
