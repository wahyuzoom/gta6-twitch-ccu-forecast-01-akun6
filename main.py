"""GTA VI launch-day Twitch CCU forecast, akun 6 (wahyuzoom / steady-gecko-37).

Competition: gta6-twitch-ccu-forecast-01. The task: for each of hours 2-24 of
GTA VI's release day (2026-11-19 UTC), forecast five quantiles of the Twitch
category's concurrent viewers, given hour 1's print and the case's comparables.

Model (per-hour ensemble of Theil-Sen and recency-weighted OLS, sharp
spread). This is a rebuild of the account's first entry, a blend of
comparable shapes that scored a practice loss of 0.196. The rebuild fits,
for each hour h, TWO lines across the rolling-midnight comparables:
log(viewers_h) = a_h + b_h * log(viewers_1), one by Theil-Sen (median of
pairwise slopes, robust to outlying launches) and one by recency-weighted
OLS (newer launches weighted heavier, 1.0 + 0.1*i in list order). The
forecast center is the average of the two fits. The spread is the
ensemble's residual standard deviation scaled by 0.75 before clamping to
[0.20, 1.00], so every hour's band is a quarter narrower in log terms
than a plain fit's.

Why this combination? In the leave-one-out backtest over the 12 rolling
comparables, Theil-Sen x0.75 scored 0.2413 and WOLS x0.75 scored 0.2434
— the two best single estimators — and their ensemble scored 0.2401,
better than either alone: averaging cancels each estimator's own error
while keeping both strengths (robustness from Theil-Sen, recency from
WOLS). The sharp spread was the winning move on the practice board,
where the sharp-spread OLS entry beat its plain sibling (0.157 against
0.162): the pinball loss rewards tight intervals when the center is
right. This entry stacks all three: the most robust center, recency
weighting, and the sharp spread.

Quantiles: q_tau(h) = exp(pred_h + z_tau * s_h), with s_h the scaled,
clamped ensemble residual spread. Deterministic, standard library only,
no network, no clock.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. It reads every <id>.in in the input directory and writes
<id>.out to the output directory: {"hours": [{"hour": h, "q": [q05, q25, q50,
q75, q95]}, ...]}, exactly 23 rows, hours 2-24 each once, each q five finite
numbers above 0 and at most 1e9, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The residual spread of an hour's fit, scaled by SPREAD_SCALE then clamped.
SPREAD_MIN = 0.20
SPREAD_MAX = 1.00
SPREAD_SCALE = 0.75
# A comparable needs at least this many release-day hours to count.
MIN_HOURS = 20
# The baseline's spread, used as a fallback when no comparable qualifies.
BASELINE_SIGMA = 0.15


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp ("...Z") as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def unlock_kind(comparable: dict) -> str:
    """How the launch unlocked: 'rolling', 'global', or 'other'.

    GTA VI unlocks at midnight local, rolling country by country, so only
    'rolling' comparables share its daily shape. The wording comes from the
    kit's own unlock field.
    """
    text = (comparable.get("unlock") or "").lower()
    if "midnight local" in text or "rolling midnight" in text:
        return "rolling"
    if "global" in text or "simultaneous" in text or "single global" in text:
        return "global"
    return "other"


def release_day_curve(comparable: dict) -> dict | None:
    """The comparable's release-day hourly viewers, {hour 1..24: viewers}.

    Returns None when hour 1 is missing or zero, or fewer than MIN_HOURS of the
    release day's hours are present (e.g. the GTA VI pre-release comparable,
    which has no release-day data yet).
    """
    day_start = parse_time(comparable["day_start"])
    by_hour: dict[int, float] = {}
    for point in comparable.get("hourly", []):
        at = parse_time(point["from"])
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = point["viewers"]
    if by_hour.get(1, 0) <= 0 or len(by_hour) < MIN_HOURS:
        return None
    return by_hour


def theil_sen_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """The Theil-Sen line y = a + b*x through the points (xs, ys).

    The slope is the median of the slopes between every pair of points with
    distinct x; the intercept is the median of y_i - b*x_i. Medians make the
    fit robust: a few outlying launches cannot move it. With no usable pair
    (every launch had the same hour-1 print), the slope is 0 and the intercept
    is the median of y.
    """
    slopes: list[float] = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if abs(xs[j] - xs[i]) > 1e-12:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    if not slopes:
        return statistics.median(ys), 0.0
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in zip(xs, ys)])
    return intercept, slope


def wols_fit(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """The weighted least-squares line y = a + b*x through (xs, ys).

    Weights come from the caller; here they rise with list order (1.0 + 0.1*i),
    so newer launches count more. With no spread in x, the slope is 0 and the
    intercept is the weighted mean of y.
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return my, 0.0
    slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    return my - slope * mx, slope


def fit_hourly(curves: list[dict]) -> dict[int, tuple[float, float, float]]:
    """For each hour 2-24, the ensemble fit (intercept, slope, spread).

    Each hour's center is the average of a Theil-Sen fit and a
    recency-weighted OLS fit on log(viewers_h) against log(viewers_1)
    across the given curves. The spread is the ensemble's residual
    standard deviation (residuals against the averaged prediction),
    scaled by SPREAD_SCALE, then clamped to [SPREAD_MIN, SPREAD_MAX].
    An hour fewer than 3 launches cover falls back to the previous hour's
    fit, or a flat line (intercept 0, slope 1) at hour 2.
    """
    log_p1 = [math.log(c[1]) for c in curves]
    fits: dict[int, tuple[float, float, float]] = {}
    prev = (0.0, 1.0, SPREAD_MIN)
    for h in range(2, 25):
        xs: list[float] = []
        ys: list[float] = []
        for curve, x in zip(curves, log_p1):
            if h in curve and curve[h] > 0:
                xs.append(x)
                ys.append(math.log(curve[h]))
        if len(xs) < 3:
            fits[h] = prev
            continue
        a_ts, b_ts = theil_sen_fit(xs, ys)
        # Recency weights are indexed by position in THIS hour's filtered
        # list (xs), not by position in the full curve list — exactly as
        # the leave-one-out backtest weights them. An hour some launches
        # do not cover still weights its newest covering launch heaviest.
        ws = [1.0 + 0.1 * i for i in range(len(xs))]
        a_w, b_w = wols_fit(xs, ys, ws)
        intercept = (a_ts + a_w) / 2
        slope = (b_ts + b_w) / 2
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        mean_resid = sum(residuals) / len(residuals)
        variance = sum((r - mean_resid) ** 2 for r in residuals) / len(residuals)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, math.sqrt(variance) * SPREAD_SCALE))
        fits[h] = (intercept, slope, spread)
        prev = fits[h]
    return fits


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_quantiles(p1: float, h: int, fits: dict[int, tuple[float, float, float]]) -> list[float | int]:
    """The five quantiles for hour h: the ensemble prediction plus a log-normal spread."""
    intercept, slope, spread = fits[h]
    center = math.exp(intercept + slope * math.log(p1))
    out = []
    for z in Z:
        q = center * math.exp(z * spread)
        # Every quantile must be a finite number above 0 and at most 1e9.
        q = max(0.01, min(1_000_000_000, q))
        out.append(round2(q))
    # Rounding can only move a quantile by a cent; keep them non-decreasing.
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # Only rolling-midnight comparables share GTA VI's daily shape; with
    # fewer than 5 of them, fall back to every usable comparable.
    rolling = []
    all_curves = []
    for comparable in case.get("comparables", []):
        curve = release_day_curve(comparable)
        if curve is None:
            continue
        all_curves.append(curve)
        if unlock_kind(comparable) == "rolling":
            rolling.append(curve)
    curves = rolling if len(rolling) >= 5 else all_curves
    if curves:
        fits = fit_hourly(curves)
    else:
        # No comparable at all: fall back to the house baseline's flat shape.
        fits = {h: (0.0, 1.0, BASELINE_SIGMA * math.sqrt(h - 1)) for h in range(2, 25)}

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], fits)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
