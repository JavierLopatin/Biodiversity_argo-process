"""Annual composites and cross-index contrasts: the control that has no phenological shape.

The project's central claim is that the *shape* of the seasonal curve predicts community
composition. Section 4 of docs/09_predictors.md points out that the obvious control was
never run: an annual composite carries level and spread but no shape and no date, and the
annual mean of kNDVI alone already reaches Spearman -0.664 against ``pcoa1_pa``.

Two families live here, both computed from the stored 52-step curves so that they describe
exactly the same pixels, window and mask as the curve they are being compared against:

``composite``
    Twelve distribution statistics per index. Order is destroyed on purpose — permuting the
    52 steps leaves every one of these unchanged, which is precisely what makes the contrast
    against the curve interpretable.

``contrast``
    Differences and ratios between indices. NDVI and SAVI answer the same question with a
    different soil-background assumption, so their *difference* is a soil-exposure signal
    that neither carries alone; measured on this dataset it reaches Spearman -0.693 against
    ``pcoa1_pa``, stronger than any single feature in the current system.

A composite computed here still carries the smoother's fingerprint, because it is built on
the interpolated curve rather than on the observations. The observation-level version lives
in :mod:`biodiv.geomedian` and comes out of ``scripts/18_geomedian_from_cubes.py``; the two
are kept separate so the effect of smoothing is itself measurable.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

#: Statistics of the 52 curve values. Every one is invariant to permuting the time axis.
STAT_NAMES = ["mean", "sd", "p10", "p25", "p50", "p75", "p90", "min", "max",
              "range", "iqr", "cv"]


def curve_stats(V: np.ndarray) -> dict[str, np.ndarray]:
    """Column-wise statistics of a (n_plots, n_steps) curve matrix."""
    q10, q25, q50, q75, q90 = (np.nanpercentile(V, p, axis=1) for p in (10, 25, 50, 75, 90))
    mean = np.nanmean(V, axis=1)
    sd = np.nanstd(V, axis=1, ddof=1)
    vmin, vmax = np.nanmin(V, axis=1), np.nanmax(V, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(np.abs(mean) > 1e-9, sd / np.abs(mean), np.nan)
    return {"mean": mean, "sd": sd, "p10": q10, "p25": q25, "p50": q50, "p75": q75,
            "p90": q90, "min": vmin, "max": vmax, "range": vmax - vmin,
            "iqr": q75 - q25, "cv": cv}


def composite_block(curves: pd.DataFrame, ids: pd.Index, index: str, step_cols: list[str],
                    px: str = "mean5x5") -> pd.DataFrame:
    """Annual composite of one vegetation index."""
    V = curves.xs((index, px), level=("index", "px")).reindex(ids)[step_cols].to_numpy(float)
    stats = curve_stats(V)
    return pd.DataFrame({f"comp_{k}": v for k, v in stats.items()}, index=ids)


def contrast_block(curves: pd.DataFrame, ids: pd.Index, indices: list[str],
                   step_cols: list[str], px: str = "mean5x5") -> pd.DataFrame:
    """Pairwise contrasts between the annual statistics of different indices.

    Kept to the three that mean something physically: the difference in level (soil and
    canopy background differ between index definitions), the ratio of levels (the same
    contrast made scale-free), and the difference in temporal CV (which index varies more
    over the year is a different statement from which sits higher).
    """
    level, cv = {}, {}
    for ix in indices:
        V = curves.xs((ix, px), level=("index", "px")).reindex(ids)[step_cols].to_numpy(float)
        s = curve_stats(V)
        level[ix], cv[ix] = s["mean"], s["cv"]

    out: dict[str, np.ndarray] = {}
    for a, b in combinations(indices, 2):
        out[f"ctr_{a}_{b}_diff"] = level[a] - level[b]
        denom = np.where(np.abs(level[b]) > 1e-6, level[b], np.nan)
        out[f"ctr_{a}_{b}_ratio"] = level[a] / denom
        out[f"ctr_{a}_{b}_cvdiff"] = cv[a] - cv[b]
    return pd.DataFrame(out, index=ids)
