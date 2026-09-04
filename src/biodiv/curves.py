"""Turning irregular satellite observations into a regular step grid.

Lives here rather than in a script because two callers need exactly the same grid, and a
second copy of it would be a silent divergence: `scripts/29_refit_curves_from_cubes.py` builds
the labelled plots' curves, and `scripts/31_pretrain_mae.py` builds the unlabelled pool's. If
the two ever interpolated differently, the transfer result would be confounded with a change
of substrate and nothing would raise.
"""

from __future__ import annotations

import numpy as np

#: Steps per cycle. 52 is weekly and was a choice, not a constraint.
NGS = 52
ROLL = 5


def interp_grid(t: np.ndarray, v: np.ndarray, ngs: int, roll: int = ROLL,
                t_min: float | None = None, t_max: float | None = None) -> np.ndarray:
    """One series onto ``ngs`` regular steps, with the shrinking-window smoother.

    ``t`` is in days, ``v`` the observations; non-finite values are dropped. Returns all-NaN
    when fewer than 5 observations survive, which is the same floor the plot pipeline uses.

    ``t_min``/``t_max`` pin the grid to a fixed span. Without them the grid stretches to each
    series' own first and last observation, so two samples from the same causal window would
    get step axes covering different amounts of real time -- the failure that
    `tests/test_step_cols.py` was written for, one level down.
    """
    ok = np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 5:
        return np.full(ngs, np.nan)
    t, v = t[ok], v[ok]
    lo = t.min() if t_min is None else t_min
    hi = t.max() if t_max is None else t_max
    if not (hi > lo):
        return np.full(ngs, np.nan)
    grid = np.linspace(lo, hi, ngs)
    o = np.argsort(t)
    g = np.interp(grid, t[o], v[o])
    if roll and roll > 1:
        c = np.cumsum(np.insert(g, 0, 0.0))
        h = roll // 2
        a = np.maximum(np.arange(ngs) - h, 0)
        b = np.minimum(np.arange(ngs) + h + 1, ngs)
        g = (c[b] - c[a]) / (b - a)
    return g


def raw_series(da, ngs: int, roll: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the raw observations onto a regular grid over the WHOLE window.

    The alternative to `PhenoShape`, and a different object. PhenoShape collapses three
    years onto one composite year, which averages the interannual variation away -- and that
    variation is real signal: the year-boundary step of the composite tracks the interannual
    trend with a regression slope of -0.945 against a predicted -1
    (`docs/13_phenology_year_boundary.md`). A series kept in calendar time never destroys it.

    It also gives a bigger image. With three years at weekly spacing the grid is 156 steps,
    which folds to 13x13 against the 8x8 of a 52-step composite -- and the paper this design
    comes from flags image size and fold topology as the axis nobody has studied.

    The cost is honest and worth stating: a weekly grid over 1,088 days is ~0.8 real
    observations per step, against ~2.4 for the composite, so more of the curve is
    interpolation. Whether the extra interannual signal beats the extra smoothing is exactly
    what the experiment measures.

    Returns ``(curves, doy)`` shaped ``(ngs, y, x)`` and ``(ngs,)``.
    """
    t = da["time"].values.astype("datetime64[D]").astype(float)
    grid = np.linspace(t.min(), t.max(), ngs)
    arr = np.asarray(da.values, dtype=float)               # (time, y, x)
    ny, nx = arr.shape[1], arr.shape[2]
    out = np.full((ngs, ny, nx), np.nan)
    for yy in range(ny):
        for xx in range(nx):
            out[:, yy, xx] = interp_grid(t, arr[:, yy, xx], ngs, roll,
                                         t_min=t.min(), t_max=t.max())
    # the grid coordinate is calendar day-of-year, so downstream labelling still reads as a
    # date; the year is not recoverable from it, which is fine because nothing downstream
    # uses it for anything but axis labels
    doy = ((grid - grid.min()) % 365.25) + 1
    return out, doy
