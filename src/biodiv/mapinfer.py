"""Multitemporal map inference for the unified-pool facet model.

Turns Landsat observations of a tile into one prediction per pixel and per census year
with the deployed model (`scripts/72_train_final_map_model.py --all-data`: 2D-CNN + MAE,
kNDVI raw series, centre pixel, `topo_ctr+area` context, five seeds). The design rule is
that every step here is the *same function* the training data went through, or a
vectorised re-implementation tested against it (`tests/test_mapinfer.py`):

- **Window.** Causal three-year window ``y-2 .. y`` per target year, as in
  `scripts/01_build_subset.py` / `scripts/35_extract_living_trees.py`.
- **Grid.** 100 regular steps between the first and the last acquisition date of the
  tile inside the window (the plot cubes span the cube's first..last date,
  `biodiv.curves.raw_series`), linear interpolation of the clear observations of each
  pixel, then the 5-step shrinking moving mean of `biodiv.curves.interp_grid`. Pixels
  with fewer than five clear observations get NaN, the same floor as training.
- **Image.** The same `serpentine` transform, applied as a fixed index permutation
  (`serpentine_perm`) so it is exactly the training transform and vectorises.
- **Context.** Eight centre-pixel topographic variables (`features.TOPO_VARS`, computed
  with `scripts/03_extract_topography.py:terrain`) plus the flat-terrain flag, and the
  two non-mappable covariates held constant: plot area (``area_m2``) and the
  abundance-recording protocol indicator (``stratum``). Their values are a *decision*,
  recorded in the output metadata, not a default hidden in code.
- **Model.** Each seed checkpoint carries its own `Preprocessor` and `PowerTransformer`;
  predictions are back-transformed per seed (with Duan smearing when residuals are
  supplied, plain inverse otherwise) and averaged, which is what `metrics.ensemble_oof`
  does across seeds in the cross-validation runs.

Nothing here touches the datacube except `load_kndvi` and `load_terrain`, which are thin
wrappers around `dc.load` kept separate (and importing `biodiv.cube` lazily) so the
numerical core is testable offline, without the `datacube` package.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from . import targets as tg
from .curves import ROLL, interp_grid
from .features import STRATA, TOPO_VARS, Preprocessor
from .models_conv import build_model
from .transforms1d import make_transform

NGS = 100
MIN_OBS = 5
KNDVI_BANDS = ["red", "nir", "qa_pixel"]
EPOCH = np.datetime64("1970-01-01", "D")


# --------------------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------------------

def days_since_epoch(times: np.ndarray) -> np.ndarray:
    """datetime64 -> float days, the unit `interp_grid` expects."""
    return (times.astype("datetime64[D]") - EPOCH).astype(float)


def window_mask(times: np.ndarray, year: int) -> np.ndarray:
    """Boolean mask of acquisition dates inside the causal window ``year-2 .. year``."""
    lo = np.datetime64(f"{year - 2}-01-01")
    hi = np.datetime64(f"{year}-12-31")
    t = times.astype("datetime64[D]")
    return (t >= lo) & (t <= hi)


def rolling_mean_shrinking(g: np.ndarray, roll: int) -> np.ndarray:
    """The `interp_grid` smoother, along the last axis, for a 2-D array."""
    if not roll or roll <= 1:
        return g
    n = g.shape[-1]
    c = np.cumsum(np.concatenate([np.zeros(g.shape[:-1] + (1,), g.dtype), g], axis=-1),
                  axis=-1)
    h = roll // 2
    a = np.maximum(np.arange(n) - h, 0)
    b = np.minimum(np.arange(n) + h + 1, n)
    return (c[..., b] - c[..., a]) / (b - a)


def interp_common_grid(t: np.ndarray, obs: np.ndarray, grid: np.ndarray,
                       roll: int = ROLL, min_obs: int = MIN_OBS
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `interp_grid` for many pixels sharing one grid.

    ``t`` (T,) float days, sorted; ``obs`` (T, N) with NaN where the observation is not
    usable; ``grid`` (G,) float days. Returns ``(curves (N, G) float32, n_obs (N,))``. A
    pixel with fewer than ``min_obs`` finite observations is all-NaN. Matches
    ``interp_grid(t[ok], v[ok], G, roll, t_min=grid[0], t_max=grid[-1])`` pixel by pixel
    (np.interp semantics: constant extrapolation beyond the first/last observation).
    """
    t = np.asarray(t, float)
    obs = np.asarray(obs, np.float32)
    if obs.ndim != 2:
        raise ValueError("obs must be (T, N)")
    T, N = obs.shape
    G = len(grid)
    order = np.argsort(t, kind="stable")
    t, obs = t[order], obs[order]
    finite = np.isfinite(obs)
    n_obs = finite.sum(axis=0)

    # forward fill: last valid value/time at or before each observation row
    idx = np.where(finite, np.arange(T)[:, None], -1)
    prev_idx = np.maximum.accumulate(idx, axis=0)                      # (T, N), -1 = none
    # backward fill: first valid at or after each row
    idx_b = np.where(finite, np.arange(T)[:, None], T)
    next_idx = np.minimum.accumulate(idx_b[::-1], axis=0)[::-1]        # (T, N), T = none

    # position of each grid point among the observation times
    pos = np.searchsorted(t, grid, side="right") - 1                   # last t <= grid
    pos_after = np.searchsorted(t, grid, side="left")                  # first t >= grid
    pos_c = np.clip(pos, 0, T - 1)
    pos_after_c = np.clip(pos_after, 0, T - 1)

    p_idx = np.where(pos[:, None] >= 0, prev_idx[pos_c], -1)           # (G, N)
    n_idx = np.where(pos_after[:, None] <= T - 1, next_idx[pos_after_c], T)

    have_p = p_idx >= 0
    have_n = n_idx < T
    p_safe = np.where(have_p, p_idx, 0)
    n_safe = np.where(have_n, n_idx, 0)
    cols = np.arange(N)[None, :]
    vp = obs[p_safe, cols].astype(np.float64)
    vn = obs[n_safe, cols].astype(np.float64)
    tp = t[p_safe]
    tn = t[n_safe]

    out = np.full((G, N), np.nan)
    both = have_p & have_n
    same = both & (tn <= tp)
    w = np.zeros_like(out)
    denom = np.where(both & ~same, tn - tp, 1.0)
    w = np.where(both & ~same, (grid[:, None] - tp) / denom, 0.0)
    out = np.where(both & ~same, vp + w * (vn - vp), out)
    out = np.where(same, vp, out)
    out = np.where(have_p & ~have_n, vp, out)                          # beyond last obs
    out = np.where(~have_p & have_n, vn, out)                          # before first obs

    curves = rolling_mean_shrinking(out.T, roll)                       # (N, G)
    curves = np.where((n_obs >= min_obs)[:, None], curves, np.nan)
    return curves.astype(np.float32), n_obs


def year_curves(times: np.ndarray, obs: np.ndarray, year: int, ngs: int = NGS,
                roll: int = ROLL) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Curves of every pixel for target ``year`` from the tile's observations.

    ``times`` (T,) datetime64 for all loaded acquisitions, ``obs`` (T, N) kNDVI with NaN
    where not clear. Returns ``(curves (N, ngs), n_obs (N,), t_first, t_last)`` where the
    grid spans the first and last acquisition inside the window that has at least one
    clear pixel in the tile (the plot-cube convention). Curves are all-NaN when fewer than
    two such dates exist.
    """
    m = window_mask(times, year)
    if m.sum() == 0:
        return (np.full((obs.shape[1], ngs), np.nan, np.float32),
                np.zeros(obs.shape[1], int), np.nan, np.nan)
    t = days_since_epoch(times[m])
    o = obs[m]
    any_clear = np.isfinite(o).any(axis=1)
    if any_clear.sum() < 2:
        return (np.full((obs.shape[1], ngs), np.nan, np.float32),
                np.isfinite(o).sum(axis=0), np.nan, np.nan)
    lo, hi = t[any_clear].min(), t[any_clear].max()
    grid = np.linspace(lo, hi, ngs)
    curves, n_obs = interp_common_grid(t, o, grid, roll=roll)
    return curves, n_obs, float(lo), float(hi)


def reference_curve(t_days: np.ndarray, v: np.ndarray, t_min: float, t_max: float,
                    ngs: int = NGS, roll: int = ROLL) -> np.ndarray:
    """The training-path curve for one pixel (`interp_grid`), for tests and audits."""
    return interp_grid(np.asarray(t_days, float), np.asarray(v, float), ngs, roll,
                       t_min=t_min, t_max=t_max)


# --------------------------------------------------------------------------------------
# image
# --------------------------------------------------------------------------------------

def serpentine_perm(ngs: int = NGS) -> np.ndarray:
    """(side, side) index map such that ``curve[perm]`` is the training serpentine image."""
    tf = make_transform("serpentine", normalize="none")
    perm = tf.transform(np.arange(ngs, dtype=float))
    perm = np.rint(perm).astype(int)
    if perm.min() < 0 or perm.max() >= ngs:
        raise RuntimeError("serpentine permutation out of range")
    return perm


def images_from_curves(curves: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """(N, ngs) -> (N, 1, side, side) float32."""
    return np.ascontiguousarray(curves[:, perm][:, None, :, :], dtype=np.float32)


# --------------------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------------------

def context_frame(columns: list[str], topo: dict[str, np.ndarray], area_m2: float,
                  stratum: str) -> pd.DataFrame:
    """Context block in the exact column order the checkpoint's Preprocessor expects.

    ``topo`` maps each of `TOPO_VARS` to a (N,) array at the pixel; NaN is allowed and is
    what the flat-terrain flag and the fold-median imputation were built for.
    """
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}, got {stratum!r}")
    n = len(next(iter(topo.values())))
    cols: dict[str, np.ndarray] = {}
    for v in TOPO_VARS:
        if v not in topo:
            raise KeyError(f"missing topographic variable {v!r}")
        cols[f"topo_{v}"] = np.asarray(topo[v], float)
    cols["topo_topo_flat"] = np.isnan(cols["topo_heat_load"]).astype(float)
    cols["area_log10"] = np.full(n, np.log10(float(area_m2)))
    for s in STRATA[1:]:
        cols[f"area_stratum_{s}"] = np.full(n, float(s == stratum))
    df = pd.DataFrame(cols)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"checkpoint expects context columns not built here: {missing}")
    return df[list(columns)]


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

@dataclass
class Member:
    path: Path
    model: torch.nn.Module
    pre: Preprocessor
    scaler: object
    targets: list[str]
    input_shape: tuple


class FacetEnsemble:
    """The five all-data seed checkpoints of the deployed 2D-CNN, as one predictor."""

    def __init__(self, ckpt_paths: list, device: str = "cpu", width: str = "B"):
        """``ckpt_paths`` are paths, or open binary files.

        The file-like form is what the gateway workers get: they cannot see the home
        directory, so the five checkpoints are broadcast to them as bytes rather than read
        from disk (`biodiv.maptask`). Same tensors either way -- ``torch.load`` does not care
        -- and the member keeps a synthetic name so error messages still say which seed.
        """
        self.device = torch.device(device)
        self.members: list[Member] = []
        for i, p in enumerate(ckpt_paths):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            pre: Preprocessor = ck["ctx_preprocessor"]
            targets = list(ck["targets"])
            n_ctx = len(pre.feature_names)
            model = build_model("C2D", c_in=1, n_out=len(targets), n_ctx=n_ctx, width=width,
                                pad_mode="zeros", fusion="late")
            model.load_state_dict(ck["state_dict"])
            model.to(self.device).eval()
            name = Path(p) if isinstance(p, (str, Path)) else Path(f"<seed{i}:in-memory>")
            self.members.append(Member(name, model, pre, ck["target_scaler"], targets,
                                       tuple(ck.get("input_shape", ()))))
        if not self.members:
            raise ValueError("no checkpoints")
        t0 = self.members[0].targets
        c0 = list(self.members[0].pre.columns_)
        for m in self.members[1:]:
            if m.targets != t0 or list(m.pre.columns_) != c0:
                raise ValueError(f"checkpoint {m.path} disagrees on targets/context columns")

    @property
    def targets(self) -> list[str]:
        return self.members[0].targets

    @property
    def context_columns(self) -> list[str]:
        return list(self.members[0].pre.columns_)

    @torch.no_grad()
    def predict_scaled(self, images: np.ndarray, ctx: pd.DataFrame,
                       batch: int = 8192) -> np.ndarray:
        """(n_members, N, n_targets) in the transformed target space."""
        out = np.full((len(self.members), images.shape[0], len(self.targets)), np.nan,
                      np.float32)
        ok = np.isfinite(images).all(axis=(1, 2, 3))
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return out
        x_all = torch.from_numpy(images[idx])
        for k, m in enumerate(self.members):
            c_all = torch.from_numpy(m.pre.transform(ctx.iloc[idx]))
            preds = []
            for s in range(0, len(idx), batch):
                xb = x_all[s:s + batch].to(self.device)
                cb = c_all[s:s + batch].to(self.device)
                preds.append(m.model(xb, cb).cpu().numpy())
            out[k, idx] = np.concatenate(preds, axis=0)
        return out

    def scaled_bounds(self, y_train: np.ndarray, member: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Training range of each target in the member's transformed space.

        Predictions are clipped to it before the inverse transform. This matters most for
        LCBD, whose Yeo-Johnson lambda is about -4200 (the facet spans 3.4e-4..4.4e-4 on
        2,499 plots): a transformed value a little outside the fitted range explodes on
        the way back, so the clip is what keeps a pixel finite.
        """
        sc = self.members[member].scaler
        y = np.asarray(y_train, float)
        filled = np.where(np.isfinite(y), y, np.nanmedian(y, axis=0))
        z = np.where(np.isfinite(y), sc.transform(filled), np.nan)
        return np.nanmin(z, axis=0), np.nanmax(z, axis=0)

    def predict(self, images: np.ndarray, ctx: pd.DataFrame,
                resid_scaled: np.ndarray | None = None,
                y_train: np.ndarray | None = None, batch: int = 8192) -> np.ndarray:
        """(N, n_targets) in original units: per-seed back-transform, then seed mean.

        With ``y_train`` the transformed predictions are clipped to the training range
        before inversion and the results to the observed range after it (the same guard
        `targets.inverse_with_smearing` applies); without it nothing is clipped.
        """
        scaled = self.predict_scaled(images, ctx, batch=batch)
        outs = []
        for k, m in enumerate(self.members):
            s = scaled[k].astype(np.float64)
            ok = np.isfinite(s).all(axis=1)
            o = np.full_like(s, np.nan, dtype=np.float64)
            if ok.any():
                if y_train is not None:
                    zlo, zhi = self.scaled_bounds(y_train, k)
                    s = np.clip(s, zlo, zhi)
                if resid_scaled is not None:
                    o[ok] = tg.inverse_with_smearing(s[ok], m.scaler, resid_scaled,
                                                     y_train=y_train, seed=k)
                else:
                    o[ok] = tg.inverse_target_scaler(s[ok], m.scaler)
                    if y_train is not None:
                        lo = np.nanmin(y_train, axis=0)
                        hi = np.nanmax(y_train, axis=0)
                        o[ok] = np.clip(o[ok], lo, hi)
            outs.append(o)
        stack = np.stack(outs, axis=0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(stack, axis=0).astype(np.float32)


def oof_residuals_scaled(oof_csv: Path | str, scaler, targets: list[str]) -> np.ndarray:
    """Residuals of a cross-validation run, moved to the deployed scaler's space.

    The all-data refit has no held-out residuals of its own; the out-of-fold predictions
    of the same configuration under `kfold5_block20_unified` are the closest honest
    substitute for Duan smearing (`targets.inverse_with_smearing` needs residuals the
    model did not fit directly). Both observed and predicted values are pushed through the
    deployed `PowerTransformer` so the residuals live where the smearing draws are added.
    """
    df = pd.read_csv(oof_csv)
    if "seed" in df.columns:
        df = df[df["seed"] == df["seed"].min()]
    obs = np.column_stack([df[f"{t}_obs"].to_numpy(float) for t in targets])
    pred = np.column_stack([df[f"{t}_pred"].to_numpy(float) for t in targets])
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs_s = np.full_like(obs, np.nan)
    pred_s = np.full_like(pred, np.nan)
    # PowerTransformer.transform is column-wise but refuses NaN: fill, transform, re-mask
    o_f = np.where(ok, obs, np.nanmedian(obs, axis=0))
    p_f = np.where(ok, pred, np.nanmedian(pred, axis=0))
    obs_s = np.where(ok, scaler.transform(o_f), np.nan)
    pred_s = np.where(ok, scaler.transform(p_f), np.nan)
    return obs_s - pred_s


def training_targets(derived: Path | str, targets: list[str]) -> np.ndarray:
    """(n_plots, n_targets) observed facet values on the unified pool, for range clipping.

    Read straight from the padded parquets `targets.TARGET_SOURCE` maps each facet to, so
    no curve table is needed (the plot order is irrelevant for a range).
    """
    derived = Path(derived)
    cols = []
    for t in targets:
        f = tg.TARGET_SOURCE.get(t)
        if f is None:
            raise KeyError(f"{t!r} is not a known target (targets.TARGET_SOURCE)")
        cols.append(pd.read_parquet(derived / f)[t].to_numpy(float))
    return np.column_stack(cols)


# --------------------------------------------------------------------------------------
# datacube wrappers
# --------------------------------------------------------------------------------------

def load_kndvi(dc, bbox_utm, year_start: int, year_end: int, resolution: int = 30,
               crs: str = "EPSG:32719", dask_chunks: dict | None = None
               ) -> xr.DataArray | None:
    """Clear-sky kNDVI ``(time, y, x)`` for a tile, three bands instead of seven.

    Same products, scaling, QA mask (per product, before concatenation), grouping and
    resampling as `cube.load_window` + `cube.to_indices`; only ``red``, ``nir`` and
    ``qa_pixel`` are read because kNDVI needs nothing else.
    """
    from . import cube as cubemod          # imports `datacube`; kept off the module path
    query = dict(
        x=(bbox_utm[0], bbox_utm[2]),
        y=(bbox_utm[1], bbox_utm[3]),
        crs=crs,
        time=(f"{year_start}-01-01", f"{year_end}-12-31"),
        output_crs=crs,
        resolution=(-resolution, resolution),
        group_by="solar_day",
        resampling="nearest",
    )
    if dask_chunks:
        query["dask_chunks"] = dask_chunks
    parts = []
    for prod in cubemod.products_for(year_start, year_end):
        ds = dc.load(product=prod, measurements=KNDVI_BANDS, **query)
        if ds.sizes.get("time", 0) == 0:
            continue
        clear = cubemod.clear_mask(ds["qa_pixel"])
        red = (ds["red"].where(ds["red"] > 0) * cubemod.SR_SCALE + cubemod.SR_OFFSET).where(clear)
        nir = (ds["nir"].where(ds["nir"] > 0) * cubemod.SR_SCALE + cubemod.SR_OFFSET).where(clear)
        ndvi = ((nir - red) / (nir + red)).clip(-1, 1)
        kndvi = np.tanh(ndvi ** 2).astype(np.float32)
        kndvi = kndvi.assign_coords(
            sensor=("time", np.array([prod.replace("_c2l2_sr", "")] * ds.sizes["time"])))
        parts.append(kndvi.rename("kndvi"))
    if not parts:
        return None
    da = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time")
    return da


def _terrain_fn():
    """`terrain()` from scripts/03, so the script stays the single source of the derivatives.

    Two ways in, and the order matters. On the pod the script is read from the repo by path,
    which is the original arrangement and keeps `scripts/03` authoritative. On a dask-gateway
    worker there is no repo -- `biodiv` arrives as a zip on sys.path -- and a path load cannot
    reach inside a zip anyway, so the caller (`scripts/73.package_biodiv`) copies that same
    file into the archive as `biodiv._terrain_src` and this falls through to importing it.
    The copy is rebuilt from the live script on every run, so the two cannot drift apart
    within a run; that, and not a second implementation, is what keeps the terrain identical
    on both sides.
    """
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "03_extract_topography.py"
    if script.exists():
        spec = importlib.util.spec_from_file_location("extract_topography", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.terrain
    from . import _terrain_src                     # shipped copy, worker side
    return _terrain_src.terrain


def load_terrain(dc, template: xr.DataArray, lat_deg: float, resolution: int = 30,
                 crs: str = "EPSG:32719", halo_px: int = 20,
                 product: str = "copernicus_dem_30") -> dict[str, np.ndarray]:
    """The eight `TOPO_VARS` at every pixel of ``template`` (a ``(y, x)`` grid).

    Loads the DEM with a halo, computes the derivatives on the wide window
    (`scripts/03_extract_topography.py:terrain`, unchanged) and crops to the template grid.
    """
    from scipy import ndimage
    x0, x1 = float(template.x.min()), float(template.x.max())
    y0, y1 = float(template.y.min()), float(template.y.max())
    buf = halo_px * resolution
    dem = dc.load(product=product, x=(x0 - buf, x1 + buf), y=(y0 - buf, y1 + buf),
                  crs=crs, output_crs=crs, resolution=(-resolution, resolution),
                  resampling="bilinear")
    if dem.sizes.get("time", 0):
        dem = dem.isel(time=0)
    elev = dem["elevation"].values.astype(float)
    elev = np.where(elev <= -1000, np.nan, elev)
    if np.isnan(elev).all():
        raise ValueError("empty DEM")
    if np.isnan(elev).any():
        elev = np.where(np.isnan(elev),
                        ndimage.generic_filter(np.nan_to_num(elev), np.nanmean, size=3,
                                               mode="nearest"),
                        elev)
    terr = _terrain_fn()(elev, resolution, lat_deg)
    out = {}
    for v in TOPO_VARS:
        da = xr.DataArray(terr[v], coords={"y": dem.y.values, "x": dem.x.values},
                          dims=("y", "x"))
        out[v] = da.sel(y=template.y, x=template.x, method="nearest").values.astype(np.float32)
    return out


def tile_grid(extent_utm: tuple[float, float, float, float], tile_m: float,
              resolution: int = 30) -> pd.DataFrame:
    """Tiles of ``tile_m`` covering ``extent_utm``, snapped to the tile grid.

    Columns: ``tile_id, xmin, ymin, xmax, ymax`` in the projected CRS. ``tile_m`` is
    snapped to a whole number of pixels (10 km at 30 m -> 9,990 m = 333 px), and tile
    origins to multiples of that size, so every tile sits on the same pixel lattice as
    the datacube grid (origin at the CRS origin) and mosaics need no resampling.
    """
    tile_m = int(round(tile_m / resolution)) * resolution     # snap: "10 km" -> 9,990 m
    if tile_m <= 0:
        raise ValueError("tile size must be at least one pixel")
    xmin, ymin, xmax, ymax = extent_utm
    ix0, ix1 = int(np.floor(xmin / tile_m)), int(np.floor(xmax / tile_m))
    iy0, iy1 = int(np.floor(ymin / tile_m)), int(np.floor(ymax / tile_m))
    rows = []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            rows.append(dict(tile_id=f"t{ix}_{iy}", xmin=ix * tile_m, ymin=iy * tile_m,
                             xmax=(ix + 1) * tile_m, ymax=(iy + 1) * tile_m))
    return pd.DataFrame(rows)
