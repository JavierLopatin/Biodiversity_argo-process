"""Design matrices: the feature blocks and how they are assembled, imputed and scaled.

One entry point, :func:`build_design`, composes named blocks with ``+``::

    X, cols, ids = build_design("lsp+topo+area", index="ndvi")

Everything is reindexed onto the canonical plot order of ``plots_subset.parquet``. This is
deliberate and load-bearing: the six source parquet files have different natural row orders,
and misaligning X against Y is the failure mode that produces a plausible R-squared from
nothing.

Block choices that are decisions, not defaults:

- **LSP uses ``*_mean5x5``, not the centre pixel.** Measured: centre ``los/ios/sw/mos`` have
  113 NaN each and ``rog``/``ros`` 64/77; the 5x5 means have 2 NaN in total. The centre-pixel
  variant is available as ``lsp_ctr`` for an ablation, not as the default.
- **``aspect`` is dropped.** It is a circular variable encoded as degrees — 359 and 1 are
  adjacent in the world and maximally distant to a tree split — and it is already represented
  losslessly by ``northness``/``eastness``, which are its sine and cosine.
- **Topography uses the patch mean and sd, never the centre pixel.** Same reasoning as LSP:
  centre and 5x5 mean correlate at r = 0.73-1.00. Every predictor block therefore describes
  the same 5x5 Landsat window, and no variable is represented twice.
- **``log10_area`` and ``stratum`` are in every design.** Risk R3 is real here: Spearman
  between plot area and ``hill_q0`` is +0.33. Leaving area out does not remove the confound,
  it only hides it. The `area` block is also fitted alone (baseline B02) so the share of
  richness attributable to sampling effort is quoted next to every richness result.
- **NaN is imputed with the training-fold median plus a ``_isna`` indicator.** Never dropped:
  a missing LSP metric means a degenerate season, which correlates with aridity and is
  therefore informative rather than absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import composites

ID_COL = "PlotObservationID"

#: Default step count -- what `scripts/05_flatten_phenoshape.py` writes for the composite
#: year. It is a **default, not a fact about the data**: `BIODIV_CURVES` can point the
#: pipeline at a table with any number of steps (the raw 3-year series uses 24 to 196).
NGS = 52
STEP_COLS = [f"s{i:02d}" for i in range(NGS)]


def step_cols(df: pd.DataFrame) -> list[str]:
    """The step columns actually present in ``df``, in order.

    Always prefer this over the `STEP_COLS` constant when reading a curve table. Indexing a
    DataFrame with a hardcoded ``s00..s51`` does one of two things and neither is acceptable:
    it raises for a shorter table, or -- far worse -- it **silently keeps the first 52
    columns** of a longer one. The second is what happened to the first raw-series sweep: the
    68/100/156/196-step curves were quietly cropped to their first 52 steps, so the runs
    compared how much of the 3-year window fits in 52 columns rather than the resolution they
    were labelled with. They scored, they ranked, and they meant something else.
    """
    cols = [c for c in df.columns if len(c) > 1 and c[0] == "s" and c[1:].isdigit()]
    if not cols:
        raise ValueError(f"no step columns (s00, s01, ...) in {list(df.columns)[:8]}")
    return sorted(cols, key=lambda c: int(c[1:]))

LSP_METRICS = ["sos", "pos", "eos", "vsos", "vpos", "veos", "los", "msp", "mau",
               "vmsp", "vmau", "ampl", "ios", "rog", "ros", "sw", "trough", "mos"]

#: LSP metrics whose value is a day-of-year, i.e. circular. Optionally re-encoded as
#: (sin, cos) with ``--circular-doy``; left raw by default because trees handle a monotone
#: DOY axis fine and the re-encoding only matters for the MLP.
DOY_METRICS = ["sos", "pos", "eos", "msp", "mau", "trough"]

TOPO_VARS = ["elevation", "slope", "northness", "eastness", "heat_load",
             "tpi", "tri", "curvature"]          # 'aspect' deliberately excluded
#: `northness`, `eastness` and `heat_load` carry a sign whose meaning depends on the aspect
#: convention of the table being read, and the two tables in the repo do NOT agree: see the
#: warning in `docs/07_run_record.md` section 3 before interpreting their direction.

#: Patch mean and patch sd, never the centre pixel. Measured on this dataset the centre value
#: and the 5x5 mean correlate at r = 0.73-1.00 (exactly 1.000 for elevation), so carrying both
#: adds no information and splits the permutation importance of one variable across two
#: columns. The sd is kept because it is genuinely different (r = -0.37 to 0.41 against the
#: centre) and measures within-plot terrain heterogeneity.
#:
#: This also makes the design matrix uniform: LSP uses `*_mean5x5`, the curve uses the
#: `mean5x5` pixel level, and topography now uses the patch mean. Every predictor describes
#: the same 5x5 Landsat window.
TOPO_SUFFIX = ["_mean", "_std"]

QC_RAW = ["degenerate_px_frac", "aseasonal_frac", "strength_median", "anchor_circ_sd"]

STRATA = ["cover", "counts", "presence", "basal"]

INDICES = ["ndvi", "evi", "kndvi", "nbr", "savi"]


# --------------------------------------------------------------------------------------
# source tables
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tables:
    plots: pd.DataFrame       # canonical order, indexed by plot id
    lsp: pd.DataFrame         # MultiIndex (plot_id, index)
    curves: pd.DataFrame      # MultiIndex (plot_id, index, px)
    pixels: Path              # loaded lazily; 28 MB
    topo: pd.DataFrame        # indexed by plot id
    doy_grid: pd.DataFrame    # indexed by plot id
    derived: Path
    cube: pd.DataFrame | None  # geomedian / observation composites, indexed by plot id


#: Suffix appended to the three phenoshape parquets, read from the environment.
#:
#: `scripts/29_refit_curves_from_cubes.py` writes `phenoshape_*_v2.parquet` rather than
#: overwriting, so the corrected curves can be compared against the originals with one
#: factor changing. Setting BIODIV_CURVES=_v2 points the whole pipeline at them without
#: editing anything -- and, crucially, without a half-corrected state where some blocks read
#: the new curves and others the old. `runlog.make_run_id` appends it to every run id so the
#: two versions cannot land in the same results directory.
CURVE_SUFFIX_ENV = "BIODIV_CURVES"


def curve_suffix() -> str:
    return os.environ.get(CURVE_SUFFIX_ENV, "")


#: Set to switch every table `load_tables` reads from the Parcelas-CL-only 1,082-plot
#: subset to the unified 3,102-plot pool (Parcelas-CL + Living Trees, `scripts/50`/`51`).
#: Same rationale as `BIODIV_CURVES` (`curve_suffix`, above): without an explicit switch, a
#: process that flips between the two would keep serving whichever it loaded first, and a
#: half-unified state (unified plots, Parcelas-CL-only curves) would silently misalign X.
#:
#: `topo` now has a real Living Trees extraction (`scripts/70`, from
#: `data/derived/living_trees/topography.parquet`). `lsp`/`doy_grid` still don't -- see
#: `_load_tables`'s relaxed check below. Specs built from `curve`/`curve_all`/`area`/
#: `coords`/`year`/`topo` are safe on the unified pool; `lsp`/`clim`/`gm`/`obscomp`/`seas`/
#: `svh`/`contrast`/`composite` will still NaN-impute entirely for every Living Trees row
#: until that extraction exists.
UNIFIED_ENV = "BIODIV_UNIFIED"


def unified_flag() -> bool:
    return os.environ.get(UNIFIED_ENV, "") not in ("", "0")


def load_tables(derived: str = "data/derived") -> Tables:
    # the suffix (and the unified flag) are part of the cache key: without them, a process
    # that switches curve version or pool would keep serving the tables it loaded first
    return _load_tables(derived, curve_suffix(), unified_flag())


@lru_cache(maxsize=4)
def _load_tables(derived: str, sfx: str, unified: bool = False) -> Tables:
    d = Path(derived)
    plots_name = "plots_unified.parquet" if unified else "plots_subset.parquet"
    plots = pd.read_parquet(d / plots_name)
    plots = plots.sort_values(ID_COL, kind="stable").reset_index(drop=True)
    plots["log10_area"] = np.log10(plots["PlotSize_m2"].astype(float))

    curves_name = f"phenoshape_by_index{sfx}_unified.parquet" if unified \
        else f"phenoshape_by_index{sfx}.parquet"
    curves = pd.read_parquet(d / curves_name).set_index(["plot_id", "index", "px"])

    ids = pd.Index(plots[ID_COL])
    missing_curves = ids.difference(curves.index.get_level_values(0).unique())
    if len(missing_curves):
        raise SystemExit(f"curves: {len(missing_curves)} plots absent, "
                         f"e.g. {list(missing_curves[:5])}")

    # lsp/doy_grid are still Parcelas-CL-only -- unified reindexes onto the wider id set and
    # warns (not aborts) when the gap is exactly the Living Trees subset (expected, per the
    # deferral above); a gap on the Parcelas-CL side would still mean something is wrong.
    # topo now has a real Living Trees extraction (`scripts/70`, DEM-derived) -- use the
    # unified file when it exists so the topo context block stops NaN-imputing for 2,020
    # rows; fall back to the Parcelas-CL-only table (and the warning below) if it doesn't.
    lsp = pd.read_parquet(d / "lsp_all_auto.parquet").set_index(["plot_id", "index"])
    topo_unified_path = d / "topography_unified.parquet"
    topo_has_lt = unified and topo_unified_path.exists()
    topo_path = topo_unified_path if topo_has_lt else d / "topography" / "topography.parquet"
    topo = pd.read_parquet(topo_path).set_index("plot_id")
    doy = pd.read_parquet(d / f"phenoshape_doy_grid{sfx}.parquet").set_index("plot_id")

    if unified:
        # These still use Parcelas-CL's raw numeric id ("32477"), never migrated to the
        # "PCL_"-prefixed scheme `scripts/51` gives plots_unified.parquet -- without this,
        # every `.reindex(ids)` below would silently return all-NaN for Parcelas-CL too,
        # not just the expected Living Trees gap. topo skips this when it already came from
        # `topography_unified.parquet`, which is PCL_/LT_-prefixed from the start.
        lsp.index = lsp.index.set_levels("PCL_" + lsp.index.levels[0].astype(str), level=0)
        if not topo_has_lt:
            topo.index = "PCL_" + topo.index.astype(str)
        doy.index = "PCL_" + doy.index.astype(str)

    pcl_ids = ids if not unified else pd.Index(plots.loc[plots["source"] == "parcelas_cl", ID_COL])
    for name, tbl in (("lsp", lsp),):
        missing = pcl_ids.difference(tbl.index.get_level_values(0).unique())
        if len(missing):
            raise SystemExit(f"{name}: {len(missing)} Parcelas-CL plots absent, "
                             f"e.g. {list(missing[:5])}")
    for name, tbl in (("topography", topo), ("doy_grid", doy)):
        missing = pcl_ids.difference(tbl.index)
        if len(missing):
            raise SystemExit(f"{name}: {len(missing)} Parcelas-CL plots absent, "
                             f"e.g. {list(missing[:5])}")
    if unified:
        n_no_topo = ids.difference(topo.index).size
        if n_no_topo:
            print(f"[features] {n_no_topo} unified plots (Living Trees) have no topography "
                 "yet -- topo/clim/gm/obscomp/seas/svh/contrast/composite blocks will "
                 "NaN-impute for them until that extraction exists.")

    # Optional: only present once scripts/18 and 18b have run. Every block that needs it
    # raises a pointed error rather than a KeyError, because "the cube predictors were never
    # extracted" and "this column name is wrong" are different problems.
    cube_path = d / "cube_predictors.parquet"
    cube = None
    if cube_path.exists():
        cube = pd.read_parquet(cube_path)
        cube = cube.set_index(ID_COL) if ID_COL in cube.columns else cube.set_index("plot_id")

    return Tables(plots=plots, lsp=lsp, curves=curves,
                  pixels=d / f"phenoshape_pixels{sfx}.parquet",
                  topo=topo, doy_grid=doy, derived=d, cube=cube)


def plot_ids(derived: str = "data/derived") -> pd.Index:
    """The canonical plot order every matrix in this project is built against."""
    return pd.Index(load_tables(derived).plots[ID_COL])


# --------------------------------------------------------------------------------------
# individual blocks
# --------------------------------------------------------------------------------------

def _block_lsp(t: Tables, ids: pd.Index, index: str, centre: bool = False,
               circular_doy: bool = False) -> pd.DataFrame:
    cols = LSP_METRICS if centre else [f"{m}_mean5x5" for m in LSP_METRICS]
    sub = t.lsp.xs(index, level="index").reindex(ids)[cols]
    sub.columns = [f"lsp_{c}" for c in cols]
    if circular_doy:
        suffix = "" if centre else "_mean5x5"
        extra = {}
        for m in DOY_METRICS:
            d = sub[f"lsp_{m}{suffix}"].to_numpy(float)
            extra[f"lsp_{m}_sin"] = np.sin(2 * np.pi * d / 365.0)
            extra[f"lsp_{m}_cos"] = np.cos(2 * np.pi * d / 365.0)
        sub = pd.concat([sub, pd.DataFrame(extra, index=sub.index)], axis=1)
    return sub


def _block_curve(t: Tables, ids: pd.Index, index: str, px: str = "mean5x5") -> pd.DataFrame:
    cols = step_cols(t.curves)
    sub = t.curves.xs((index, px), level=("index", "px")).reindex(ids)[cols]
    sub.columns = [f"curve_{c}" for c in cols]
    return sub


def _block_qc(t: Tables, ids: pd.Index, index: str) -> pd.DataFrame:
    sub = t.lsp.xs(index, level="index").reindex(ids)
    out = sub[QC_RAW].copy()
    out["order_coherent"] = sub["order_coherent"].astype(float)
    out["is_winter_peaking"] = (sub["phase_group"].astype(str) == "winter_peaking").astype(float)
    out.columns = [f"qc_{c}" for c in out.columns]
    return out


def _block_topo(t: Tables, ids: pd.Index, centre: bool = False) -> pd.DataFrame:
    """Patch mean + sd by default; ``centre`` gives the single centre-pixel value instead.

    The centre variant exists for the pixel-level ablation (``--px center``), which asks
    whether describing the plot by its centre Landsat pixel beats describing it by the 5x5
    window. There is no sd for a single pixel, so the centre design is 8 columns against 16 —
    part of what the ablation measures is exactly that loss of within-plot information.
    """
    if centre:
        cols = list(TOPO_VARS)
        na_key = "heat_load"
    else:
        cols = [f"{v}{s}" for v in TOPO_VARS for s in TOPO_SUFFIX]
        na_key = "heat_load_mean"
    sub = t.topo.reindex(ids)[cols].copy()
    # A few plots sit on flat terrain where heat_load is undefined. Flag them rather than
    # letting the median imputation pretend they are ordinary.
    sub["topo_flat"] = sub[na_key].isna().astype(float)
    sub.columns = [f"topo_{c}" for c in sub.columns]
    return sub


def _block_area(t: Tables, ids: pd.Index) -> pd.DataFrame:
    p = t.plots.set_index(ID_COL).reindex(ids)
    out = pd.DataFrame({"area_log10": p["log10_area"].to_numpy(float)}, index=ids)
    for s in STRATA[1:]:            # drop the first level, cover, as the reference
        out[f"area_stratum_{s}"] = (p["stratum"].astype(str) == s).astype(float).to_numpy()
    return out


def _block_coords(t: Tables, ids: pd.Index) -> pd.DataFrame:
    p = t.plots.set_index(ID_COL).reindex(ids)
    return pd.DataFrame({
        "geo_lon": p["lon"].to_numpy(float),
        "geo_lat": p["lat"].to_numpy(float),
        "geo_elevation": t.topo.reindex(ids)["elevation_mean"].to_numpy(float),
    }, index=ids)


def _block_year(t: Tables, ids: pd.Index) -> pd.DataFrame:
    p = t.plots.set_index(ID_COL).reindex(ids)
    return pd.DataFrame({"year": p["Year"].to_numpy(float)}, index=ids)


# --------------------------------------------------------------------------------------
# blocks without phenological shape — the controls of docs/09_predictors.md
# --------------------------------------------------------------------------------------
#
# Everything below describes the same 5x5 window, the same three-year causal window and the
# same cloud mask as the curve. They differ from it in one respect only: none of them can
# tell you *when* anything happened. That is what makes "does shape help?" answerable.

#: How the 25 pixels are collapsed into one plot value. Screened as a factor (row X10 of
#: docs/09_predictors.md) rather than assumed: `median` is robust to the single failing
#: pixel that `np.nanmean` silently absorbs.
DEFAULT_AGG = "median"


def _cube(t: Tables) -> pd.DataFrame:
    if t.cube is None:
        raise SystemExit(
            "data/derived/cube_predictors.parquet not found — run "
            "`python scripts/18_geomedian_from_cubes.py` then "
            "`python scripts/18b_aggregate_cube_predictors.py` first."
        )
    return t.cube


def _cube_cols(t: Tables, ids: pd.Index, prefixes: tuple[str, ...], agg: str,
               name: str) -> pd.DataFrame:
    """Pull one aggregation level of a family of cube columns, in canonical plot order."""
    cube = _cube(t)
    suffix = f"_{agg}"
    cols = [c for c in cube.columns
            if c.endswith(suffix) and c[: -len(suffix)].startswith(prefixes)]
    if not cols:
        raise ValueError(f"no {name} columns for agg={agg!r}")
    sub = cube.reindex(ids)[cols].copy()
    sub.columns = [c[: -len(suffix)] for c in cols]
    return sub


def _block_composite(t: Tables, ids: pd.Index, index: str, px: str = "mean5x5"
                     ) -> pd.DataFrame:
    """Annual statistics of the 52-step curve: level and spread, no shape and no date."""
    return composites.composite_block(t.curves, ids, index, step_cols(t.curves), px=px)


def _block_contrast(t: Tables, ids: pd.Index, px: str = "mean5x5") -> pd.DataFrame:
    """Differences and ratios between the annual levels of the five indices."""
    return composites.contrast_block(t.curves, ids, INDICES, step_cols(t.curves), px=px)


def _block_gm(t: Tables, ids: pd.Index, agg: str = DEFAULT_AGG) -> pd.DataFrame:
    """Geomedian spectrum, its five derived indices, and the three MADs.

    The MADs are the load-bearing part: within-window variability with the time axis
    discarded. If the fitted curve cannot beat a geomedian plus its MADs, the gain the
    project attributes to phenology was never about phenology.
    """
    return _cube_cols(t, ids, ("gm_",), agg, "geomedian").add_prefix("gmA_")


def _block_obscomp(t: Tables, ids: pd.Index, index: str | None = None,
                   agg: str = DEFAULT_AGG) -> pd.DataFrame:
    """Composites over the *real observations* rather than over the interpolated curve.

    Distinct from ``composite``: that one inherits the smoother's fingerprint (rollWindow=5
    and the interpolation onto 52 steps), this one does not. Comparing the two isolates what
    the smoothing did.
    """
    pref = ("obs_",) if index is None else (f"obs_{index}_",)
    return _cube_cols(t, ids, pref, agg, "observation composite").add_prefix("oc_")


def _block_seas(t: Tables, ids: pd.Index, index: str | None = None,
                agg: str = DEFAULT_AGG) -> pd.DataFrame:
    """Median per austral season plus the summer-minus-winter contrast.

    Phenology reduced to four numbers. Between the annual composite (no time at all) and the
    52-step curve (all of it), this is the intermediate rung: if it recovers most of the
    curve's advantage, the useful part of the temporal signal is an amplitude, not a shape.
    """
    pref = ("seas_",) if index is None else (f"seas_{index}_",)
    return _cube_cols(t, ids, pref, agg, "seasonal").add_prefix("sea_")


def _block_clim(t: Tables, ids: pd.Index) -> pd.DataFrame:
    """Bioclimatic normals, 1971-2000, from CR2MET (``scripts/22_extract_climate.py``).

    The only block here that does not describe the 150 m window: CR2MET is a 0.05 degree
    grid, so the 1,082 plots occupy 253 distinct cells and several plots share a value. That
    is a real limitation of support and is stated rather than hidden — but it is the axis the
    GDM comparison says is missing, since geographic distance alone already reaches Spearman
    +0.435 against observed dissimilarity and everything else added +0.057 on top.

    The normal ends in 2000, before the earliest census in 2003, so it is causal for every
    plot in the subset.
    """
    path = t.derived / "climate.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python scripts/22_extract_climate.py` first "
            "(needs a Data Cube Chile connection)."
        )
    clim = pd.read_parquet(path).set_index(ID_COL)
    return clim.reindex(ids)


def _block_svh(t: Tables, ids: pd.Index) -> pd.DataFrame:
    """Spectral variation between the 25 pixels — axis A of docs/01_state_of_the_art.md.

    Two warnings are designed in rather than discovered afterwards. Dispersion is confounded
    with level (measured r = 0.22-0.52 depending on index), so ``_cv`` is carried beside
    ``_sd`` and any reported correlation must be the partial one controlling for the mean.
    And the sign is expected to be *negative* for richness in ndvi/kndvi/nbr here, opposite
    to the classical spectral-variation hypothesis; if it holds up it is a result, not a bug.
    """
    cube = _cube(t)
    keep = [c for c in cube.columns
            if (c.endswith("_sd") or c.endswith("_cv"))
            and c.startswith(("gm_", "obs_", "seas_"))]
    sub = cube.reindex(ids)[keep].copy()
    return sub.add_prefix("svh_")


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------

def build_design(spec: str, index: str | None = None, derived: str = "data/derived",
                 px: str = "mean5x5", circular_doy: bool = False,
                 ids: pd.Index | None = None,
                 agg: str = DEFAULT_AGG) -> tuple[pd.DataFrame, pd.Index]:
    """Assemble a design matrix from ``+``-joined block names.

    Phenology-bearing blocks: ``lsp``, ``lsp_ctr``, ``curve``, ``qc``, plus ``lsp_all`` /
    ``curve_all`` which stack all five vegetation indices side by side.

    Shape-free controls (docs/09_predictors.md): ``composite`` / ``composite_all``,
    ``contrast``, ``gm``, ``obscomp`` / ``obscomp_all``, ``seas`` / ``seas_all``, ``svh``.

    Environment: ``clim`` (CR2MET bioclimatic normals; coarser support than the rest).

    Context: ``topo``, ``topo_ctr``, ``area``, ``coords``, ``year``.

    ``agg`` chooses how the 25 pixels are collapsed for the cube-derived blocks — one of
    ``median``, ``mean``, ``trimmed``, ``center``. It is a screened factor, not a default
    nobody looked at.

    Returns the frame (rows in canonical plot order, NaN preserved) and the plot id index.
    Imputation and scaling happen later, per fold, in :class:`Preprocessor` — doing them here
    would leak test-fold statistics into the training data.
    """
    t = load_tables(derived)
    ids = pd.Index(t.plots[ID_COL]) if ids is None else pd.Index(ids)
    needs_index = {"lsp", "lsp_ctr", "curve", "qc", "composite"}
    parts: list[pd.DataFrame] = []

    for name in spec.split("+"):
        name = name.strip()
        if name in needs_index and index is None:
            raise ValueError(f"block {name!r} requires --index")
        if name == "lsp":
            parts.append(_block_lsp(t, ids, index, centre=False, circular_doy=circular_doy))
        elif name == "lsp_ctr":
            parts.append(_block_lsp(t, ids, index, centre=True, circular_doy=circular_doy))
        elif name == "curve":
            parts.append(_block_curve(t, ids, index, px=px))
        elif name == "qc":
            parts.append(_block_qc(t, ids, index))
        elif name == "topo":
            parts.append(_block_topo(t, ids))
        elif name == "topo_ctr":
            parts.append(_block_topo(t, ids, centre=True))
        elif name == "area":
            parts.append(_block_area(t, ids))
        elif name == "coords":
            parts.append(_block_coords(t, ids))
        elif name == "year":
            parts.append(_block_year(t, ids))
        elif name == "lsp_all":
            for ix in INDICES:
                b = _block_lsp(t, ids, ix, centre=False, circular_doy=circular_doy)
                parts.append(b.add_prefix(f"{ix}_"))
        elif name == "curve_all":
            for ix in INDICES:
                parts.append(_block_curve(t, ids, ix, px=px).add_prefix(f"{ix}_"))
        elif name == "composite":
            parts.append(_block_composite(t, ids, index, px=px))
        elif name == "composite_all":
            for ix in INDICES:
                parts.append(_block_composite(t, ids, ix, px=px).add_prefix(f"{ix}_"))
        elif name == "contrast":
            parts.append(_block_contrast(t, ids, px=px))
        elif name == "gm":
            parts.append(_block_gm(t, ids, agg=agg))
        elif name == "obscomp":
            parts.append(_block_obscomp(t, ids, index=index, agg=agg))
        elif name == "obscomp_all":
            parts.append(_block_obscomp(t, ids, index=None, agg=agg))
        elif name == "seas":
            parts.append(_block_seas(t, ids, index=index, agg=agg))
        elif name == "seas_all":
            parts.append(_block_seas(t, ids, index=None, agg=agg))
        elif name == "svh":
            parts.append(_block_svh(t, ids))
        elif name == "clim":
            parts.append(_block_clim(t, ids))
        else:
            raise ValueError(f"unknown feature block {name!r}")

    X = pd.concat(parts, axis=1)
    X.index = ids
    if X.columns.duplicated().any():
        dup = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature columns: {dup[:5]}")
    return X, ids


#: The context vector every DL family concatenates before its head. Keeping it identical
#: across MLP / 1D-CNN / 2D-CNN is what makes the "does the second dimension help?"
#: comparison clean — the families differ only in how they read the curve.
CONTEXT_SPEC = "topo+area"


class Preprocessor:
    """Fold-local median imputation, NaN indicators and optional standardisation.

    Fitted on the training rows only. The *set* of indicator columns is decided from the full
    design matrix rather than from the training fold, so the column layout is identical across
    folds and models; only the imputed values and the scaling statistics come from training
    data. Missingness of a predictor is a property of the predictor, not of the response, so
    this is a structural decision rather than leakage.
    """

    def __init__(self, standardise: bool = True, add_indicators: bool = True):
        self.standardise = standardise
        self.add_indicators = add_indicators
        self.columns_: list[str] = []
        self.indicator_cols_: list[str] = []
        self.median_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.sd_: pd.Series | None = None

    def fit(self, X_full: pd.DataFrame, fit_rows: pd.Index) -> "Preprocessor":
        self.columns_ = list(X_full.columns)
        self.indicator_cols_ = (
            [c for c in self.columns_ if X_full[c].isna().any()] if self.add_indicators else []
        )
        tr = X_full.loc[fit_rows]
        self.median_ = tr.median(numeric_only=True)
        if self.median_.isna().any():                     # a column all-NaN in this fold
            self.median_ = self.median_.fillna(X_full.median(numeric_only=True)).fillna(0.0)
        if self.standardise:
            filled = tr.fillna(self.median_)
            self.mean_ = filled.mean()
            sd = filled.std(ddof=0)
            self.sd_ = sd.where(sd > 1e-12, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("Preprocessor.fit must be called first")
        ind = [X[c].isna().astype(float).rename(f"{c}_isna") for c in self.indicator_cols_]
        out = X[self.columns_].fillna(self.median_)
        if self.standardise:
            out = (out - self.mean_) / self.sd_
        if ind:
            out = pd.concat([out] + ind, axis=1)
        return out.to_numpy(dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return self.columns_ + [f"{c}_isna" for c in self.indicator_cols_]
