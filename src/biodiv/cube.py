"""Landsat C2L2 loading, scaling and masking from Data Cube Chile.

Three things that, if omitted, silently invalidate every downstream index:

1. ``usgs-landsat`` is a **requester-pays** bucket: ``configure_s3_access`` must be called
   before the first load or rasterio raises AccessDenied.
2. Landsat C2 Level-2 SR is delivered as scaled integers: ``reflectance = DN*0.0000275 - 0.2``.
3. ``qa_pixel`` flag values are product-specific — ``dilated_cloud`` takes
   ``not_dilated``/``dilated``, not ``not_high_confidence``; verified against
   ``masking.describe_variable_flags``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from datacube.utils import masking

# Collection 2 Level-2 Surface Reflectance scale/offset
SR_SCALE, SR_OFFSET = 0.0000275, -0.2

# Soil-adjustment factor for SAVI (Huete 1988). 0.5 is the canonical value for
# intermediate vegetation density, which is the matorral case.
SAVI_L = 0.5

# Vegetation indices computed for every plot. Each gets its own PhenoShape curve and its
# own LSP metrics, so index choice is a factor in the benchmark rather than a fixed default.
INDEX_NAMES = ["ndvi", "evi", "kndvi", "nbr", "savi"]
BAND_VARS = ["band_blue", "band_green", "band_red", "band_nir", "band_swir1", "band_swir2"]

# Operating range of each sensor: avoids empty loads and the cost of querying for them.
SENSOR_YEARS = {
    "landsat5_c2l2_sr": (1984, 2011),
    "landsat7_c2l2_sr": (1999, 2022),
    "landsat8_c2l2_sr": (2013, 2100),
    "landsat9_c2l2_sr": (2021, 2100),
}

# Aliases verified present across L5/7/8/9 (primary names differ: red is SR_B3 on L5/7
# but SR_B4 on L8/9).
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "qa_pixel"]

# Keep: no nodata, no high-confidence cloud/shadow/cirrus/snow, no cloud dilation.
#
# `cirrus` exists ONLY on L8/L9 (OLI); L5/L7 (TM/ETM+) have no such band. Requesting the
# flag on a product that does not define it makes `make_mask` raise `Unknown flag:
# "cirrus"`, and because the mask used to be built after concatenating sensors, whether it
# failed depended on which product happened to come first. The mask is therefore computed
# **per product, before concatenation**, filtered against the flags actually defined.
QA_KEEP = dict(
    nodata=False,
    cloud="not_high_confidence",
    cloud_shadow="not_high_confidence",
    dilated_cloud="not_dilated",
    cirrus="not_high_confidence",
    snow="not_high_confidence",
)


def clear_mask(qa: xr.DataArray) -> xr.DataArray:
    """Usable-pixel mask, applying only the flags this product actually defines."""
    available = set(masking.describe_variable_flags(qa).index)
    flags = {k: v for k, v in QA_KEEP.items() if k in available}
    return masking.make_mask(qa, **flags)


def products_for(year_start: int, year_end: int) -> list[str]:
    """Landsat products with coverage inside the window."""
    return [
        p for p, (a, b) in SENSOR_YEARS.items()
        if not (b < year_start or a > year_end)
    ]


def load_window(dc, bbox_utm, year_start: int, year_end: int,
                resolution: int = 30, crs: str = "EPSG:32719",
                dask_chunks: dict | None = None) -> xr.Dataset | None:
    """Load and concatenate the Landsat sensors active within the window.

    ``bbox_utm`` is ``(xmin, ymin, xmax, ymax)`` in ``crs``. Returns ``None`` when no
    observation exists (e.g. the 2012 gap outside the L7 path).

    With ``dask_chunks={"time": 1}`` the load is lazy and the COG reads are spread across
    cluster workers. That is the difference between reading ~100 scenes from S3 serially
    or in parallel, and it dominates total runtime.
    """
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
    for prod in products_for(year_start, year_end):
        ds = dc.load(product=prod, measurements=BANDS, **query)
        if ds.sizes.get("time", 0) == 0:
            continue
        ds = ds.assign_coords(
            sensor=("time", np.array([prod.replace("_c2l2_sr", "")] * ds.sizes["time"]))
        )
        ds["clear"] = clear_mask(ds["qa_pixel"])
        parts.append(ds)
    if not parts:
        return None
    ds = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time")
    return ds


def to_indices(ds: xr.Dataset) -> tuple[xr.Dataset, xr.DataArray]:
    """Scale, mask, and compute the vegetation indices used as the phenological substrate.

    Returns ``(data, clear)``. ``data`` carries the vegetation indices *and* the scaled
    bands they were built from; ``clear`` is the boolean per-observation, per-pixel mask — kept so quality covariates can be derived from it. The mask is already computed
    per product in ``load_window`` (see ``clear_mask``).
    """
    clear = ds["clear"]

    def band(name: str) -> xr.DataArray:
        # nodata = 0 in every SR band; elsewhere apply the scale/offset
        raw = ds[name].where(ds[name] > 0)
        return (raw * SR_SCALE + SR_OFFSET).where(clear)

    scaled = {b: band(b) for b in ("blue", "green", "red", "nir", "swir1", "swir2")}
    blue, red, nir, swir2 = scaled["blue"], scaled["red"], scaled["nir"], scaled["swir2"]

    ndvi = ((nir - red) / (nir + red)).clip(-1, 1)
    evi = (2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)).clip(-1, 1)
    kndvi = np.tanh(ndvi**2)
    nbr = ((nir - swir2) / (nir + swir2)).clip(-1, 1)
    # SAVI (Huete 1988) with the canonical L = 0.5. Matters here: sclerophyll matorral
    # leaves a large exposed-soil fraction inside a 30 m pixel, and NDVI both saturates and
    # takes on a soil-brightness bias under that background. The soil-adjustment term damps
    # it. Bounded by 1+L rather than 1.
    savi = (((nir - red) / (nir + red + SAVI_L)) * (1.0 + SAVI_L)).clip(-1.5, 1.5)

    out = xr.Dataset({"ndvi": ndvi, "evi": evi, "kndvi": kndvi, "nbr": nbr, "savi": savi})
    # Keep the scaled bands too. Indices are lossy: SAVI cannot be recovered from NDVI
    # because it needs NIR and Red separately. Storing the bands means any further index
    # (MSAVI, NDMI, NIRv, ...) is a local computation instead of another pass over S3.
    out = out.assign({f"band_{b}": v for b, v in scaled.items()})
    out = out.assign_coords(
        doy=("time", ds.time.dt.dayofyear.values),
        year=("time", ds.time.dt.year.values),
        sensor=("time", ds.sensor.values),
    )
    return out, clear


def observation_stats(da: xr.DataArray) -> dict:
    """Predictor quality covariates. These are exclusion criteria, not decoration.

    The largest gap along the DOY axis matters more than the total observation count:
    25 observations concentrated in summer do not constrain SOS, and in Mediterranean
    Chile cloud cover concentrates precisely at the start of the growing season — so the
    missingness is not random with respect to the metric being estimated.
    """
    valid_t = da.notnull().any(dim=[d for d in da.dims if d != "time"])
    n_obs = int(valid_t.sum())
    if n_obs == 0:
        return dict(n_obs=0, max_doy_gap=np.nan, mean_doy_gap=np.nan,
                    n_obs_per_year={}, frac_valid_px=0.0)
    doy = np.sort(np.asarray(da.doy.values)[valid_t.values])
    gaps = np.diff(doy) if len(doy) > 1 else np.array([np.nan])
    years = np.asarray(da.year.values)[valid_t.values]
    uy, cy = np.unique(years, return_counts=True)
    return dict(
        n_obs=n_obs,
        max_doy_gap=float(np.nanmax(gaps)) if len(gaps) else np.nan,
        mean_doy_gap=float(np.nanmean(gaps)) if len(gaps) else np.nan,
        n_obs_per_year={int(y): int(c) for y, c in zip(uy, cy)},
        frac_valid_px=float(da.notnull().mean()),
    )
