#!/usr/bin/env python3
"""Extract per-plot topography from the Copernicus GLO-30 DEM.

Derivatives are computed over a wide window and **then** cropped to the plot window:
slope, aspect and curvature are neighbourhood operators, so computing them on a 5x5 crop
would contaminate half the pixels with edge effects.

Variables produced:

- ``elevation`` — metres above sea level.
- ``slope`` — degrees.
- ``aspect`` — degrees; unusable as a raw number because of the 0/360 discontinuity,
  hence the two variables below.
- ``northness`` / ``eastness`` — ``cos``/``sin`` of aspect: continuous and directly
  interpretable. In Mediterranean Chile north-facing slopes carry the highest heat load.
- ``heat_load`` — McCune & Keon (2002) heat load index, adapted to the southern hemisphere
  by folding aspect about 315 degrees (the warmest slope is NW, not SW).
- ``tpi`` — topographic position index: elevation minus the neighbourhood mean.
  Separates valley bottoms from ridge crests, which in Mediterranean drylands separates
  plant communities.
- ``tri`` — terrain ruggedness index (Riley).
- ``curvature`` — total curvature (Laplacian); positive means divergent.

The site table is any parquet carrying ``X``/``Y`` (EPSG:32719), ``lat`` and a grouping
``cell``, so the same code serves Parcelas-CL and the Living_Trees_Chile plots:

    python scripts/03_extract_topography.py
    python scripts/03_extract_topography.py \\
        --subset data/derived/living_trees/sites.parquet --id-col site_id --id-name site_id \\
        --out-dir data/derived/living_trees/topography --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEM_PRODUCT = "copernicus_dem_30"
CRS = "EPSG:32719"


def terrain(elev: np.ndarray, res: float, lat_deg: float,
            tpi_radius: int = 3) -> dict[str, np.ndarray]:
    """Topographic derivatives over a projected grid of spacing ``res`` metres."""
    dzdy, dzdx = np.gradient(elev, res, res)

    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)

    # Cartographic convention: 0 = north, increasing clockwise, and pointing **downslope**
    # -- the direction the slope faces, which is what "north-facing" means.
    # `dzdy` is the derivative along the array rows, which run north -> south in the loaded
    # DEM, so the downslope vector is (-dzdx east, +dzdy north): taking `atan2(-dzdy, dzdx)`
    # instead returns the uphill bearing, 180 degrees off, and silently flips the sign of
    # `northness`, `eastness` and the aspect term of `heat_load`.
    # Checked against `gdaldem aspect` on tilted planes in `tests/test_topography.py`.
    aspect_deg = (90.0 - np.degrees(np.arctan2(dzdy, -dzdx))) % 360.0
    # Aspect is undefined on flat terrain; forcing NaN avoids inventing an orientation
    # that the model would later treat as signal. It has to be masked *before* the sine and
    # cosine are taken, or `northness`/`eastness` keep the orientation of the rounding noise
    # while `aspect` and `heat_load` are honest about not having one.
    aspect_deg = np.where(slope_deg < 0.5, np.nan, aspect_deg)
    aspect_rad = np.radians(aspect_deg)

    # McCune & Keon (2002), folded for the southern hemisphere (warmest slope ~ NW = 315)
    folded = np.radians(np.abs(180.0 - np.abs(aspect_deg - 315.0)))
    lat_rad = np.radians(abs(lat_deg))
    heat_load = (
        0.339
        + 0.808 * np.cos(lat_rad) * np.cos(slope_rad)
        - 0.196 * np.sin(lat_rad) * np.sin(slope_rad)
        - 0.482 * np.cos(folded) * np.sin(slope_rad)
    )

    size = 2 * tpi_radius + 1
    mean_neigh = ndimage.uniform_filter(elev, size=size, mode="nearest")
    tpi = elev - mean_neigh

    # Riley TRI: root of the summed squared differences against the 8 neighbours
    sq = np.zeros_like(elev, dtype=float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sq += (elev - np.roll(np.roll(elev, dy, 0), dx, 1)) ** 2
    tri = np.sqrt(sq)

    curvature = ndimage.laplace(elev.astype(float)) / (res**2)

    return {
        "elevation": elev.astype("float32"),
        "slope": slope_deg.astype("float32"),
        "aspect": aspect_deg.astype("float32"),
        "northness": np.cos(aspect_rad).astype("float32"),
        "eastness": np.sin(aspect_rad).astype("float32"),
        "heat_load": heat_load.astype("float32"),
        "tpi": tpi.astype("float32"),
        "tri": tri.astype("float32"),
        "curvature": curvature.astype("float32"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="data/derived/plots_subset.parquet")
    p.add_argument("--id-col", default="PlotObservationID",
                   help="site key column in --subset")
    p.add_argument("--id-name", default="plot_id",
                   help="name that key takes in the outputs")
    p.add_argument("--out-dir", default="data/derived/topography")
    p.add_argument("--patch", type=int, default=5)
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--halo", type=int, default=20,
                   help="extra pixels per side before differentiating (avoids edge effects)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="keep the sites already in --out-dir and load only the rest")
    p.add_argument("--flush-every", type=int, default=100,
                   help="checkpoint interval, in cells")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plots = pd.read_parquet(args.subset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl_path, nc_path = out_dir / "topography.parquet", out_dir / "topography_patches.nc"

    groups = list(plots.groupby("cell", sort=True))
    print(f"plots: {len(plots)} | cells: {len(groups)}")
    if args.dry_run:
        print("product:", DEM_PRODUCT, "| variables:",
              "elevation slope aspect northness eastness heat_load tpi tri curvature")
        print(f"key: {args.id_col} -> {args.id_name} | out: {tbl_path}")
        return

    import datacube
    from datacube.utils.aws import configure_s3_access

    configure_s3_access(aws_unsigned=False, requester_pays=True)
    dc = datacube.Datacube(app="biodiv_topo")

    if args.limit:
        groups = groups[: args.limit]

    rows, arrays = [], {}
    half = args.patch // 2

    def flush() -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.to_parquet(tbl_path, index=False)
        if arrays:
            xr.concat(list(arrays.values()), dim=args.id_name).assign_coords(
                {args.id_name: list(arrays)}).to_netcdf(nc_path)
        return df

    if args.resume and tbl_path.exists():
        rows = pd.read_parquet(tbl_path).to_dict("records")
        if nc_path.exists():
            with xr.open_dataset(nc_path) as prev:
                prev = prev.load()
            arrays = {k: prev.sel({args.id_name: k}, drop=True)
                      for k in prev[args.id_name].values.tolist()}
        print(f"resume: {len(rows)} sites already extracted, {len(arrays)} patches")
    done = {r[args.id_name] for r in rows}

    for gi, (cell, g) in enumerate(groups, 1):
        if done and not (set(g[args.id_col]) - done):
            continue
        buf = (args.halo + args.patch) * args.resolution
        try:
            dem = dc.load(
                product=DEM_PRODUCT,
                x=(g["X"].min() - buf, g["X"].max() + buf),
                y=(g["Y"].min() - buf, g["Y"].max() + buf),
                crs=CRS, output_crs=CRS, resolution=(-args.resolution, args.resolution),
                resampling="bilinear",
            )
            if dem.sizes.get("time", 0):
                dem = dem.isel(time=0)
            elev = dem["elevation"].values.astype(float)
            elev = np.where(elev <= -1000, np.nan, elev)
            if np.isnan(elev).all():
                raise ValueError("empty DEM")
            # fill isolated holes so the gradients do not propagate NaN
            if np.isnan(elev).any():
                elev = np.where(np.isnan(elev),
                                ndimage.generic_filter(np.nan_to_num(elev), np.nanmean,
                                                       size=3, mode="nearest"),
                                elev)
            terr = terrain(elev, args.resolution, float(g["lat"].mean()))
        except Exception as e:
            print(f"[{gi}/{len(groups)}] {cell}: FAILED -- {type(e).__name__}: {e}")
            continue

        for _, plot in g.iterrows():
            if plot[args.id_col] in done:
                continue
            iy = int(np.abs(dem.y.values - plot["Y"]).argmin())
            ix = int(np.abs(dem.x.values - plot["X"]).argmin())
            y0, y1 = max(0, iy - half), min(elev.shape[0], iy + half + 1)
            x0, x1 = max(0, ix - half), min(elev.shape[1], ix + half + 1)

            rec = {args.id_name: plot[args.id_col]}
            patch_vars = {}
            for name, arr in terr.items():
                win = arr[y0:y1, x0:x1]
                patch_vars[name] = xr.DataArray(win, dims=("py", "px"))
                rec[name] = float(arr[iy, ix])          # value at the plot pixel
                rec[f"{name}_mean"] = float(np.nanmean(win))
                rec[f"{name}_std"] = float(np.nanstd(win))
            rows.append(rec)
            arrays[plot[args.id_col]] = xr.Dataset(patch_vars)

        print(f"[{gi}/{len(groups)}] {cell}: {len(g)} plots", flush=True)
        if gi % args.flush_every == 0:
            flush()

    df = flush()
    print(f"\n{len(df)} plots with topography -> {out_dir}/")
    if len(df):
        print(df[["elevation", "slope", "heat_load", "tpi"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
