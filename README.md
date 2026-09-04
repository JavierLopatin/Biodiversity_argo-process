# Biodiversity Chile — map inference pipeline

Code that generates the biodiversity map tiles (2000-2026, 30 m) from the
Chilean Open Data Cube. This repo contains **only the pipeline code** — no
data, no model weights, no logs.

## Requirements not included here

- Data Cube Chile ODC index access (`DB_HOSTNAME`, `DB_PORT`, `DB_DATABASE`,
  `DB_USERNAME`, `DB_PASSWORD`) — the pipeline calls `dc.load` against the
  Chilean datacube index, not a public STAC catalog. It only runs inside the
  EASI environment where that index is reachable.
- Model artifacts (`model_seed{0..4}.pt`, `oof_predictions.csv`) and derived
  data (`*_unified_padded.parquet`, `tiles_native_10km_run.csv`).
- Environment variables: `BIODIV_UNIFIED=1`, `BIODIV_CURVES=_raw100`.

## Layout

- `scripts/73_map_inference.py` — entry point, one tile per invocation.
- `scripts/03_extract_topography.py` — terrain features (aspect-convention
  fixed).
- `src/biodiv/` — pipeline modules: cube loading, feature extraction,
  curve/composite construction, model inference, tile task orchestration.

## Tile scheme

Tiles are 9,990 m squares in EPSG:32719 (UTM 19S), addressed as `t<ix>_<iy>`
with `xmin = ix*9990`, `ymin = iy*9990`. A tile is complete once all 27
annual rasters (2000-2026) exist for it.
