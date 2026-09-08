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
  data (`*_unified_padded.parquet`, `tiles_native_10km.csv`).
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

## Argo (EASI cluster)

`argo/` runs the same entry point as a fan-out over tiles, one pod per chunk.

| file | purpose |
| --- | --- |
| `argo/test-s3-access.yaml` | credential smoke test — run this first |
| `argo/probe-s3.yaml` | read-only: what is already in S3 (writes nothing) |
| `argo/workflow-template.yaml` | `biodiv-map-inference` WorkflowTemplate |
| `argo/tile_progress.py` | S3 vs. tile list; also usable standalone |
| `argo/verify_run.py` | reads the written rasters back and checks them |
| `argo/upload_run_evidence.py` | copies `manifest.csv` / `run.json` to S3 |

Verified cluster identity (from `test-s3-access.yaml`): namespace
`easi-workflows-argo`, service account `easi-workflows-team-sa-argo`, S3 prefix
`s3://easido-prod-dc-data-projects/easi-workflows-team/`. That test writes to S3
with **no AWS credential Secret mounted**, so the service account carries the IAM
role (IRSA); every S3 path in the template stays under that prefix for that
reason.

git-sync pulls this **public** repo, so no git credential is needed. The private
inputs (model checkpoints, `oof_predictions.csv`, derived parquet, the tile list,
MapBiomas rasters) are staged from S3 by an initContainer — the one-time upload
commands are in the header of `argo/workflow-template.yaml`.

```bash
argo submit -n easi-workflows-argo argo/test-s3-access.yaml           # 1. credentials
argo template create -n easi-workflows-argo argo/workflow-template.yaml
argo submit -n easi-workflows-argo --from workflowtemplate/biodiv-map-inference \
  -p limit-tiles=3 -p num-chunks=3                                    # 2. 3-tile test
```

### Single-tile test, and checking it afterwards

One tile for one year is the cheapest end-to-end test — it exercises git-sync,
the asset staging, the ODC connection, the model load and the S3 write, without
paying for 27 years of Landsat reads:

```bash
argo submit -n easi-workflows-argo --from workflowtemplate/biodiv-map-inference \
  -p limit-tiles=1 -p num-chunks=1 -p years=2020-2020 -p tag=smoke1 --watch
```

A green workflow only means every pod exited 0. The pipeline is written so that
a tile with no clear-sky observations, an empty native mask, or smearing
silently disabled still exits 0 and writes a plausible-looking GeoTIFF. So check
the output, not the exit code:

```bash
# what the pods recorded: one row per (tile, year), with status and error
aws s3 sync s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/maps/chile_30m_2000_2026/_runs/<workflow-name> ./evidence
cat evidence/chunk-0/smoke1-0/manifest.csv

# read the rasters back and validate them
python argo/verify_run.py \
  --s3-dest s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/maps/chile_30m_2000_2026 \
  --years 2020-2020 \
  --tiles-csv results/figures/tiles_native_10km.csv \
  --manifest evidence/chunk-0/smoke1-0/manifest.csv
```

`verify_run.py` exits non-zero and lists what is wrong. It checks CRS/pixel
size/bounds against the tile CSV, that band descriptions are consistent and end
in `n_obs, span_days, native`, that each facet band is finite over enough of the
native-masked pixels and is not constant, and that the GDAL tags say
`smearing=oof` with the expected seed count and checkpoint directory.

Still unconfirmed with EASI admins: whether `base-image` ships torch + rasterio,
and the name of the ODC user-db Secret readable by the Argo service account
(`db-config-secret`). Both are marked `EDIT-ME` in the template.
