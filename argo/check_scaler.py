#!/usr/bin/env python3
"""Gate 1 of `scripts/74_check_map_consistency.py`, isolated so it runs in the Argo image.

WHY THIS EXISTS SEPARATELY. The checkpoints are loaded with
``torch.load(weights_only=False)``, which unpickles real Python objects -- among them
``ck["target_scaler"]``, a fitted scikit-learn ``PowerTransformer`` whose ``.transform``
is called on every prediction and again in the smearing inverse
(``biodiv.mapinfer`` lines around 316-380). ``docs/21`` section 7 of the private repo
records that those pickles were written by scikit-learn 1.3.1 and read back under 1.8.0,
raising ``InconsistentVersionWarning``, and names this round trip as the check that says
whether the mismatch matters. That check was run on the pod. The Argo image is a third
environment and has never been checked.

This is the failure mode worth spending a pod on: a torch version that cannot load the
weights fails loudly at ``load_state_dict``. A scikit-learn version that unpickles a
fitted transformer it does not fully understand does not fail at all -- it returns
different numbers, and the maps look plausible.

The full gate also compares inputs and outputs against the training path, which needs the
curves, substrates and design matrices -- i.e. the private repo's whole data tree. Nothing
here needs any of that: the target values come from the same padded parquets already
staged in S3 for inference, through the same ``mapinfer.training_targets`` the inference
run itself calls for range clipping.

LCBD is the channel to watch. Its Yeo-Johnson lambda is about -4.2 and its scale about
3.5e-6 (``docs/21`` D9), so it is where a subtly different transform shows up first.

Usage:
    python argo/check_scaler.py --ckpt-dir /work/assets/models/final \
        --derived /work/assets/derived
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapinfer as mi          # noqa: E402

# Pinned in Biodiversity_Chile/requirements.txt -- the environment the results in
# docs/10_findings.md were produced in. Reported, not enforced: a difference is a
# question for the round trip below to answer, not a reason to refuse to run.
PINNED = {"numpy": "2.3.5", "pandas": "3.0.3", "scipy": "1.17.1",
          "sklearn": "1.8.0", "torch": "2.12.0"}


def versions() -> None:
    print(f"python       {sys.version.split()[0]}  |  {platform.platform()}")
    print(f"\n{'module':<12} {'this image':<16} {'requirements.txt':<18} ")
    for name in list(PINNED) + ["joblib", "rasterio", "datacube"]:
        try:
            v = importlib.import_module(name).__version__
        except Exception as e:                      # noqa: BLE001
            v = f"MISSING ({type(e).__name__})"
        want = PINNED.get(name, "")
        flag = "" if not want else ("same" if v == want else "DIFFERS")
        print(f"{name:<12} {v:<16} {want:<18} {flag}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default="/work/assets/models/final", dest="ckpt_dir")
    p.add_argument("--derived", default="/work/assets/derived")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="max relative error allowed on inverse_transform(transform(y))")
    args = p.parse_args()

    versions()

    ckpts = sorted(Path(args.ckpt_dir).glob("model_seed*.pt"))
    if not ckpts:
        print(f"\nFAIL  no model_seed*.pt under {args.ckpt_dir}")
        return 1

    # Record rather than silence: the warning's presence is itself the finding.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ens = mi.FacetEnsemble([str(c) for c in ckpts])
    unpickle_warnings = [str(w.message)[:160] for w in caught
                         if "Version" in type(w.message).__name__]

    print(f"\n{len(ckpts)} checkpoints loaded; targets {list(ens.targets)}")
    if unpickle_warnings:
        print(f"{len(unpickle_warnings)} unpickling version warning(s):")
        for w in unpickle_warnings[:3]:
            print(f"  {w}")
    else:
        print("no unpickling version warnings")

    y = mi.training_targets(args.derived, list(ens.targets))
    # Same NaN handling as the original gate: a column's own median stands in, so the
    # round trip is exercised on real values rather than on whatever NaN propagates to.
    yf = np.where(np.isfinite(y), y, np.nanmedian(y, axis=0))
    print(f"training targets {yf.shape} from {args.derived}")

    ok_all = True
    print(f"\n{'target':<24} {'scale':>12} {'max rel err':>12}  verdict")
    for i, member in enumerate(ens.members):
        if i:                                # every seed carries its own fitted scaler
            break
        sc = member.scaler
        back = sc.inverse_transform(sc.transform(yf))
        rel = np.nanmax(np.abs(back - yf) / (np.abs(yf) + 1e-12), axis=0)
        for j, t in enumerate(ens.targets):
            good = bool(rel[j] < args.tolerance)
            ok_all &= good
            print(f"{t:<24} {np.nanmedian(np.abs(yf[:, j])):>12.3e} "
                  f"{rel[j]:>12.3e}  {'ok' if good else 'FAIL'}")

    # Every seed's scaler, not just seed 0: the ensemble averages back-transformed
    # predictions, so one bad scaler out of five is enough to bias the mean.
    worst = 0.0
    for member in ens.members:
        sc = member.scaler
        r = np.nanmax(np.abs(sc.inverse_transform(sc.transform(yf)) - yf)
                      / (np.abs(yf) + 1e-12))
        worst = max(worst, float(r))
    print(f"\nworst relative error across all {len(ens.members)} seeds: {worst:.3e}")
    ok_all &= worst < args.tolerance

    print("\nGATE PASS -- the scaler computes here what it computed when fitted"
          if ok_all else
          "\nGATE FAIL -- the unpickled scaler does not round trip; DO NOT produce maps")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
