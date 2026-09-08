"""Random Forest and tabular MLP.

**Why one forest per target rather than sklearn's multi-output forest.** A multi-output
forest requires a complete target matrix; it cannot mask. Three of the nine targets are NaN
for 536 of the 1,082 plots by design (the cover abundance tier), so a multi-output fit would
either drop half the data or impute it — both of which the project has explicitly ruled out.
Per-target forests also happen to be the stronger baseline, which matters: the point of the
RF tier is to be a fair opponent for the CNN, not a straw man. A single multi-output forest
is still fitted once, as run `RF07`, so "does joint modelling help?" is answerable inside the
RF family too.

**The target power transform is applied to RF as well.** A forest is invariant to monotone
transforms of its *features* but not of its *targets*: splits are chosen to minimise
within-node variance, and on a target with skew 2.8 that objective is dominated by the tail.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

#: Deliberately not tuned per run. Hyperparameter search inside each of ~500 runs would cost
#: more than the whole rest of the matrix and would make the families incomparable; these are
#: standard regression defaults with a leaf size raised slightly for n<=870 training rows.
RF_DEFAULTS = dict(
    n_estimators=500,
    min_samples_leaf=2,
    max_features=0.33,
    bootstrap=True,
    n_jobs=-1,
)


def make_rf(seed: int = 0, **kw) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=seed, **(RF_DEFAULTS | kw))


def save_rf(rf: RandomForestRegressor, path: str | Path) -> None:
    """Persist a fitted forest. No run in this project has ever saved one -- every driver
    fits, predicts and discards, so re-scoring new data (a map, a sensitivity check) meant
    refitting from scratch. `joblib` rather than `torch.save`/`pickle` directly: it is
    scikit-learn's own recommended serialisation for estimators with large numpy arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, path)


def load_rf(path: str | Path) -> RandomForestRegressor:
    return joblib.load(path)


def fit_predict_rf(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                   seed: int = 0, importance: bool = False,
                   save_dir: str | Path | None = None, target_names: list[str] | None = None,
                   **kw) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """One forest per target column, each fitted on that target's observed rows only.

    Returns ``(pred, importances, oob_residuals)``. ``pred`` is (n_test, n_targets);
    ``oob_residuals`` is (n_train, n_targets) with NaN where the target was unobserved, and
    feeds Duan's smearing correction in :func:`biodiv.targets.inverse_with_smearing`.
    Out-of-bag rather than in-sample residuals: a forest's in-sample residuals are far too
    small and would leave most of the retransformation bias in place.

    A target with fewer than 20 observed training rows yields NaN predictions rather than a
    forest fitted on noise.

    ``save_dir``, if given (with matching ``target_names``), persists each fitted forest via
    :func:`save_rf` as it is fitted -- the only place a caller can reach the estimator before
    it goes out of scope, since this function fits-and-discards by design otherwise.
    """
    n_te, n_t = X_te.shape[0], Y_tr.shape[1]
    pred = np.full((n_te, n_t), np.nan)
    resid = np.full((X_tr.shape[0], n_t), np.nan)
    imps = np.full((X_tr.shape[1], n_t), np.nan) if importance else None
    for j in range(n_t):
        ok = np.isfinite(Y_tr[:, j])
        if ok.sum() < 20:
            continue
        rf = make_rf(seed=seed, oob_score=True, **kw).fit(X_tr[ok], Y_tr[ok, j])
        pred[:, j] = rf.predict(X_te)
        oob = getattr(rf, "oob_prediction_", None)
        if oob is not None:
            resid[np.flatnonzero(ok), j] = Y_tr[ok, j] - oob
        if importance:
            imps[:, j] = rf.feature_importances_
        if save_dir is not None and target_names is not None:
            save_rf(rf, Path(save_dir) / f"rf_{target_names[j]}.joblib")
    return pred, imps, resid


def fit_predict_rf_multioutput(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                               seed: int = 0, **kw) -> np.ndarray:
    """Single joint forest. Only valid on target columns with no NaN (i.e. TARGETS_MAIN)."""
    if not np.isfinite(Y_tr).all():
        raise ValueError("multi-output RF cannot mask NaN; restrict to the complete targets")
    rf = make_rf(seed=seed, **kw).fit(X_tr, Y_tr)
    return rf.predict(X_te)


class MLPMulti(nn.Module):
    """Multi-output MLP over a tabular design matrix.

    The NaN targets never reach the network: the mask lives in the loss. BatchNorm rather
    than LayerNorm because batches of 64 over ~700 training rows are stable and the batch
    noise is itself useful regularisation at this sample size.

    Measured parameter counts (n_out=9):

    ========  ==========  ===================  =====================  =================
    option    hidden      lsp+topo+area d=46   curve+topo+area d=80   5-index d=288
    ========  ==========  ===================  =====================  =================
    MLP-A     (64, 32)                  6,663                  7,687            20,999
    MLP-B     (128, 64)                17,415                 19,463            46,087
    MLP-C     (256,128,64)             59,143                 63,239           116,487
    ========  ==========  ===================  =====================  =================

    MLP-B is the default for single-index blocks, MLP-A for the five-index stack. MLP-C is
    over the 50k budget and is kept only as an over-fitting control.
    """

    WIDTHS = {"A": (64, 32), "B": (128, 64), "C": (256, 128, 64)}

    def __init__(self, d_in: int, hidden: tuple[int, ...] = (128, 64), n_out: int = 9,
                 p_drop: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(p_drop)]
            prev = w
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        # ctx is accepted so the trainer can treat every family through one call signature;
        # for the MLP the context columns are already inside x.
        if ctx is not None:
            x = torch.cat([x, ctx], dim=1)
        return self.head(self.net(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
