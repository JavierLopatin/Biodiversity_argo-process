"""Signal-to-image transformations, retuned for a 52-step weekly curve.

Adapted from ``Trait_2DCNN/transforms/`` (one file per transform there; one module here).
Two changes apply to every class, and both matter more than the choice of transform:

**1. Per-sample min-max normalisation is off by default.** Every `Trait_2DCNN` transform ends
with ``(img - img.min()) / (img.max() - img.min())``. For hyperspectral reflectance that is
defensible. For a phenological curve it deletes mean greenness and seasonal amplitude — the
two strongest known correlates of productivity and richness — and hands the network only the
*shape*. ``normalize="none"`` is the default here; the per-sample behaviour is retained as an
explicit level of the `C2D-N` ablation so the cost of that convention can be quoted.

**2. The 224x224 zoom is gone.** It existed because `timm` backbones demand it. Zooming a
52-point signal to 50,176 pixels adds no information and a great deal of interpolation: for
`cwt` the shipped settings produce an image that is 99.5% interpolated. Every transform here
returns its natural size.

Retuning forced by n=52 rather than n=1721:

- ``reshape``/``serpentine``/``hilbert`` need 64 cells and have 52. **Wrap-padding, not zero
  padding**: the DOY axis is circular, so the 12 missing cells are exactly ``curve[:12]``.
  Zero padding invents a hard trough that a 3x3 kernel reads as a real phenological event.
- ``mtf`` drops from 8 quantile bins to 5: 52 points across 8 bins leaves 6.5 observations per
  bin and an 8x8 transition matrix estimated from 51 transitions, which is noise.
- ``cwt`` drops from 128 scales to 24. Scale 24 already spans half the series; scales above
  that are pure edge artefact on a 52-point signal.
- ``spectrogram`` drops ``nperseg`` from 64 to 16. As shipped, ``nperseg=64 > 52`` makes scipy
  silently truncate to a single frame. It is kept in the benchmark for completeness, with the
  expectation stated in advance that it loses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

NGS = 52


def wrap_pad(x: np.ndarray, n: int) -> np.ndarray:
    """Extend a circular signal to length ``n`` by wrapping, never by zero-filling."""
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


class BaseTransform(ABC):
    """``transform(curve: (n,)) -> (H, W) or (H, W, C)``, float32."""

    name = "base"
    n_channels = 1

    def __init__(self, normalize: str = "none"):
        if normalize not in ("none", "perSample", "global"):
            raise ValueError(f"unknown normalize={normalize!r}")
        self.normalize = normalize

    @abstractmethod
    def _build(self, x: np.ndarray) -> np.ndarray:
        ...

    def _norm(self, img: np.ndarray) -> np.ndarray:
        if self.normalize != "perSample":
            return img            # "global" is applied once, per fold, in the dataset
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        return (img - lo) / max(hi - lo, 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._norm(np.asarray(self._build(np.asarray(x, dtype=np.float64)),
                                     dtype=np.float32))

    def shape(self, n: int = NGS) -> tuple[int, int, int]:
        """(C, H, W) of the output, for building the model before touching any data."""
        out = self.transform(np.linspace(0.1, 0.6, n))
        return (1, *out.shape) if out.ndim == 2 else (out.shape[2], out.shape[0], out.shape[1])


class ReshapeTransform(BaseTransform):
    """Fold the curve into a square. Won the hyperspectral benchmark in `Trait_2DCNN`."""

    name = "reshape"

    def __init__(self, side: int | None = None, normalize: str = "none"):
        super().__init__(normalize)
        self.side = side

    def _build(self, x):
        side = self.side or int(np.ceil(np.sqrt(len(x))))
        return wrap_pad(x, side * side).reshape(side, side)


class SerpentineTransform(ReshapeTransform):
    """Reshape with alternate rows reversed, so time is continuous across row boundaries."""

    name = "serpentine"

    def _build(self, x):
        img = super()._build(x).copy()
        img[1::2] = img[1::2, ::-1]
        return img


def _d2xy(order: int, d: int) -> tuple[int, int]:
    """Hilbert curve index -> (x, y). Vendored from Trait_2DCNN/transforms/hilbert_transform.py."""
    n = 1 << order
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


class HilbertTransform(BaseTransform):
    """Space-filling-curve ordering: neighbouring weeks stay neighbours in 2-D."""

    name = "hilbert"

    def __init__(self, order: int = 3, normalize: str = "none"):
        super().__init__(normalize)
        self.order = order
        self.side = 1 << order
        self.xy = [_d2xy(order, d) for d in range(self.side * self.side)]

    def _build(self, x):
        v = wrap_pad(x, self.side * self.side)
        img = np.zeros((self.side, self.side))
        for d, (px, py) in enumerate(self.xy):
            img[py, px] = v[d]
        return img


class GAFTransform(BaseTransform):
    """Gramian angular field, summation and difference. Two channels.

    The only transform where per-sample rescaling is intrinsic rather than a convention: the
    polar encoding requires the signal in [-1, 1]. Documented rather than made optional.
    """

    name = "gaf"
    n_channels = 2

    def _build(self, x):
        lo, hi = x.min(), x.max()
        s = 2.0 * (x - lo) / max(hi - lo, 1e-8) - 1.0
        s = np.clip(s, -1.0, 1.0)
        phi = np.arccos(s)
        gasf = np.cos(phi[:, None] + phi[None, :])
        gadf = np.sin(phi[:, None] - phi[None, :])
        return np.stack([gasf, gadf], axis=-1)


class MTFTransform(BaseTransform):
    """Markov transition field over quantile bins of the curve."""

    name = "mtf"

    def __init__(self, n_bins: int = 5, normalize: str = "none"):
        super().__init__(normalize)
        self.n_bins = n_bins

    def _build(self, x):
        edges = np.unique(np.percentile(x, np.linspace(0, 100, self.n_bins + 1)[1:-1]))
        if edges.size == 0:
            return np.zeros((len(x), len(x)))
        q = np.digitize(x, edges)
        nb = int(q.max()) + 1
        trans = np.zeros((nb, nb))
        for a, b in zip(q[:-1], q[1:]):
            trans[a, b] += 1
        rows = trans.sum(axis=1, keepdims=True)
        trans = np.divide(trans, rows, out=np.zeros_like(trans), where=rows > 0)
        return trans[q[:, None], q[None, :]]


class NDITransform(BaseTransform):
    """Normalised difference between every pair of weeks — relative greenness change."""

    name = "ndi"

    def _build(self, x):
        num = x[:, None] - x[None, :]
        den = x[:, None] + x[None, :]
        return np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-8)


class CWTTransform(BaseTransform):
    """Continuous wavelet transform: scale x time. Natural for multi-scale phenology."""

    name = "cwt"

    def __init__(self, n_scales: int = 24, wavelet: str = "morl", normalize: str = "none"):
        super().__init__(normalize)
        self.n_scales = n_scales
        self.wavelet = wavelet

    def _build(self, x):
        import pywt
        coef, _ = pywt.cwt(x, np.arange(1, self.n_scales + 1), self.wavelet)
        return np.abs(coef)


class COS2DTransform(BaseTransform):
    """Two-dimensional correlation (synchronous / asynchronous), Hilbert-Noda. Two channels."""

    name = "cos2d"
    n_channels = 2

    def __init__(self, n_segments: int = 10, normalize: str = "none"):
        super().__init__(normalize)
        self.n_segments = n_segments

    def _build(self, x):
        n = len(x)
        seg = max(2, n // self.n_segments)
        rows = np.stack([x[i:i + seg] for i in range(0, n - seg + 1, max(1, seg // 2))])
        d = rows - rows.mean(axis=0, keepdims=True)
        m = d.shape[0]
        sync = d.T @ d / max(m - 1, 1)
        k = np.arange(m)
        noda = np.zeros((m, m))
        diff = k[:, None] - k[None, :]
        nz = diff != 0
        noda[nz] = 1.0 / (np.pi * diff[nz])
        asyn = d.T @ noda @ d / max(m - 1, 1)
        return np.stack([sync, asyn], axis=-1)


class SpectrogramTransform(BaseTransform):
    """Short-time Fourier transform. Marginal on 52 samples; included for completeness."""

    name = "spectrogram"

    def __init__(self, nperseg: int = 16, noverlap: int = 12, normalize: str = "none"):
        super().__init__(normalize)
        self.nperseg = nperseg
        self.noverlap = noverlap

    def _build(self, x):
        from scipy import signal
        _, _, z = signal.stft(x, nperseg=self.nperseg, noverlap=self.noverlap)
        return np.abs(z)


#: Registry consumed by ``scripts/11_run_conv.py --substrate``.
TRANSFORMS: dict[str, type[BaseTransform]] = {
    "reshape": ReshapeTransform,
    "serpentine": SerpentineTransform,
    "hilbert": HilbertTransform,
    "gaf": GAFTransform,
    "mtf": MTFTransform,
    "ndi": NDITransform,
    "cwt": CWTTransform,
    "cos2d": COS2DTransform,
    "spectrogram": SpectrogramTransform,
}


def make_transform(name: str, normalize: str = "none", **kw) -> BaseTransform:
    return TRANSFORMS[name](normalize=normalize, **kw)
