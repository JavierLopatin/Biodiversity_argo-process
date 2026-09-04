"""Small convolutional models for n ~ 1,000, and the topography fusion options.

These are deliberately **not** the `Trait_2DCNN` architectures. That project fed 1,721-band
hyperspectral signals to `timm` backbones of 4-28 M parameters, with thousands of training
samples per fold. Here the signal is 52 weekly steps and there are 1,082 plots: the same
backbones memorise the training set before epoch 5. Parameter budget is under 50 k, achieved
with depthwise-separable convolutions and global average pooling instead of `flatten`.

Measured parameter counts (n_out=9, n_ctx=28) are in the class docstrings. They are measured
by instantiating the classes, not estimated; ``scripts/11_run_conv.py --count-params``
reprints them.

Two details that are specific to phenology rather than to vision:

- **Kernel width 5 on the 1-D model.** One step is 7 days, so a 3-tap kernel sees 21 days —
  shorter than the green-up it is supposed to detect. Five taps see ~35 days, and after two
  strides the receptive field spans about five months.
- **Circular padding on the DOY axis.** The 52-step axis wraps: step 51 is adjacent to step
  0 in the world. Zero padding invents a trough at the array boundary that a kernel reads as
  a real phenological feature. But circular padding must not be applied to *both* axes of a
  folded 8x8 `reshape` image, where the vertical axis is an artefact of the folding — hence
  ``pad_mode`` is set per substrate by :data:`biodiv.substrates.PAD_MODE`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------

class SepConv1d(nn.Module):
    """Depthwise-separable 1-D conv + BN + GELU. About 8x fewer parameters than a plain conv."""

    def __init__(self, ci: int, co: int, k: int = 5, stride: int = 1, p: float = 0.1,
                 circular: bool = True):
        super().__init__()
        self.dw = nn.Conv1d(ci, ci, k, stride=stride, padding=k // 2, groups=ci, bias=False,
                            padding_mode="circular" if circular else "zeros")
        self.pw = nn.Conv1d(ci, co, 1, bias=False)
        self.bn = nn.BatchNorm1d(co)
        self.act = nn.GELU()
        self.dr = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x):
        return self.dr(self.act(self.bn(self.pw(self.dw(x)))))


class SepConv2d(nn.Module):
    """Depthwise-separable 2-D conv + BN + GELU.

    ``pad_mode`` may be a single string (both axes) or a ``(vertical, horizontal)`` pair, for
    substrates whose two axes mean different things — ``stack5`` is (index, DOY), so the DOY
    axis wraps and the index axis does not.
    """

    def __init__(self, ci: int, co: int, k: int = 3, stride: int = 1, p: float = 0.1,
                 pad_mode: str | tuple[str, str] = "zeros"):
        super().__init__()
        self.k = k
        self.mixed = isinstance(pad_mode, tuple)
        self.pad_mode = pad_mode
        single = "zeros" if self.mixed else pad_mode
        self.dw = nn.Conv2d(ci, ci, k, stride=stride, padding=0 if self.mixed else k // 2,
                            groups=ci, bias=False,
                            padding_mode="zeros" if self.mixed else single)
        self.pw = nn.Conv2d(ci, co, 1, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.GELU()
        self.dr = nn.Dropout2d(p) if p > 0 else nn.Identity()

    def _pad(self, x):
        if not self.mixed:
            return x
        pv, ph = self.pad_mode
        p = self.k // 2
        x = F.pad(x, (p, p, 0, 0), mode="circular" if ph == "circular" else "constant")
        x = F.pad(x, (0, 0, p, p), mode="circular" if pv == "circular" else "constant")
        return x

    def forward(self, x):
        return self.dr(self.act(self.bn(self.pw(self.dw(self._pad(x))))))


# --------------------------------------------------------------------------------------
# topography fusion
# --------------------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise linear modulation of the trunk by the topographic context.

    The ecological argument: an NDVI amplitude of 0.3 does not mean the same thing at 2,500 m
    as at 400 m. Late concatenation can only add topography to the answer; FiLM lets it change
    how the curve is read. 7,264 parameters at widths (32, 64).
    """

    def __init__(self, n_ctx: int = 28, widths: tuple[int, ...] = (32, 64), hidden: int = 32):
        super().__init__()
        self.widths = widths
        self.gen = nn.Sequential(nn.Linear(n_ctx, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * sum(widths)))

    def forward(self, ctx: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        p = self.gen(ctx)
        out, i = [], 0
        for w in self.widths:
            g, b = p[:, i:i + w], p[:, i + w:i + 2 * w]
            i += 2 * w
            out.append((1.0 + g[:, :, None, None], b[:, :, None, None]))
        return out


class TopoPatchBranch(nn.Module):
    """(B, 9, 5, 5) -> (B, emb). 2,048 parameters at emb=32.

    The only fusion option that uses *within-plot* terrain heterogeneity rather than a plot
    mean. That pairs directly with the `pxcube` substrate and with the beta-diversity targets:
    a plot spanning a ravine and a ridge is compositionally distinct for a reason a mean
    elevation cannot express.
    """

    def __init__(self, c_in: int = 9, emb: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.Conv2d(16, emb, 1, bias=False), nn.BatchNorm2d(emb), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.emb = emb

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------

class Pheno1D(nn.Module):
    """1-D CNN over the 52-week curve. The control for "does the second dimension help?".

    Input ``(B, C, 52)`` with C = 1 (one vegetation index) or 5 (indices as channels).

    ========  ==============  ===============  ===============
    option    width           params c_in=1    params c_in=5
    ========  ==============  ===============  ===============
    1D-A      (8, 16, 32)               6,887            7,047
    1D-B      (16, 32, 64)             15,303           15,623
    1D-C      (32, 64, 128)            43,655           44,295
    ========  ==============  ===============  ===============
    """

    WIDTHS = {"A": (8, 16, 32), "B": (16, 32, 64), "C": (32, 64, 128)}

    def __init__(self, c_in: int = 1, width: tuple[int, int, int] = (16, 32, 64),
                 n_out: int = 9, n_ctx: int = 28, circular: bool = True,
                 p_conv: float = 0.1, p_head: float = 0.3):
        super().__init__()
        w1, w2, w3 = width
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, w1, 5, padding=2, bias=False,
                      padding_mode="circular" if circular else "zeros"),
            nn.BatchNorm1d(w1), nn.GELU())
        self.blocks = nn.Sequential(
            SepConv1d(w1, w2, p=p_conv, circular=circular),
            SepConv1d(w2, w2, stride=2, p=p_conv, circular=circular),
            SepConv1d(w2, w3, stride=2, p=p_conv, circular=circular),
            SepConv1d(w3, w3, p=p_conv, circular=circular))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(p_head), nn.Linear(w3 + n_ctx, 64), nn.GELU(),
            nn.Dropout(p_head), nn.Linear(64, n_out))

    def embed(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)

    def forward(self, x, ctx=None):
        z = self.embed(x)
        if ctx is not None:
            z = torch.cat([z, ctx], dim=1)
        return self.head(z)


class PhenoNetS(nn.Module):
    """2-D CNN over a phenological image. Shape-agnostic thanks to ``AdaptiveAvgPool2d(1)``.

    Accepts (1,8,8) reshape, (2,52,52) gaf, (1,5,52) stack5, (1,25,52) pxcube — all with the
    same weights layout, which is what makes the substrate benchmark a fair comparison.

    ========  ==============  =============  ==============  ==============
    option    width           c_in=1 ctx=0   c_in=1 ctx=28   c_in=5 ctx=28
    ========  ==============  =============  ==============  ==============
    2D-A      (8, 16, 32)             4,135           5,031           5,319
    2D-B      (16, 32, 64)           14,151          15,943          16,519
    2D-C      (32, 64, 128)          51,847          55,431          56,583
    2D-X      (128, 256, 512)             -         787,977               -
    ========  ==============  =============  ==============  ==============

    2D-X is the over-parameterised negative control. `docs/03_cnn_architecture.md` proposed
    `efficientnet_b0` for that role; `timm` is not installed, and a same-family model whose
    only difference is width is in any case a cleaner demonstration that the regime, not the
    architecture family, is what breaks.

    ``fusion``: ``none`` (phenology only), ``late`` (concatenate the context vector to the
    pooled embedding), ``film`` (modulate the trunk), ``patch`` (a parallel 5x5 topography
    branch). ``late`` is the default because it is the same fusion point the MLP and the 1-D
    model use, which keeps the three families comparable.
    """

    WIDTHS = {"A": (8, 16, 32), "B": (16, 32, 64), "C": (32, 64, 128), "X": (128, 256, 512)}

    def __init__(self, c_in: int = 1, width: tuple[int, int, int] = (16, 32, 64),
                 n_out: int = 9, n_ctx: int = 28, pad_mode: str | tuple[str, str] = "zeros",
                 fusion: str = "late", topo_patch_channels: int = 9,
                 p_conv: float = 0.1, p_head: float = 0.3):
        super().__init__()
        w1, w2, w3 = width
        self.fusion = fusion
        single = "zeros" if isinstance(pad_mode, tuple) else pad_mode
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, w1, 3, padding=1, bias=False, padding_mode=single),
            nn.BatchNorm2d(w1), nn.GELU())
        self.b1 = SepConv2d(w1, w2, p=p_conv, pad_mode=pad_mode)
        self.b2 = SepConv2d(w2, w2, p=p_conv, pad_mode=pad_mode)
        self.b3 = SepConv2d(w2, w3, stride=2, p=p_conv, pad_mode=pad_mode)
        self.b4 = SepConv2d(w3, w3, p=p_conv, pad_mode=pad_mode)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.film = FiLM(n_ctx, widths=(w2, w3)) if fusion == "film" else None
        self.patch = (TopoPatchBranch(topo_patch_channels)
                      if fusion in ("patch", "patchctx") else None)

        # `patch` REEMPLAZA al vector de contexto; `patchctx` lo AÑADE. La distincion importa
        # porque la ablacion 4c comparaba "parche 2D en vez del resumen" y perdia por 0,04
        # contra `late` -- lo que no dice si el parche aporta algo ENCIMA del resumen, que es
        # otra pregunta. `patchctx` la responde con un solo factor de diferencia contra `late`.
        pemb = self.patch.emb if self.patch else 0
        extra = {"none": 0, "film": 0, "late": n_ctx,
                 "patch": pemb, "patchctx": n_ctx + pemb}[fusion]
        self.head = nn.Sequential(
            nn.Dropout(p_head), nn.Linear(w3 + extra, w3), nn.GELU(),
            nn.Dropout(p_head), nn.Linear(w3, n_out))

    def embed(self, x, ctx=None, patch=None):
        h = self.b2(self.b1(self.stem(x)))
        if self.film is not None and ctx is not None:
            (g2, b2), (g3, b3) = self.film(ctx)
            h = g2 * h + b2
        h = self.b4(self.b3(h))
        if self.film is not None and ctx is not None:
            h = g3 * h + b3
        return self.pool(h).flatten(1)

    def forward(self, x, ctx=None, patch=None):
        z = self.embed(x, ctx=ctx)
        if self.fusion == "late" and ctx is not None:
            z = torch.cat([z, ctx], dim=1)
        elif self.fusion == "patch" and patch is not None:
            z = torch.cat([z, self.patch(patch)], dim=1)
        elif self.fusion == "patchctx" and patch is not None and ctx is not None:
            z = torch.cat([z, ctx, self.patch(patch)], dim=1)
        return self.head(z)


class SEBlock(nn.Module):
    """Squeeze-and-excitation: recalibrate channels by a learned global weighting.

    ~2c^2/r parameters, so a few hundred at these widths. It earns its place here because
    the channels are not interchangeable: for `stack5` they are five vegetation indices,
    for `gaf` the summation and difference fields. A kernel treats them symmetrically;
    this lets the network learn that one of them matters more for a given plot.
    """

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(4, c // r)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]


class ResSepConv2d(nn.Module):
    """`SepConv2d` with an identity shortcut, projected when the shape changes.

    The point is depth: without a skip, stacking more separable blocks on 1,082 samples
    degrades rather than helps. With one, the block starts near the identity and only has
    to learn a correction.
    """

    def __init__(self, ci, co, stride=1, p=0.1, pad_mode="zeros", se=False):
        super().__init__()
        self.conv = SepConv2d(ci, co, stride=stride, p=p, pad_mode=pad_mode)
        self.se = SEBlock(co) if se else None
        self.proj = (nn.Conv2d(ci, co, 1, stride=stride, bias=False)
                     if (ci != co or stride != 1) else nn.Identity())

    def forward(self, x):
        h = self.conv(x)
        if self.se is not None:
            h = self.se(h)
        return h + self.proj(x)


class MultiScaleBlock(nn.Module):
    """Kernels 3, 5 and 7 in parallel, concatenated.

    On a 52-step year folded into an 8x8 image, a 3x3 kernel spans about three weeks along
    a row and two months down a column. Different phenological events live at different
    scales -- a green-up edge is sharp, a dry-season plateau is broad -- and a single kernel
    size has to commit to one. Depthwise convolutions keep the three branches affordable.
    """

    def __init__(self, ci, co, p=0.1, pad_mode="zeros"):
        super().__init__()
        single = "zeros" if isinstance(pad_mode, tuple) else pad_mode
        per = max(4, co // 3)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ci, ci, k, padding=k // 2, groups=ci, bias=False,
                          padding_mode=single),
                nn.Conv2d(ci, per, 1, bias=False), nn.BatchNorm2d(per), nn.GELU())
            for k in (3, 5, 7)])
        self.mix = nn.Sequential(nn.Conv2d(per * 3, co, 1, bias=False),
                                 nn.BatchNorm2d(co), nn.GELU(), nn.Dropout2d(p))

    def forward(self, x):
        return self.mix(torch.cat([b(x) for b in self.branches], dim=1))


class PhenoNetV(PhenoNetS):
    """`PhenoNetS` with the trunk swapped for a different block type.

    Subclassing keeps the stem, the fusion machinery, the head and `forward` identical, so a
    comparison between backbones changes one thing. `arch` selects the block:

    ``res``    residual separable blocks
    ``se``     residual separable blocks with squeeze-excitation
    ``multi``  multi-scale blocks (kernels 3, 5, 7)

    All stay inside the project's <60k parameter budget at width B, which is not a style
    preference: with 1,082 plots, width X (788k) has already been measured to lose.
    """

    def __init__(self, *args, arch: str = "res", **kw):
        super().__init__(*args, **kw)
        w1, w2, w3 = kw.get("width", PhenoNetS.WIDTHS["B"])
        p = kw.get("p_conv", 0.1)
        pad = kw.get("pad_mode", "zeros")
        if arch in ("res", "se"):
            se = arch == "se"
            self.b1 = ResSepConv2d(w1, w2, p=p, pad_mode=pad, se=se)
            self.b2 = ResSepConv2d(w2, w2, p=p, pad_mode=pad, se=se)
            self.b3 = ResSepConv2d(w2, w3, stride=2, p=p, pad_mode=pad, se=se)
            self.b4 = ResSepConv2d(w3, w3, p=p, pad_mode=pad, se=se)
        elif arch == "multi":
            self.b1 = MultiScaleBlock(w1, w2, p=p, pad_mode=pad)
            self.b2 = MultiScaleBlock(w2, w2, p=p, pad_mode=pad)
            self.b3 = nn.Sequential(MultiScaleBlock(w2, w3, p=p, pad_mode=pad),
                                    nn.AvgPool2d(2, ceil_mode=True))
            self.b4 = MultiScaleBlock(w3, w3, p=p, pad_mode=pad)
        else:
            raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(family: str, *, c_in: int, n_out: int, n_ctx: int, width: str = "B",
                pad_mode="zeros", fusion: str = "late", circular: bool = True,
                d_in: int | None = None, arch: str = "sep", **kwargs) -> nn.Module:
    """Single entry point so the driver scripts never import three different constructors."""
    from .models_tabular import MLPMulti
    if family == "MLP":
        if d_in is None:
            raise ValueError("MLP needs d_in")
        return MLPMulti(d_in, hidden=MLPMulti.WIDTHS[width], n_out=n_out)
    if family == "C1D":
        return Pheno1D(c_in=c_in, width=Pheno1D.WIDTHS[width], n_out=n_out, n_ctx=n_ctx,
                       circular=circular)
    if family == "C2D":
        kw = dict(c_in=c_in, width=PhenoNetS.WIDTHS[width], n_out=n_out, n_ctx=n_ctx,
                  pad_mode=pad_mode, fusion=fusion,
                  p_conv=kwargs.get("p_conv", 0.1), p_head=kwargs.get("p_head", 0.3))
        return PhenoNetS(**kw) if arch == "sep" else PhenoNetV(arch=arch, **kw)
    raise ValueError(f"unknown family {family!r}")
