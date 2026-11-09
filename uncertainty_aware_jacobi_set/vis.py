"""Visualization of uncertainty-aware Jacobi sets (Sec. 4.3).

The Jacobi set edge probability p_e is drawn as colored interior edges
(edge_probability), the vertex aggregates as binned heatmaps
(expected_degree, signed_expected_degree), and deterministic Jacobi sets --
of the mean fields or of sampled realizations -- as lines (jacobi_set).
All functions take the geometry first and an optional ax and return the
created artist, so they compose into layered views and subplot layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import bilinear, jacobi

#: line color of the mean Jacobi set (orange)
MEAN_COLOR = "#ff7f0e"
#: line colors of Jacobi set samples (green, sand, rose, ...)
SAMPLE_COLORS = ["#117733", "#DDCC77", "#CC6677", "#882255", "#44AA99",
                 "#999933", "#AA4499"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _axes(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    return ax


def _image(ax, geo, values, **kwargs):
    """Per-vertex values as an image with pixel centers on the vertices."""
    x, y = geo["x1d"], geo["y1d"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    return ax.imshow(np.reshape(values, (len(y), len(x))), origin="lower",
                     extent=[x[0] - dx / 2, x[-1] + dx / 2,
                             y[0] - dy / 2, y[-1] + dy / 2], **kwargs)


def _colorbar(ax, artist, label):
    # tie the colorbar to the drawn axes size, so it keeps a constant width
    # after set_aspect("equal") reshapes the axes
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    return ax.figure.colorbar(artist, cax=cax, label=label)


def _finish(ax, geo, title=None, export=False):
    P = geo["P"]
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if export:
        ax.set_xticks([])
        ax.set_yticks([])


# ---------------------------------------------------------------------------
# fields and vertex aggregates
# ---------------------------------------------------------------------------

def scalar_field(geo, field, ax=None, cmap="viridis", vmin=None, vmax=None,
                 cbar=False, cbar_label=None, title=None, export=False):
    """Per-vertex scalar field as an image."""
    ax = _axes(ax)
    im = _image(ax, geo, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar:
        _colorbar(ax, im, cbar_label)
    _finish(ax, geo, title, export)
    return im


def expected_degree(geo, exp_deg, ax=None, vmax=2.0, n_bins=10, cbar=True,
                    title=None, export=False):
    """Expected Jacobi set degree E(D_i) as a binned sequential heatmap over
    [0, vmax]; larger values fall into the last bin.  Values close to 2 mark
    vertices that a Jacobi set line passes in almost every realization."""
    ax = _axes(ax)
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.0, 1.0, n_bins)))
    norm = BoundaryNorm(np.linspace(0.0, vmax, n_bins + 1), n_bins)
    im = _image(ax, geo, exp_deg, cmap=cmap, norm=norm)
    if cbar:
        _colorbar(ax, im, r"$\mathrm{E}(D_i)$")
    _finish(ax, geo, title, export)
    return im


def signed_expected_degree(geo, signed_deg, ax=None, vmax=2.0,
                           n_bins_side=10, cbar=True, title=None,
                           export=False):
    """Signed expected Jacobi set degree s_i E(D_i) with a binned diverging
    colormap centered at zero: saturation encodes E(D_i) and hue the sign of
    the expected alignment (red E(kappa) >= 0, blue E(kappa) < 0).  For
    Sigma_fg = 0, the hue changes along the mean Jacobi set.

    The colormap has an odd number of bins (2 n_bins_side + 1): a pure
    white central bin around zero, flanked by n_bins_side blue and red
    steps; values beyond +-vmax fall into the outermost bins."""
    ax = _axes(ax)
    neg = plt.get_cmap("Blues")(np.linspace(1.0, 0.0, n_bins_side + 1))[:-1]
    pos = plt.get_cmap("Reds")(np.linspace(0.0, 1.0, n_bins_side + 1))[1:]
    cmap = ListedColormap(np.vstack([neg, [[1.0, 1.0, 1.0, 1.0]], pos]))
    norm = BoundaryNorm(np.linspace(-vmax, vmax, 2 * n_bins_side + 2),
                        ncolors=2 * n_bins_side + 1, clip=True)
    im = _image(ax, geo, signed_deg, cmap=cmap, norm=norm)
    if cbar:
        cb = _colorbar(ax, im, r"$\tilde{D}_i = s_i\,\mathrm{E}(D_i)$")
        cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    _finish(ax, geo, title, export)
    return im


# ---------------------------------------------------------------------------
# edge-based line renderings
# ---------------------------------------------------------------------------

def edge_probability(geo, p_edge, ax=None, vmax=1.0, lw=1.0, cbar=True,
                     title=None, export=False):
    """Jacobi set edge probability p_e as line color of the interior edges;
    edges with p_e = 0 are not drawn."""
    ax = _axes(ax)
    p_edge = np.asarray(p_edge).ravel()
    keep = p_edge > 0.0
    lc = LineCollection(geo["seg"][keep], array=p_edge[keep], cmap="Blues",
                        linewidths=lw, norm=Normalize(vmin=0.0, vmax=vmax))
    ax.add_collection(lc)
    if cbar:
        _colorbar(ax, lc, r"$p_e$")
    _finish(ax, geo, title, export)
    return lc


def jacobi_set(geo, f, g, ax=None, color=MEAN_COLOR, lw=1.0, label=None,
               method="pl"):
    """Deterministic Jacobi set of the fields f, g (N,) as lines: pass the
    mean fields for the mean Jacobi set or a sampled realization for a
    Jacobi set sample.

    method "pl" (default) draws the critical edges themselves, "rclb" the
    reduced-connectivity local bilinear geometry of the same critical edges
    (see the bilinear module)."""
    ax = _axes(ax)
    if method == "pl":
        seg = geo["seg"][jacobi.jacobi_set(geo, f, g)]
    elif method == "rclb":
        seg = bilinear.rclb_segments(geo, f, g)
    else:
        raise ValueError("method must be 'pl' or 'rclb'")
    lc = LineCollection(seg, colors=color, linewidths=lw, label=label)
    ax.add_collection(lc)
    _finish(ax, geo)
    return lc
