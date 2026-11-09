"""Analytic test dataset of Kloetzl et al. (2022): two scalar fields on
[-1, 1]^2, each a sum of three Gaussian bumps a exp(-||p - c||^2 / w).  The
gradients are known in closed form, and so are the gradient alignment kappa
and the exact Jacobi set {kappa = 0}."""

import numpy as np

#: Gaussian bumps (amplitude a, center c, width w) of the two fields
F_BUMPS = [(15.0, (0.25, -0.25), 0.5), (5.0, (-0.25, -0.2), 0.03),
           (15.0, (-0.8, 0.5), 0.1)]
G_BUMPS = [(1.0, (0.25, -0.25), 0.5), (1.0, (-0.8, 0.5), 0.1),
           (1.0, (0.0, 0.0), 0.1)]


def bumps(x, y, params):
    """Sum of Gaussian bumps and its x and y derivatives at (x, y)."""
    val = dx = dy = 0.0
    for a, (cx, cy), w in params:
        e = a * np.exp(-((x - cx)**2 + (y - cy)**2) / w)
        val = val + e
        dx = dx - 2.0 * (x - cx) / w * e
        dy = dy - 2.0 * (y - cy) / w * e
    return val, dx, dy


def analytic_grid(res=50):
    """Sampling of the analytic pair on a (res x res) grid over [-1, 1]^2.
    Returns a dict with the coordinates x1d, y1d and the fields f, g and the
    closed-form alignment kappa_exact = det(grad f, grad g), each (res, res)."""
    lin = np.linspace(-1.0, 1.0, res)
    x, y = np.meshgrid(lin, lin)
    f, fx, fy = bumps(x, y, F_BUMPS)
    g, gx, gy = bumps(x, y, G_BUMPS)
    return {"x1d": lin, "y1d": lin, "f": f, "g": g,
            "kappa_exact": fx * gy - fy * gx}
