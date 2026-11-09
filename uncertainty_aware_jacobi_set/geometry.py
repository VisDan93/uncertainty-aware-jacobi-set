"""Triangulated grid for the piecewise linear Jacobi set computation.

build_geometry returns a dictionary with the vertex positions, the
triangulation, the per-triangle gradient operators, and the interior edges
with their two adjacent triangles.  It is computed once per grid and reused
across uncertainty models.  The helper functions work on any triangulation.
"""

from collections import defaultdict

import numpy as np


def structured_triangulation(nx, ny):
    """Triangulation of an (ny x nx) row-major grid, every cell split along
    the same diagonal.  Returns a (T, 3) array of vertex indices with all
    triangles oriented counterclockwise."""
    c, r = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    v00 = (r * nx + c).ravel()
    v10, v01, v11 = v00 + 1, v00 + nx, v00 + nx + 1
    return np.concatenate([np.column_stack([v00, v10, v11]),
                           np.column_stack([v00, v11, v01])])


def interior_edges(tris):
    """Interior edges of a triangulation, i.e., edges shared by exactly two
    triangles.  The critical edge test is defined on these.

    Returns
    -------
    EA, EB : (E,) endpoint vertex indices of each edge
    T1, T2 : (E,) indices of the two adjacent triangles
    """
    edge_to_tris = defaultdict(list)
    for ti, tr in enumerate(tris):
        for r in range(3):
            a, b = sorted((int(tr[r]), int(tr[(r + 1) % 3])))
            edge_to_tris[(a, b)].append(ti)
    edges = [(e, ts) for e, ts in edge_to_tris.items() if len(ts) == 2]
    EA = np.array([e[0] for e, _ in edges])
    EB = np.array([e[1] for e, _ in edges])
    T1 = np.array([ts[0] for _, ts in edges])
    T2 = np.array([ts[1] for _, ts in edges])
    return EA, EB, T1, T2


def gradient_operators(P, tris):
    """Per-triangle PL gradient operators G_T with grad f|_T = G_T f_T,

        G_T = E_T^{-1} [[-1, 1, 0], [-1, 0, 1]],
        E_T = [(p_b - p_a)^T; (p_v - p_a)^T],

    for T = <a, b, v>.  Returns a (T, 2, 3) array."""
    p0, p1, p2 = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    E = np.stack([p1 - p0, p2 - p0], axis=1)
    return np.einsum('tij,jk->tik', np.linalg.inv(E),
                     np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]))


def build_geometry(x1d, y1d):
    """Geometry of the structured grid with coordinates x1d (nx,) and
    y1d (ny,), every cell split along the same diagonal (Sec. 3.2).

    Returns a dict with
        P        : (N, 2) vertex positions, row-major (x varies fastest)
        tris     : (T, 3) triangles, counterclockwise
        G        : (T, 2, 3) PL gradient operators
        EA, EB   : (E,) endpoints of the interior edges
        T1, T2   : (E,) the two triangles adjacent to each interior edge
        seg      : (E, 2, 2) interior edges as line segments
        x1d, y1d : the grid coordinates
    """
    x1d, y1d = np.asarray(x1d, float), np.asarray(y1d, float)
    X, Y = np.meshgrid(x1d, y1d)
    P = np.column_stack([X.ravel(), Y.ravel()])
    tris = structured_triangulation(len(x1d), len(y1d))
    EA, EB, T1, T2 = interior_edges(tris)
    return dict(P=P, tris=tris, G=gradient_operators(P, tris),
                EA=EA, EB=EB, T1=T1, T2=T2,
                seg=np.stack([P[EA], P[EB]], axis=1), x1d=x1d, y1d=y1d)
