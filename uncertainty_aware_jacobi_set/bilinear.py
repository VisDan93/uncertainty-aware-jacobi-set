"""Reduced-connectivity local bilinear (RCLB) Jacobi set lines.

Optional line geometry for deterministic Jacobi sets -- the mean Jacobi set
and Jacobi set samples -- following the local bilinear method of Kloetzl et
al. (2022) with the reduced connectivity of Kloetzl et al. (2022a).  The
paper uses the PL lines throughout; this module is an alternative for
smoother lines.  The critical edges are the same as in the PL method
(jacobi.critical_edges); only the drawn geometry changes:

    * On each critical edge, a Jacobi set point is placed on the polyline
      v1 -> m -> v2 (m the edge midpoint, v1/v2 the opposite vertices of the
      two adjacent triangles) at the zero crossing of the bilinear alignment
      values kappa^bi at v1 and v2 -- with a linear interpolation of the
      per-triangle kappa as fallback where the bilinear values do not change
      sign.
    * The Jacobi set points are connected with reduced connectivity: a
      vertex incident to exactly two critical edges connects their points
      directly; higher (even) degrees connect each point to the barycenter
      of the incident points.  This is homotopy-equivalent to the PL set and
      keeps the Even Degree Lemma intact.

The probabilistic quantities (p_e, expected degrees) remain defined on the
PL edges; this module only affects how deterministic Jacobi sets are drawn.
"""

import numpy as np

from . import jacobi


def _opposite_vertices(tris, T, EA, EB):
    """For each interior edge, the vertex of triangle T[e] that is not an
    edge endpoint."""
    tv = tris[T]                                          # (E, 3)
    mask = (tv != EA[:, None]) & (tv != EB[:, None])
    return np.take_along_axis(tv, np.argmax(mask, axis=1)[:, None], 1)[:, 0]


def bilinear_kappas(P4, F4, G4):
    """Bilinear gradient alignment values kappa^bi at the two opposite
    vertices v1, v2 of the quadrilateral cell (a, b, v1, v2) of an interior
    edge (Kloetzl et al. 2022).

    The bilinear interpolant c0 + c1 x + c2 y + c3 xy of each field is
    fitted through the four vertex values; its gradient is
    (c1 + c3 y, c2 + c3 x), and kappa^bi is the gradient alignment of the
    two interpolants, evaluated at v1 and v2.

    Parameters
    ----------
    P4     : (n, 4, 2) vertex positions in the order (a, b, v1, v2)
    F4, G4 : (n, 4) field values in the same order

    Returns
    -------
    kt1, kt2 : (n,) kappa^bi at v1 and v2
    ok       : (n,) bool, False where the cell is degenerate for the
               bilinear basis (no refinement possible)
    """
    n = len(P4)
    A = np.concatenate([np.ones((n, 4, 1)), P4,
                        (P4[:, :, :1] * P4[:, :, 1:])], axis=2)   # [1, x, y, xy]
    ok = np.abs(np.linalg.det(A)) > 1e-12
    kt1, kt2 = np.zeros(n), np.zeros(n)
    if ok.any():
        cf = np.linalg.solve(A[ok], F4[ok][:, :, None])[:, :, 0]  # (m, 4)
        cg = np.linalg.solve(A[ok], G4[ok][:, :, None])[:, :, 0]

        def kappa_at(p):                                          # (m, 2)
            gfx, gfy = cf[:, 1] + cf[:, 3] * p[:, 1], cf[:, 2] + cf[:, 3] * p[:, 0]
            ggx, ggy = cg[:, 1] + cg[:, 3] * p[:, 1], cg[:, 2] + cg[:, 3] * p[:, 0]
            return gfx * ggy - gfy * ggx

        kt1[ok] = kappa_at(P4[ok][:, 2])
        kt2[ok] = kappa_at(P4[ok][:, 3])
    return kt1, kt2, ok


def _collapse(js_points, EA, EB, crit):
    """Reduced connectivity (Kloetzl et al. 2022a): connect the Jacobi set
    points of each vertex's incident critical edges -- directly for degree
    two, via their barycenter for higher (even) degrees.

    Returns (n_segments, 2, 2) line segments."""
    incident = {}
    for e in np.flatnonzero(crit):
        incident.setdefault(EA[e], []).append(e)
        incident.setdefault(EB[e], []).append(e)

    segments = []
    for edges in incident.values():
        if len(edges) == 2:
            segments.append((js_points[edges[0]], js_points[edges[1]]))
        elif len(edges) > 2:
            bary = js_points[edges].mean(axis=0)
            for e in edges:
                segments.append((bary, js_points[e]))
    return np.asarray(segments).reshape(-1, 2, 2)


def rclb_segments(geo, f, g):
    """RCLB line segments (n, 2, 2) of the Jacobi set of the fields f, g (N,),
    e.g., the mean fields or one sampled realization.  The critical edges are
    those of jacobi.jacobi_set(geo, f, g)."""
    P, tris = geo["P"], geo["tris"]
    EA, EB, T1, T2 = geo["EA"], geo["EB"], geo["T1"], geo["T2"]
    f, g = np.asarray(f, float).ravel(), np.asarray(g, float).ravel()

    kappa = jacobi.pl_kappa(geo, f, g)
    crit = jacobi.critical_edges(geo, kappa)
    ci = np.flatnonzero(crit)
    v1 = _opposite_vertices(tris, T1, EA, EB)[ci]
    v2 = _opposite_vertices(tris, T2, EA, EB)[ci]

    # zero crossing along v1 -> v2: linear interpolation of the per-triangle
    # kappa, refined by the bilinear values where these change sign
    k1, k2 = kappa[T1[ci]], kappa[T2[ci]]
    lam = -k1 / (k2 - k1)
    quad = np.stack([EA[ci], EB[ci], v1, v2], axis=1)            # (m, 4)
    bt1, bt2, ok = bilinear_kappas(P[quad], f[quad], g[quad])
    refine = ok & ((bt1 >= 0.0) != (bt2 >= 0.0))
    lam[refine] = -bt1[refine] / (bt2[refine] - bt1[refine])

    # place the point on the polyline v1 -> m -> v2 (m = edge midpoint)
    m = 0.5 * (P[EA[ci]] + P[EB[ci]])
    lam2 = 2.0 * lam
    lo = lam2 <= 1.0
    pts = np.empty((len(ci), 2))
    pts[lo] = P[v1][lo] + lam2[lo, None] * (m[lo] - P[v1][lo])
    pts[~lo] = m[~lo] + (lam2[~lo, None] - 1.0) * (P[v2][~lo] - m[~lo])

    points = np.full((len(EA), 2), np.nan)
    points[ci] = pts
    return _collapse(points, EA, EB, crit)
