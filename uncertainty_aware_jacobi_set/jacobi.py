"""Uncertainty-aware Jacobi set computation (Sec. 4.2 and Algorithm 1).

The pair of scalar fields is a joint Gaussian X = (f; g) ~ N(mu, L L^T) over
the vertices of a triangulation (see models.FieldPair and geometry):

    1. The PL gradients are linear in the vertex values, so the per-triangle
       gradient vector Y_T = (df/dx, df/dy, dg/dx, dg/dy)|_T is again
       Gaussian; its covariance factor consists of the rows G_T L_T (Eq. 6).
    2. The gradient alignment kappa_T = Y_T^T Q Y_T is a quadratic form, so
       its mean and variance and the covariance of the alignments of the two
       triangles adjacent to an edge follow in closed form (Eqs. 7-9).
    3. The critical edge test (Eq. 1) is lifted to the Jacobi set edge
       probability p_e = P(kappa_T1 kappa_T2 < 0) by matching
       (kappa_T1, kappa_T2) with a bivariate Gaussian (Eq. 10).
    4. The p_e are aggregated per vertex into the expected Jacobi set degree
       and its signed variant (Eqs. 11 and 12).

Only rows of L are contracted, chunk by chunk, so the full covariance matrix
is never formed.
"""

import numpy as np
from scipy.stats import norm

#: kappa_T = Y_T^T Q Y_T with Y_T = (df/dx, df/dy, dg/dx, dg/dy)|_T
Q = 0.5 * np.array([[0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0]])

#: triangles or edges per chunk when contracting rows of the factor L
CHUNK = 1024


# ---------------------------------------------------------------------------
# deterministic PL Jacobi set (Sec. 3.2)
# ---------------------------------------------------------------------------

def pl_gradients(geo, values):
    """PL gradient G_T v_T of a vertex field on every triangle.

    values is one field (N,) or a batch of fields (n, N); returns the x and y
    components, each of shape (T,) or (n, T).  The sums are written out so
    that the mean fields and all sampled realizations go through identical
    floating-point operations; triangles with kappa ~ 0 then get the same
    sign in the mean Jacobi set, in p_e, and in the samples."""
    G, v = geo["G"], np.asarray(values, float)[..., geo["tris"]]
    gx = G[:, 0, 0] * v[..., 0] + G[:, 0, 1] * v[..., 1] + G[:, 0, 2] * v[..., 2]
    gy = G[:, 1, 0] * v[..., 0] + G[:, 1, 1] * v[..., 1] + G[:, 1, 2] * v[..., 2]
    return gx, gy


def pl_kappa(geo, f, g):
    """Gradient alignment kappa_T = det(grad f|_T, grad g|_T) per triangle,
    for single fields (N,) or batches (n, N)."""
    fx, fy = pl_gradients(geo, f)
    gx, gy = pl_gradients(geo, g)
    return fx * gy - fy * gx


def critical_edges(geo, kappa):
    """Critical edge test (Eq. 1): an interior edge is critical iff kappa
    changes sign across its two adjacent triangles.  The test is binary
    ({kappa >= 0} vs. {kappa < 0}), i.e., exact zeros count as positive; with
    this tie-break the Even Degree Lemma also holds on exactly flat regions.
    kappa may be (T,) or batched (n, T)."""
    s = np.asarray(kappa) >= 0.0
    return s[..., geo["T1"]] != s[..., geo["T2"]]


def jacobi_set(geo, f, g):
    """PL Jacobi set of the fields f, g as a boolean mask over the interior
    edges -- (E,) for single fields, (n, E) for batches."""
    return critical_edges(geo, pl_kappa(geo, f, g))


# ---------------------------------------------------------------------------
# moments of the Gaussian gradients and alignments
# ---------------------------------------------------------------------------

def _chunks(n):
    return (slice(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK))


def _gradient_factor(geo, pair, tri):
    """Covariance factor rows of Y_T for the triangles tri -- (len(tri), 4, r),
    so that Cov(Y_T, Y_T') is the product of the rows of T and T'."""
    G, t = geo["G"][tri], geo["tris"][tri]
    return np.concatenate([G @ pair.Lf[t], G @ pair.Lg[t]], axis=1)


def triangle_moments(geo, pair):
    """Mean m_T (T, 4) and covariance S_T (T, 4, 4) of the per-triangle
    gradient vector Y_T (Eq. 6)."""
    fx, fy = pl_gradients(geo, pair.mean_f)
    gx, gy = pl_gradients(geo, pair.mean_g)
    m = np.stack([fx, fy, gx, gy], axis=1)
    S = np.empty((len(m), 4, 4))
    for sl in _chunks(len(m)):
        Y = _gradient_factor(geo, pair, sl)
        S[sl] = Y @ Y.transpose(0, 2, 1)
    return m, S


def kappa_moments(m, S):
    """Mean and variance of the alignment kappa_T = Y_T^T Q Y_T (Eqs. 3, 4):

        E(kappa_T)   = tr(Q S_T) + m_T^T Q m_T,
        Var(kappa_T) = 2 tr(Q S_T Q S_T) + 4 m_T^T Q S_T Q m_T.

    The mean is written out in the same order as pl_kappa, so without
    uncertainty it equals the alignment of the mean fields exactly."""
    mean = (S[:, 0, 3] - S[:, 1, 2]) + (m[:, 0] * m[:, 3] - m[:, 1] * m[:, 2])
    QS = Q @ S
    var = 2.0 * np.einsum("tij,tji->t", QS, QS) \
        + 4.0 * np.einsum("ti,tij,jk,tk->t", m, QS, Q, m)
    return mean, np.clip(var, 0.0, None)


def edge_alignment_cov(geo, pair, m):
    """Covariance of the alignments of the two triangles adjacent to each
    interior edge (Eq. 9).  With the block matrices Q_1, Q_2 and the
    cross-triangle block C = Cov(Y_T1, Y_T2) of Sigma_e, Eq. 9 reduces to

        Cov(kappa_T1, kappa_T2) = 2 tr(Q C Q C^T) + 4 m_T1^T Q C Q m_T2.

    The shared edge vertices and the spatial correlation both enter via C."""
    T1, T2 = geo["T1"], geo["T2"]
    cov = np.empty(len(T1))
    for sl in _chunks(len(T1)):
        C = _gradient_factor(geo, pair, T1[sl]) \
            @ _gradient_factor(geo, pair, T2[sl]).transpose(0, 2, 1)
        QC, QCt = Q @ C, Q @ C.transpose(0, 2, 1)
        cov[sl] = 2.0 * np.einsum("eij,eji->e", QC, QCt) \
            + 4.0 * np.einsum("ei,eij,ej->e", m[T1[sl]] @ Q, C, m[T2[sl]] @ Q)
    return cov


# ---------------------------------------------------------------------------
# edge probability (Eq. 10)
# ---------------------------------------------------------------------------

def Phi2(a, b, rho, n_nodes=48):
    """Bivariate standard normal CDF Phi_2(a, b; rho), vectorized.

    Uses the single-integral form of Drezner and Wesolowsky,
        Phi_2 = Phi(a) Phi(b) + (1/2pi) int_0^{asin(rho)}
                exp(-(a^2 + b^2 - 2ab sin t) / (2 cos^2 t)) dt,
    evaluated with fixed Gauss-Legendre nodes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = np.clip(np.asarray(rho, float), -0.99999, 0.99999)
    u, w = np.polynomial.legendre.leggauss(n_nodes)
    th_max = np.arcsin(rho)
    theta = 0.5 * th_max[:, None] * (u[None, :] + 1.0)
    expo = np.exp(-(a[:, None]**2 + b[:, None]**2
                    - 2.0 * a[:, None] * b[:, None] * np.sin(theta))
                  / (2.0 * np.cos(theta)**2))
    integral = 0.5 * th_max * np.sum(w[None, :] * expo, axis=1)
    return np.clip(norm.cdf(a) * norm.cdf(b) + integral / (2.0 * np.pi),
                   0.0, 1.0)


def edge_probability(mu1, mu2, var1, var2, cov):
    """Jacobi set edge probability p_e = P(kappa_T1 kappa_T2 < 0) under the
    moment-matched bivariate Gaussian (Eq. 10),

        p_e = Phi(-mu1/s1) + Phi(-mu2/s2) - 2 Phi_2(-mu1/s1, -mu2/s2; rho),

    with rho = cov / (s1 s2)."""
    s1 = np.sqrt(np.clip(var1, 1e-300, None))
    s2 = np.sqrt(np.clip(var2, 1e-300, None))
    a1, a2 = -mu1 / s1, -mu2 / s2
    p = norm.cdf(a1) + norm.cdf(a2) - 2.0 * Phi2(a1, a2, cov / (s1 * s2))

    # edges with (numerically) zero variance fall back to the deterministic
    # sign test, with the same binary convention as critical_edges
    det = (var1 < 1e-30 * np.maximum(mu1**2, 1.0)) \
        & (var2 < 1e-30 * np.maximum(mu2**2, 1.0))
    p = np.where(det, ((mu1 >= 0.0) != (mu2 >= 0.0)).astype(float), p)
    return np.clip(p, 0.0, 1.0)


# ---------------------------------------------------------------------------
# vertex aggregation (Eqs. 11 and 12)
# ---------------------------------------------------------------------------

def expected_degree(geo, p_edge):
    """Expected Jacobi set degree E(D_i) = sum_{e ni i} p_e (Eq. 11), exact
    by linearity of the expectation."""
    deg = np.zeros(len(geo["P"]))
    np.add.at(deg, geo["EA"], p_edge)
    np.add.at(deg, geo["EB"], p_edge)
    return deg


def vertex_alignment_sign(geo, mean_kappa):
    """Alignment sign s_i = sgn(sum_{T ni i} E(kappa_T)) in {-1, +1} of the
    signed expected Jacobi set degree (Eq. 12)."""
    acc = np.zeros(len(geo["P"]))
    for c in range(3):
        np.add.at(acc, geo["tris"][:, c], mean_kappa)
    return np.where(acc >= 0.0, 1.0, -1.0)


# ---------------------------------------------------------------------------
# the full pipeline
# ---------------------------------------------------------------------------

def compute(geo, pair):
    """Uncertainty-aware Jacobi set (Algorithm 1).

    Parameters
    ----------
    geo  : geometry dictionary from geometry.build_geometry
    pair : models.FieldPair

    Returns
    -------
    dict with
        p_edge          : (E,) Jacobi set edge probability p_e (Eq. 10)
        expected_degree : (N,) expected Jacobi set degree E(D_i) (Eq. 11)
        signed_degree   : (N,) signed expected Jacobi set degree (Eq. 12)
        mean_crit       : (E,) bool, the Jacobi set of the mean fields
    """
    T1, T2 = geo["T1"], geo["T2"]
    m, S = triangle_moments(geo, pair)
    mean, var = kappa_moments(m, S)
    cov = edge_alignment_cov(geo, pair, m)
    p_edge = edge_probability(mean[T1], mean[T2], var[T1], var[T2], cov)
    deg = expected_degree(geo, p_edge)
    return {"p_edge": p_edge,
            "expected_degree": deg,
            "signed_degree": vertex_alignment_sign(geo, mean) * deg,
            "mean_crit": jacobi_set(geo, pair.mean_f, pair.mean_g)}
