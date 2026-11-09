"""Gaussian uncertainty model for a pair of scalar fields (Sec. 4.1).

The vertex values f, g in R^N of the two fields are modeled jointly as

    X = (f; g) ~ N(mu, Sigma),   mu = (mu_f; mu_g),   Sigma = L L^T,

with a factor L in R^{2N x r}.  The Jacobi set computation only contracts
rows of L, so the full 2N x 2N covariance is never formed.  L is estimated
from ensemble members (FieldPair.from_ensemble) or built from a kernel model
per field (kernel_factor and FieldPair.independent, where Sigma_fg = 0).
"""

import numpy as np
from scipy.spatial.distance import cdist


def kernel_factor(P, sigma, ell):
    """Covariance factor L (N, r) of the squared exponential kernel model of
    one field,

        (Sigma)_ij = sigma_i sigma_j exp(-||p_i - p_j||^2 / (2 ell^2)),

    with vertex positions P (N, 2), a per-vertex (or constant) standard
    deviation sigma >= 0, and the length scale ell; ell = 0 gives the
    variance-only model Sigma = diag(sigma^2).

    The dense N x N matrix is formed and eigendecomposed (eigenpairs below
    1e-12 of the largest are dropped), which needs O(N^2) memory and O(N^3)
    time and limits kernel models to grids of about 10^4 vertices.
    Ensemble models do not have this limitation."""
    P = np.asarray(P, float)
    sigma = np.broadcast_to(np.asarray(sigma, float).ravel(), len(P))
    K = np.eye(len(P)) if ell == 0 \
        else np.exp(-cdist(P, P, "sqeuclidean") / (2.0 * ell**2))
    w, V = np.linalg.eigh(np.outer(sigma, sigma) * K)
    keep = w > 1e-12 * w[-1]
    return V[:, keep] * np.sqrt(w[keep])


class FieldPair:
    """Joint Gaussian model of the fields f, g.

    Holds the mean fields mean_f, mean_g (N,) and the factor L (2N, r) of the
    joint covariance.  Its rows Lf = L[:N] and Lg = L[N:] belong to f and g,
    so that Sigma_ff = Lf Lf^T, Sigma_gg = Lg Lg^T, and Sigma_fg = Lf Lg^T.
    """

    def __init__(self, mean_f, mean_g, L):
        self.mean_f = np.asarray(mean_f, float).ravel()
        self.mean_g = np.asarray(mean_g, float).ravel()
        self.L = np.asarray(L, float)
        n = len(self.mean_f)
        self.Lf, self.Lg = self.L[:n], self.L[n:]

    @classmethod
    def independent(cls, mean_f, Lf, mean_g, Lg):
        """Separately modeled fields, e.g., with kernel_factor: the factors
        Lf (N, r_f) and Lg (N, r_g) are combined block-diagonally, so that
        Sigma_fg = 0 exactly."""
        Lf, Lg = np.asarray(Lf, float), np.asarray(Lg, float)
        L = np.block([[Lf, np.zeros((len(Lf), Lg.shape[1]))],
                      [np.zeros((len(Lg), Lf.shape[1])), Lg]])
        return cls(mean_f, mean_g, L)

    @classmethod
    def from_ensemble(cls, members_f, members_g):
        """Model estimated from E ensemble members of both fields (each of
        shape (E, N) or (E, ny, nx), with a common member axis).  The mean is
        the sample mean, and L = [x^(1) - mu, ..., x^(E) - mu] / sqrt(E - 1)
        is the centered and scaled member matrix, so that L L^T is the
        unbiased sample covariance including the empirical cross-field
        covariance."""
        X = np.concatenate([np.reshape(members_f, (len(members_f), -1)),
                            np.reshape(members_g, (len(members_g), -1))],
                           axis=1).astype(float)
        mu = X.mean(axis=0)
        n = X.shape[1] // 2
        return cls(mu[:n], mu[n:], (X - mu).T / np.sqrt(len(X) - 1))

    def sample(self, n_samples, rng):
        """Draw joint realizations from N(mu, L L^T); returns (fs, gs), each
        of shape (n_samples, N)."""
        x = rng.standard_normal((n_samples, self.L.shape[1])) @ self.L.T
        n = len(self.mean_f)
        return self.mean_f + x[:, :n], self.mean_g + x[:, n:]
