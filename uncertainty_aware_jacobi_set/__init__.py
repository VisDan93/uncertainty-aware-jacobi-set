"""uncertainty_aware_jacobi_set -- uncertainty-aware Jacobi set computation.

Reference implementation of

    Kloetzl and Weiskopf, "Uncertainty-Aware Jacobi Set Computation",
    IEEE Workshop on Uncertainty Visualization, 2026.

Modules
-------
geometry    triangulated grid, PL gradient operators, interior edges
models      Gaussian input model (FieldPair) from ensembles or kernels
jacobi      the uncertainty-aware Jacobi set (Algorithm 1) and the PL Jacobi set
vis         edge probabilities, (signed) expected Jacobi set degree, lines
bilinear    optional reduced-connectivity local bilinear line geometry
data        analytic test dataset with closed-form Jacobi set
"""

from . import bilinear, data, geometry, jacobi, models, vis
from .models import FieldPair

__version__ = "1.0.0"
