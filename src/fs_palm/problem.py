import numpy as np
np.NaN = np.nan
import jax
import fs_palm
from jax import grad
from .utils.utils import GradEvalCounter


class Problem:
    """
    Defines a general optimization problem for PFS-ALM.
    """
    def __init__(self, f1, f2=None, f2_lbda=None, h=None, g=None, f1_grad=None, jittable=True, callback=None):
        """
        Initializes the optimization problem.

        Parameters:
            f1: Smooth objective function --> f1(x)
            f2: Nonsmooth regularization function. Define as one of the following alpaqa proxop classes:
                    - fs_palm.L1Norm(lbda)
                    - fs_palm.Box(lower=lower, upper=upper)
                    - fs_palm.NuclearNorm(lbda)
                See alpaqa documentation for details.
            f2_lbda: Regularization parameter (float or list)
            h: List of equality constraint functions --> h_i(x) = 0, e.g., h: [h_1, h_2, ..., h_m]
            g: List of inequality constraint functions --> g_i(x) <= 0, e.g., g: [g_1, g_2, ..., g_p]
            f1_grad: Gradient of the smooth objective function --> f1_grad(x)
            h_grad: Gradient of the equality constraint functions --> h_grad(x)
            g_grad: Gradient of the inequality constraint functions --> g_grad(x)
            jittable: Boolean indicating if problem functions can be JIT-compiled (user provided functions should be JAX-compatible)
            callback: Optional callback function called at each iteration of PFS-ALM
                        callback(current_iter,
                                    x_{k},
                                    x_{k-1},
                                    lbda_{k},
                                    mu_{k},
                                    rho_{k},
                                    nu_{k},
                                    gamma_{k},
                                    x_0)
            lbda_sizes: Sizes of equality constraint multipliers (these are set automatically in PFS-ALM.solve)
            mu_sizes: Sizes of inequality constraint multipliers (these are set automatically in PFS-ALM.solve)
        """

        self.f1 = jax.jit(f1) if jittable else f1
        self.f2 = f2
        self.f2_lbda = f2_lbda
        self.h = h
        self.g = g
        self.h_grad = None
        self.g_grad = None
        self.lbda_sizes = None
        self.mu_sizes = None
        if jittable:
            self.f1_grad = GradEvalCounter(jax.jit(f1_grad) if f1_grad else jax.jit(grad(self.f1)))
        else:
            self.f1_grad = GradEvalCounter(f1_grad if f1_grad else grad(self.f1))
        self.jittable = jittable
        self.callback = callback
    def reset_counters(self):
        if hasattr(self, 'f1_grad') and hasattr(self.f1_grad, 'reset'):
            self.f1_grad.reset()
        if hasattr(self, 'h_grad') and self.h_grad is not None and hasattr(self.h_grad, 'reset'):
            self.h_grad.reset()
        if hasattr(self, 'g_grad') and self.g_grad is not None and hasattr(self.g_grad, 'reset'):
            self.g_grad.reset()