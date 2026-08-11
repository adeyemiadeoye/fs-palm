"""Shared definition of the convex QP experiments, used by qp_sample.py
(illustrative per-problem convergence plots) and alpaqa_alm_bench.py (the
CUTEst performance profiles).

See the accompanying paper (arXiv:2509.02894), Sec. "Convex QPs". The
problem is \\eqref{eq:prob} with

    f1(x) = 1/2 x'Qx + q'x + c,     Q psd
    f2(x) = indicator of the box  Omega = {x : x_l <= x <= x_u}
    g(x)  = [Ax - u,  l - Ax]                     (two-sided linear)

TWO SOURCES OF INSTANCES, ONE INTERFACE.

  load_cutest(name)   selected Maros-Meszaros problems (DUALC1, CVXQP2_M and
                      LOTSCHD for the condition-number comparison; GENHS28
                      and AUG3D for the alpha/xi parameter sweep), read
                      through alpaqa's CUTEst bindings.
                      Requires the problems to be compiled -- see the
                      docstring -- which is a machine-local prerequisite.

  make_instance(...)  synthetic convex QPs with a PRESCRIBED condition number
                      kappa(Q). The purpose is to show behaviour "under
                      different levels of complexity" via kappa(Q), but
                      with named MM problems kappa is whatever those problems
                      happen to have: it cannot be varied, so the effect of
                      conditioning cannot be separated from everything else
                      that differs between DUALC1, CVXQP2_M and LOTSCHD.
                      Generating Q with a chosen spectrum makes conditioning
                      an independent variable, which is what a claim about
                      conditioning needs.

Both produce an `Instance` with the same interface, so qp_sample.py and
alpaqa_alm_bench.py work with either.

REFERENCE VALUE. These problems are CONVEX, so f* is the global optimum and
is computed once per instance with an interior-point solver (Clarabel via
cvxpy) rather than taken as "the best value any solver found". Cached on the
instance.

FEASIBLE START. Synthetic instances are built so that a strictly feasible x0
is known in closed form (the constraint bounds are placed around A x0), which
keeps phase I out of the measurement. CUTEst instances have no such guarantee
and fall back to phase I, as the manuscript describes.
"""
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
# Prefer this repo's solver over any installed pbalm. A pip-installed pbalm
# in the same environment would otherwise shadow it and the script would
# benchmark the published package instead of the working copy.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "src")
if _os.path.isdir(_os.path.join(_SRC, "pbalm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import pbalm

jax.config.update("jax_platform_name", "cpu")
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


# Named Maros-Meszaros problems used by the example scripts.
MM_COND_PROBLEMS = ("DUALC1", "CVXQP2_M", "LOTSCHD")   # condition-number comparison
MM_SWEEP_PROBLEMS = ("GENHS28", "AUG3D")                # alpha/xi parameter sweep

# Condition numbers used when sweeping difficulty synthetically. Spread over
# six orders so that "well conditioned" and "badly conditioned" are genuinely
# different regimes rather than two draws from one.
CONDS = (1e2, 1e4, 1e6, 1e8)


class Instance:
    """One convex QP in the form above."""

    def __init__(self, name, Q, q, c, A, lo, hi, x_lb, x_ub, x0=None):
        self.name = name
        self.Q, self.q, self.c = Q, q, c
        self.A, self.lo, self.hi = A, lo, hi
        self.x_lb, self.x_ub = x_lb, x_ub
        self.n = Q.shape[0]
        self.m = 0 if A is None else A.shape[0]
        self._x0 = x0
        self._f_star = None
        self._cond = None

    @property
    def cond(self):
        if self._cond is None:
            w = np.linalg.eigvalsh(self.Q)
            pos = w[w > max(1e-14, 1e-14 * np.max(np.abs(w)))]
            self._cond = float(np.max(w) / np.min(pos)) if pos.size else np.inf
        return self._cond

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(0.5 * x @ self.Q @ x + self.q @ x + self.c)

    def violation(self, x):
        """max violation of the box and the two-sided linear constraints."""
        x = np.asarray(x, dtype=float)
        v = [np.max(np.maximum(self.x_lb - x, 0.0)) if np.any(np.isfinite(self.x_lb)) else 0.0,
             np.max(np.maximum(x - self.x_ub, 0.0)) if np.any(np.isfinite(self.x_ub)) else 0.0]
        if self.A is not None:
            Ax = self.A @ x
            v.append(np.max(np.maximum(Ax - self.hi, 0.0)))
            v.append(np.max(np.maximum(self.lo - Ax, 0.0)))
        return float(max(v))

    def x0(self):
        """Starting point. Strictly feasible by construction for synthetic
        instances; for CUTEst instances this is a random point and phase I
        does the work."""
        return self._x0

    @property
    def f_star(self):
        """Global optimum, computed once with an interior-point solver."""
        if self._f_star is None:
            self._f_star = solve_reference(self)
        return self._f_star


def solve_reference(inst, solver=None, verbose=False):
    """Global optimum via cvxpy. Convex, so this is the true f*."""
    import cvxpy as cp

    x = cp.Variable(inst.n)
    obj = 0.5 * cp.quad_form(x, cp.psd_wrap(inst.Q)) + inst.q @ x + inst.c
    cons = []
    if np.any(np.isfinite(inst.x_lb)):
        cons.append(x >= np.where(np.isfinite(inst.x_lb), inst.x_lb, -1e20))
    if np.any(np.isfinite(inst.x_ub)):
        cons.append(x <= np.where(np.isfinite(inst.x_ub), inst.x_ub, 1e20))
    if inst.A is not None:
        cons += [inst.A @ x <= inst.hi, inst.A @ x >= inst.lo]
    prob = cp.Problem(cp.Minimize(obj), cons)
    # Clarabel is an interior-point method and reaches the accuracy a
    # reference value needs; first-order solvers (OSQP/SCS) do not, and a
    # loose f* would show up as a floor in every suboptimality curve.
    for s in ([solver] if solver else ["CLARABEL", "OSQP", "SCS"]):
        try:
            prob.solve(solver=s, verbose=verbose)
            if x.value is not None and prob.status in ("optimal",
                                                       "optimal_inaccurate"):
                return float(prob.value)
        except Exception:
            continue
    raise RuntimeError(f"no cvxpy solver succeeded on {inst.name}")


def make_instance(seed, n=100, m=50, cond=1e4, n_active=None, box=10.0,
                  rng=None):
    """Synthetic convex QP with a PRESCRIBED condition number kappa(Q) = cond.

    Q = V diag(w) V' with w log-spaced over [1, cond] and V a random
    orthogonal matrix, so kappa(Q) is exactly `cond` rather than an emergent
    property of the construction. This is what makes conditioning an
    independent variable: everything else about the instance is held fixed
    while it is swept.

    The linear constraints are placed around a chosen interior point x_feas,
    with `n_active` of them made tight at it, so that

      * x0 = x_feas is STRICTLY feasible for the inequalities and inside the
        box, giving the feasible start P-BALM needs with no phase I, and
      * the solution is genuinely constrained -- without some tight rows the
        QP would reduce to its unconstrained minimiser and the constraint
        handling, which is what the paper is about, would not be exercised.
    """
    rng = rng or np.random.default_rng(seed)
    n_active = max(1, m // 4) if n_active is None else n_active

    # --- objective with a prescribed spectrum ---------------------------
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w = np.logspace(0.0, np.log10(cond), n)
    Q = (V * w) @ V.T
    Q = 0.5 * (Q + Q.T)
    q = rng.standard_normal(n)
    c = 0.0

    # --- a strictly feasible interior point ------------------------------
    x_feas = rng.uniform(-1.0, 1.0, size=n)

    A = rng.standard_normal((m, n)) / np.sqrt(n)
    Ax = A @ x_feas
    # slack > 0 everywhere, then tighten a few rows to (near) activity
    slack = rng.uniform(0.5, 2.0, size=m)
    idx = rng.choice(m, size=min(n_active, m), replace=False)
    slack[idx] = 1e-3
    hi = Ax + slack
    lo = Ax - rng.uniform(0.5, 2.0, size=m)

    x_lb = np.full(n, -box)
    x_ub = np.full(n, box)

    name = f"qp_n{n}_m{m}_c{cond:.0e}_s{seed}"
    return Instance(name=name, Q=Q, q=q, c=c, A=A, lo=lo, hi=hi,
                    x_lb=x_lb, x_ub=x_ub, x0=jnp.array(x_feas))


def make_instances(n_instances, seed0=0, n=100, m=50, conds=None):
    """One instance per (seed, condition number) pair."""
    conds = tuple(conds) if conds else CONDS
    out = []
    for i in range(n_instances):
        for cd in conds:
            out.append(make_instance(seed0 + i, n=n, m=m, cond=cd))
    return out


def load_cutest(name, cutest_dir=None, seed=1234):
    """Load a Maros-Meszaros QP through alpaqa's CUTEst bindings.

    Expects the problem precompiled at <cutest_dir>/<NAME>/<NAME>.so with its
    OUTSDIF.d alongside; the default directory is ~/opt/CUTEST/compiled_QP_MM
    (override with the CUTEST_QP_DIR environment variable). Nothing is
    checked into this repo for rebuilding -- decode the SIF file with
    SIFDecode's sifdecoder, then compile with, e.g.,
        gfortran -O3 -shared -fPIC -o NAME.so *.f \\
            -Wl,--whole-archive $CUTEST/builddir/libcutest_double.a \\
            -Wl,--no-whole-archive
    (whole-archive matters: alpaqa's CUTEstProblem dlopen()s NAME.so
    expecting it to be self-contained, but *.f alone references no symbol
    in libcutest_double, so a plain -lcutest_double link drops the
    dependency entirely under Ubuntu's --as-needed default, producing a .so
    that fails at load time with "undefined symbol: fortran_close_").

    The starting point is random and generally INFEASIBLE, so phase I runs --
    as the manuscript describes for problems where a feasible point is not
    readily available.
    """
    import alpaqa as pa

    root = Path(cutest_dir or os.environ.get("CUTEST_QP_DIR")
                or (Path.home() / "opt" / "CUTEST" / "compiled_QP_MM"))
    so = root / name / f"{name}.so"
    outsdif = root / name / "OUTSDIF.d"
    if not so.is_file():
        raise FileNotFoundError(
            f"{so} not found. The MM problems must be decoded and compiled "
            f"locally; set CUTEST_QP_DIR if they live elsewhere -- see this "
            f"function's docstring.")

    prob = pa.CUTEstProblem(str(so), str(outsdif))
    n, m = prob.num_variables, prob.num_constraints
    z_n, z_m = np.zeros(n), np.zeros(m)
    # eval_lagrangian_hessian(x, y=0) is the Hessian of f1 alone (the
    # Lagrangian with zero multipliers reduces to the objective); it returns
    # only the upper triangle (Symmetry.Upper), so mirror it explicitly
    # rather than assume the unused half happens to be zero.
    H, symH = prob.eval_lagrangian_hessian(z_n, z_m)
    H = np.asarray(H)
    assert symH == pa.Symmetry.Upper, f"unexpected Hessian symmetry {symH}"
    Q = np.triu(H) + np.triu(H, k=1).T
    c = prob.eval_objective(z_n)
    q = np.asarray(prob.eval_objective_gradient(z_n))
    A, symJ = prob.eval_constraints_jacobian(z_n)
    A = np.asarray(A)
    assert symJ == pa.Symmetry.Unsymmetric, f"unexpected Jacobian symmetry {symJ}"
    g0 = np.asarray(prob.eval_constraints(z_n))
    gb = prob.general_bounds
    lo = np.asarray(gb.lower) - g0
    hi = np.asarray(gb.upper) - g0
    # Genuine IEEE infinities now (checked: this alpaqa version no longer
    # uses CUTEst's traditional +-1e20 "absent bound" sentinel), but the
    # solver still needs finite box bounds.
    lo = np.where(np.isneginf(lo), -1e32, lo)
    hi = np.where(np.isposinf(hi), 1e32, hi)
    vb = prob.variable_bounds
    x_lb = np.where(np.isneginf(vb.lower), -1e32, vb.lower)
    x_ub = np.where(np.isposinf(vb.upper), 1e32, vb.upper)

    rng = np.random.default_rng(seed)
    x0 = jnp.array(rng.standard_normal(n))
    return Instance(name=name, Q=np.asarray(Q), q=np.asarray(q), c=float(c),
                    A=np.asarray(A), lo=lo, hi=hi,
                    x_lb=x_lb, x_ub=x_ub, x0=x0)


# -------------------------------------------------- solver-side builders ---
def pbalm_problem(inst, pbalm_mod, jittable=True):
    """Build the pbalm.Problem for this instance."""
    Q = jnp.array(inst.Q)
    q = jnp.array(inst.q)
    c = float(inst.c)
    A = jnp.array(inst.A) if inst.A is not None else None
    lo = jnp.array(inst.lo) if inst.A is not None else None
    hi = jnp.array(inst.hi) if inst.A is not None else None

    def f1(x):
        return 0.5 * jnp.dot(x, Q @ x) + jnp.dot(q, x) + c

    def f1_grad(x):
        return Q @ x + q

    def g(x):
        Ax = A @ x
        return jnp.concatenate([Ax - hi, lo - Ax], axis=0)

    # Box is alpaqa's, which is bound to Eigen and takes plain float64 numpy
    # arrays -- a jnp array raises a constructor TypeError.
    box = pbalm_mod.Box(lower=np.ascontiguousarray(inst.x_lb, dtype=np.float64),
                        upper=np.ascontiguousarray(inst.x_ub, dtype=np.float64))
    return pbalm_mod.Problem(f1=f1, f1_grad=f1_grad, f2=box,
                             g=[g] if A is not None else None,
                             jittable=jittable)


if __name__ == "__main__":
    # Sanity-check the construction, then hand over to the sample plots --
    # the same entry-point convention as qcqp_problem.py / bp_problem.py.
    print("checking the synthetic QP construction\n")
    print(f"{'instance':>26s} {'kappa(Q)':>10s} {'viol(x0)':>10s} "
          f"{'f(x0)':>14s} {'f*':>14s}")
    for cd in CONDS:
        inst = make_instance(0, n=60, m=30, cond=cd)
        x0 = np.asarray(inst.x0())
        print(f"{inst.name:>26s} {inst.cond:10.2e} {inst.violation(x0):10.2e} "
              f"{inst.objective(x0):14.6f} {inst.f_star:14.6f}")
        assert inst.violation(x0) <= 1e-12, "x0 is not feasible"
    print("\n  x0 is strictly feasible and kappa(Q) tracks the request\n")

    # sample.main() defaults to --inner-solver both, so this writes a
    # SEPARATE set of figures for PANOC and for ProxGrad. The two are not
    # interchangeable: PANOC is quasi-Newton and its L-BFGS directions
    # absorb the subproblem ill-conditioning a growing penalty induces,
    # which is exactly the effect the penalty schedule is meant to control.
    print("\ngenerating sample plots for BOTH inner solvers "
          "(PANOC and ProxGrad)...\n")
    import qp_sample
    qp_sample.main()
