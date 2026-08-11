"""Shared definition of the sparse nonconvex QCQP, used by qcqp_sample.py
(illustrative single-instance plots).

See the accompanying paper (arXiv:2509.02894). The problem is the general
QCQP class of Park & Boyd (arXiv:1703.07870) with an l1 penalty added:

    minimize    1/2 x'P0 x + q0'x + lmbda ||x||_1
    subject to  1/2 x'Pi x + qi'x - ri <= 0,   i = 1..m

Why this example rather than the sparse portfolio one: it exercises the
setting of the paper in full, which the portfolio problem does not.

    P0 indefinite            -> f1 is genuinely NONCONVEX
    Pi quadratic             -> g is genuinely NONLINEAR (its Jacobian
                                varies along the iterates, unlike the affine
                                portfolio constraints)
    l1 acts in the interior  -> f2 is a prox term

Instance construction (see make_instance for the details and the .tex for
the write-up). P0 is built from a symmetric Gaussian matrix whose spectrum is
then rescaled and shifted, so that the curvature scale and the fraction of
negative eigenvalues vary from instance to instance; each Pi = Ai'Ai/n is PSD
with its own scale, and ri > 0. Hence for every draw:

  f1 nonconvex        P0 symmetric indefinite (both signs present)
  g nonlinear         each constraint an ellipsoid, bounded intersection
  x = 0 feasible      STRICTLY, since every ri > 0

The generator varies problem character (dimension, m/n, conditioning, degree
of nonconvexity, sparsity level), not just size: a set drawn from one fixed
construction is not really N problems, however many instances are drawn, and
a profile over it measures within-family scatter rather than breadth.

That last point matters practically: FS-P-ALM requires a feasible starting
point, and here one is available in closed form, so no phase I solve is
needed and the benchmark measures the outer algorithm rather than a
feasibility heuristic.

lmbda = c * ||q0||_inf with c in (0,1): for the unconstrained problem the
i-th coordinate of the solution vanishes when |(q0)_i| <= lmbda, so c
controls sparsity directly and is invariant to the scaling of the data.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------- instances ---
class Instance:
    """One sparse nonconvex QCQP instance."""

    def __init__(self, n, m, P0, q0, Ps, qs, rs, lmbda, c, seed):
        self.n, self.m = n, m
        self.P0, self.q0 = P0, q0
        self.Ps, self.qs, self.rs = Ps, qs, rs      # Ps: (m, n, n)
        self.lmbda, self.c, self.seed = lmbda, c, seed
        self.name = f"qcqp_n{n}_m{m}_s{seed}"

    # --- quantities the benchmark driver re-checks independently ---
    def objective(self, x):
        """F(x) = 1/2 x'P0x + q0'x + lmbda||x||_1, identical for every solver."""
        x = np.asarray(x, dtype=float)
        P0, q0 = np.asarray(self.P0), np.asarray(self.q0)
        return float(0.5 * x @ P0 @ x + q0 @ x + self.lmbda * np.sum(np.abs(x)))

    def constraints(self, x):
        """g(x) in R^m."""
        x = np.asarray(x, dtype=float)
        Ps, qs, rs = np.asarray(self.Ps), np.asarray(self.qs), np.asarray(self.rs)
        return 0.5 * np.einsum("mij,i,j->m", Ps, x, x) + qs @ x - rs

    def violation(self, x):
        """max{||h||_inf, ||[g]_+||_inf}; no equality constraints here."""
        return float(np.max(np.maximum(self.constraints(x), 0.0)))

    def x0(self):
        """x = 0, strictly feasible by construction since every r_i > 0."""
        return jnp.zeros(self.n)

    def nnz(self, x, tol=1e-6):
        return int(np.sum(np.abs(np.asarray(x)) > tol))


# Pool the dimension is drawn from when n is not pinned. Kept as a module
# constant so a batch driver can override it without editing this file: the
# spread of n is what makes the profile a statement about the method rather
# than about one size.
DIMS = (30, 50, 80, 120, 200)

# Range of the constraint-to-variable ratio m/n, likewise overridable.
MRATIO = (0.05, 0.4)


def make_instance(seed, n=None, m=None, c=None, rng=None, dims=None,
                  mratio=None, cond0=None, con_scale=None, par=0.0):
    """Generate one random instance.

    The generator deliberately varies problem *character*, not just size. A
    benchmark set drawn from a single fixed construction is not really N
    problems: however many instances you draw, they share one conditioning,
    one degree of nonconvexity and one sparsity level, so a performance
    profile over them measures within-family scatter rather than breadth.
    Raising the instance count does not fix that; varying the regimes does.

    What varies per instance:
      n            dimension
      m/n          ratio of constraints to variables (0.05 to 0.4)
      spectrum     P0's eigenvalues are rescaled and shifted, so both the
                   curvature scale (10^-1 to 10^1) and the fraction of
                   negative eigenvalues (roughly 25% to 75%) change: the
                   problems range from mildly to strongly nonconvex
      constraints  each ellipsoid gets its own scale, so the constraint
                   curvature relative to the objective varies
      c            sparsity level of the l1 term (0.2 to 0.8)

    Invariants preserved for every draw: P0 is symmetric and indefinite (so
    f1 is genuinely nonconvex), every Pi is PSD with r_i > 0 (so the feasible
    set is a bounded intersection of ellipsoids and x = 0 is strictly
    feasible). Pass n, m or c explicitly to pin any of them.

    To change the SIZE of the generated problems without pinning it, pass a
    different pool: dims=(200, 400) draws n from that pool instead of DIMS,
    and mratio=(lo, hi) changes the range m/n is drawn from. Cost grows fast
    in n -- each of the m+1 matrices is n x n and P0 needs an eigendecomposition
    -- so a pool topping out at 500 is a different benchmark in runtime terms,
    not just in dimension.

    HARDNESS KNOBS. The default draw is numerically benign: measured, the
    penalty never has to exceed nu ~ 1e0, so the subproblems stay well
    conditioned and nothing distinguishes a controlled penalty schedule from a
    multiplicative one. Three optional knobs make the instance genuinely hard,
    each targeting a different mechanism:

      cond0      condition number imposed on |spec(P0)| (log-spaced magnitudes,
                 signs still mixed so P0 stays indefinite). Hard objective.
      con_scale  multiplies P_i, q_i, r_i. Flatter constraints need a LARGER
                 penalty to enforce: the required nu scales like 1/con_scale^2,
                 so 1e-3 pushes it from ~1e0 to ~1e5. Note this rescales the
                 constraint FUNCTIONS, not the feasible set, so it tests
                 robustness to constraint scaling rather than changing the
                 geometry.
      par        in [0,1]: fraction of a shared direction mixed into every q_i,
                 making the constraint gradients nearly PARALLEL. This is the
                 one that ill-conditions J'J itself, i.e. a near-degenerate
                 active set, and it is what drives the augmented Lagrangian
                 Hessian grad^2 f + nu J'J badly conditioned as nu grows.

    x = 0 remains strictly feasible for every setting (r_i > 0 throughout).
    """
    rng = rng or np.random.default_rng(seed)
    dims = tuple(dims) if dims else DIMS
    lo, hi = mratio if mratio else MRATIO
    if n is None:
        n = int(rng.choice(dims))
    if m is None:
        m = max(2, int(round(n * rng.uniform(lo, hi))))
    if c is None:
        c = float(rng.uniform(0.2, 0.8))

    # --- objective: symmetric, with controlled spectrum -------------------
    B = rng.standard_normal((n, n))
    S = (B + B.T) / (2.0 * np.sqrt(n))
    w, V = np.linalg.eigh(S)
    if cond0 is None:
        w = w / np.max(np.abs(w))                 # normalise to [-1, 1]
        shift = float(rng.uniform(-0.25, 0.25))   # moves the +/- eigenvalue split
        scale = float(10 ** rng.uniform(-1.0, 1.0))   # curvature scale
        w = scale * (w + shift)
        if not (w.min() < 0.0 < w.max()):         # keep it genuinely indefinite
            w = w - np.median(w)
    else:
        # impose kappa(P0) = cond0 exactly: magnitudes log-spaced over
        # [1/cond0, 1], signs kept from the random draw so P0 stays indefinite
        sgn = np.sign(w)
        sgn[sgn == 0] = 1.0
        if not (sgn.min() < 0 < sgn.max()):
            sgn[0], sgn[-1] = -1.0, 1.0
        w = sgn * np.logspace(-np.log10(float(cond0)), 0.0, n)
    P0 = (V * w) @ V.T
    P0 = 0.5 * (P0 + P0.T)                        # symmetrise against round-off
    q0 = rng.standard_normal(n)

    # --- constraints: PSD ellipsoids, each with its own scale -------------
    cs = 1.0 if con_scale is None else float(con_scale)
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)                        # shared direction for `par`
    Ps = np.empty((m, n, n))
    qs = np.empty((m, n))
    for i in range(m):
        A = rng.standard_normal((n, n))
        Ps[i] = cs * (A.T @ A / n) * float(10 ** rng.uniform(-0.5, 0.5))
        qi = rng.standard_normal(n)
        if par:
            qi = (1.0 - par) * qi + par * np.linalg.norm(qi) * u
        qs[i] = cs * qi
    rs = cs * rng.uniform(0.5, 2.0, size=m)       # > 0 -> x=0 strictly feasible

    lmbda = float(c * np.max(np.abs(q0)))
    return Instance(n=n, m=m, P0=jnp.array(P0), q0=jnp.array(q0),
                    Ps=jnp.array(Ps), qs=jnp.array(qs), rs=jnp.array(rs),
                    lmbda=lmbda, c=c, seed=seed)


def make_instances(n_instances, seed0=0, c=None, dims=None, mratio=None,
                   cond0=None, con_scale=None, par=0.0):
    """c=None (default) lets the sparsity level vary per instance too.

    dims/mratio are forwarded to make_instance: they control the size of the
    problems, while n_instances controls how many are drawn. cond0/con_scale/
    par are the HARDNESS KNOBS documented in make_instance -- forwarded
    identically (not randomised) to every instance, so a comparison across
    this set holds difficulty fixed and varies only the random structure
    (size, spectrum, sparsity) the default draw already varies.
    """
    return [make_instance(seed0 + i, c=c, dims=dims, mratio=mratio,
                          cond0=cond0, con_scale=con_scale, par=par)
            for i in range(n_instances)]


# -------------------------------------------------- solver-side builders ---
def fs_palm_problem(inst, fs_palm_mod, jittable=True):
    """Build the fs_palm.Problem for this instance."""
    P0, q0, Ps, qs, rs = inst.P0, inst.q0, inst.Ps, inst.qs, inst.rs

    def f1(x):
        return 0.5 * x @ P0 @ x + q0 @ x

    def g(x):
        return 0.5 * jnp.einsum("mij,i,j->m", Ps, x, x) + qs @ x - rs

    return fs_palm_mod.Problem(f1=f1, f2=fs_palm_mod.L1Norm(inst.lmbda),
                             g=[g], jittable=jittable)


def solve_ipopt(inst, tol=1e-8, max_iter=3000, print_level=0, x0=None):
    """Solve with IPOPT via cyipopt, using the standard epigraph split of the
    l1 term (IPOPT cannot take a nonsmooth objective):

        min  1/2 x'P0x + q0'x + lmbda 1't
        s.t. g(x) <= 0,  x - t <= 0,  -x - t <= 0,  t >= 0

    in the variable (x, t) in R^{2n}. This is exact, not a smoothing.

    Note the problem is nonconvex, so IPOPT returns a local solution; there
    is no global reference. That is precisely why the benchmark's success
    test uses Birgin & Martinez's criterion (63), which compares each
    solver's objective against the best value found by the solvers being
    compared rather than against a known optimum.
    """
    import cyipopt

    n, m = inst.n, inst.m
    P0 = np.asarray(inst.P0, dtype=float)
    q0 = np.asarray(inst.q0, dtype=float)
    Ps = np.asarray(inst.Ps, dtype=float)
    qs = np.asarray(inst.qs, dtype=float)
    rs = np.asarray(inst.rs, dtype=float)
    lm = inst.lmbda
    N = 2 * n
    tril = np.tril_indices(N)

    class _P:
        def objective(self, z):
            x, t = z[:n], z[n:]
            return float(0.5 * x @ P0 @ x + q0 @ x + lm * np.sum(t))

        def gradient(self, z):
            x = z[:n]
            return np.concatenate([P0 @ x + q0, lm * np.ones(n)])

        def constraints(self, z):
            x, t = z[:n], z[n:]
            gq = 0.5 * np.einsum("mij,i,j->m", Ps, x, x) + qs @ x - rs
            return np.concatenate([gq, x - t, -x - t])

        def jacobian(self, z):
            x = z[:n]
            # d/dx of the quadratic rows, zero in t
            Jq = np.einsum("mij,j->mi", Ps, x) + qs
            top = np.hstack([Jq, np.zeros((m, n))])
            mid = np.hstack([np.eye(n), -np.eye(n)])
            bot = np.hstack([-np.eye(n), -np.eye(n)])
            return np.vstack([top, mid, bot]).ravel()

        def hessianstructure(self):
            return tril

        def hessian(self, z, lagrange, obj_factor):
            # objective contributes P0 in the x block; the l1 epigraph rows
            # are linear; the quadratic constraints contribute sum_i lam_i Pi
            Hx = obj_factor * P0 + np.einsum("m,mij->ij", lagrange[:m], Ps)
            H = np.zeros((N, N))
            H[:n, :n] = Hx
            return H[tril]

    n_con = m + 2 * n
    nlp = cyipopt.Problem(
        n=N, m=n_con, problem_obj=_P(),
        lb=np.concatenate([np.full(n, -1e20), np.zeros(n)]),
        ub=np.full(N, 1e20),
        cl=np.full(n_con, -1e20), cu=np.zeros(n_con),
    )
    nlp.add_option("tol", tol)
    nlp.add_option("max_iter", max_iter)
    nlp.add_option("print_level", print_level)
    nlp.add_option("sb", "yes")

    z0 = np.zeros(N) if x0 is None else np.concatenate(
        [np.asarray(x0, dtype=float), np.abs(np.asarray(x0, dtype=float))])
    z, info = nlp.solve(z0)
    return np.asarray(z[:n]), info


if __name__ == "__main__":
    # This module is the shared problem definition, imported by qcqp_sample.py.
    # Run directly it first checks the instance and both solvers, then
    # produces the sample plots by handing off to qcqp_sample (which is where
    # the plotting lives, and which takes command-line options -- see
    # `python qcqp_sample.py --help`).
    # Prefer this repo's solver over any installed fs_palm. A pip-installed fs_palm
    # in the same environment would otherwise shadow it and the script would
    # exercise the published package instead of the working copy.
    import os as _os, sys as _sys
    _SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))), "src")
    if _os.path.isdir(_os.path.join(_SRC, "fs_palm")) and _SRC not in _sys.path:
        _sys.path.insert(0, _SRC)

    import fs_palm

    inst = make_instance(0, n=50, m=10)
    x0 = np.asarray(inst.x0())
    ev = np.linalg.eigvalsh(np.asarray(inst.P0))
    print(f"instance {inst.name}")
    print(f"  n={inst.n} m={inst.m} lmbda={inst.lmbda:.4f} (c={inst.c})")
    print(f"  P0 eigenvalues in [{ev.min():.3f}, {ev.max():.3f}] -> "
          f"{'INDEFINITE (nonconvex)' if ev.min() < 0 < ev.max() else 'definite'}")
    print(f"  x0=0: F={inst.objective(x0):.6e}  violation={inst.violation(x0):.3e} "
          f"(strictly feasible: {inst.violation(x0) == 0.0})")

    x_ip, info = solve_ipopt(inst)
    print(f"  IPOPT : F={inst.objective(x_ip):+.8e}  "
          f"violation={inst.violation(x_ip):.2e}  nnz={inst.nnz(x_ip)}/{inst.n}  "
          f"status={info['status']}")

    prob = fs_palm_problem(inst, fs_palm, jittable=True)
    sol = fs_palm.solve(prob, inst.x0(), tol=1e-6, max_iter=500,
                      use_proximal=True, phi_strategy="pow", xi1=1.0, xi2=1.0,
                      alpha=4, verbosity=0)
    x_pb = np.asarray(sol.x)
    print(f"  FS-P-ALM: F={inst.objective(x_pb):+.8e}  "
          f"violation={inst.violation(x_pb):.2e}  nnz={inst.nnz(x_pb)}/{inst.n}  "
          f"status={sol.solve_status}  grad_evals={sol.grad_evals[-1]}")

    # sample.main() defaults to --inner-solver both, so this writes a
    # SEPARATE set of figures for PANOC and for ProxGrad. The two are not
    # interchangeable: PANOC is quasi-Newton and its L-BFGS directions
    # absorb the subproblem ill-conditioning a growing penalty induces,
    # which is exactly the effect the penalty schedule is meant to control.
    print("\ngenerating sample plots for BOTH inner solvers "
          "(PANOC and ProxGrad)...\n")
    import qcqp_sample
    qcqp_sample.main()
