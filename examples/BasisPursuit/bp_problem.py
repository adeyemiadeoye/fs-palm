"""Shared definition of the nonconvex basis-pursuit problem, used by
bp_sample.py (illustrative single-instance convergence plots).

See the accompanying paper (arXiv:2509.02894), Sec. "Basis pursuit
(nonconvex formulation)". Basis pursuit seeks the sparsest solution of an
underdetermined linear system,

    minimize ||z||_1   subject to   B z = b,        B in R^{p x n}, p < n,

and we solve the nonconvex reformulation of Sahin et al. (2019)

    minimize ||x||^2   subject to   Bbar x^o2 = b,  x = (x1, x2) in R^{2n},

with Bbar = [B, -B] and o2 the elementwise square. Writing z = z+ - z- for
the positive and negative parts, a minimiser has x1^o2 = z+, x2^o2 = z-.

WHY THE REFORMULATION IS EXACT. For fixed z, minimising ||x1||^2 + ||x2||^2
over x1^o2 - x2^o2 = z gives x1^o2 = z+, x2^o2 = z-, whose value is exactly
||z||_1. So the two problems have the same optimal value, and the l1 term
has been traded for a smooth objective and nonlinear equality constraints.

TWO PROPERTIES OF THIS FORMULATION THAT DRIVE THE CODE BELOW.

(1) The reference value is EXACT, not "best found". The planted z* is
    k-sparse and, in the regime used here (p >> k log(n/k)), it is the unique
    l1 minimiser, so f* = ||z*||_1 is the GLOBAL optimum. verify_optimum()
    checks this per instance with an LP rather than assuming it -- the
    property is a theorem about random B with high probability, not a
    certainty, and it degrades as k/p grows.

(2) ZERO COORDINATES ARE INVARIANT. The augmented Lagrangian has

        grad_j L_A(x) = 2 x_j [1 + Bbar_j' (lbda + rho h(x))],

    which is PROPORTIONAL to x_j. Any gradient-based method therefore leaves
    a coordinate that is exactly zero at zero, forever. This is a trap for
    the obvious feasible start: taking any z with Bz = b and setting
    x1 = sqrt(z+), x2 = sqrt(z-) zeroes one of the two coordinates for every
    index (roughly half of the 2n entries), which freezes sign(z_i) there and
    confines the solver to a single orthant. x0() therefore uses the offset
    construction described below, which is feasible AND strictly nonzero.

Instance construction follows the manuscript: B has i.i.d. N(0,1) entries,
z* has k nonzeros at positions drawn uniformly with i.i.d. N(0,1)
amplitudes, and b = B z*.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


# Pool of (p, n) shapes drawn from when the size is not pinned, and the
# sparsity range. Module constants so a batch driver can override them from
# the command line (--sizes, --krange) without editing this file.
#
# The UNDERSAMPLING RATIO p/n is varied deliberately, not just the size: it is
# the quantity that governs how hard l1 recovery is, so a pool at one fixed
# ratio (e.g. 100/256, 200/512, 400/1024 are all 0.39) would draw instances
# that differ only in scale. Ratios here span roughly 0.24 to 0.5.
SIZES = ((100, 256), (128, 256), (128, 512), (200, 512), (256, 512),
         (200, 768), (300, 768), (250, 1024), (400, 1024))
KRANGE = (5, 20)


class Instance:
    """One nonconvex basis-pursuit instance."""

    def __init__(self, p, n, k, B, b, z_star, seed):
        self.p, self.n, self.k = p, n, k
        self.B, self.b, self.z_star = B, b, z_star
        self.seed = seed
        self.Bbar = jnp.array(np.concatenate([B, -B], axis=1))
        self.bj = jnp.array(b)
        self.name = f"bp_p{p}_n{n}_k{k}_s{seed}"

    # --- quantities the benchmark driver re-checks independently ---
    @property
    def f_star(self):
        """Global optimum ||z*||_1 -- see verify_optimum()."""
        return float(np.abs(self.z_star).sum())

    def objective(self, x):
        """f(x) = ||x||^2, identical for every solver."""
        x = np.asarray(x, dtype=float)
        return float(x @ x)

    def z_of(self, x):
        """Recover z = x1^o2 - x2^o2 from the lifted variable."""
        x = np.asarray(x, dtype=float)
        return x[:self.n] ** 2 - x[self.n:] ** 2

    def constraints(self, x):
        """h(x) = Bbar x^o2 - b in R^p."""
        return self.B @ self.z_of(x) - self.b

    def violation(self, x):
        return float(np.max(np.abs(self.constraints(x))))

    def nnz(self, x, tol=1e-6):
        return int(np.sum(np.abs(self.z_of(x)) > tol))

    def recovery_error(self, x):
        """||z - z*||_inf: did we find the planted signal, not just a
        point of equal objective?"""
        return float(np.max(np.abs(self.z_of(x) - self.z_star)))

    # --- starting point -------------------------------------------------
    def x0(self, c=1.0):
        """A STRICTLY FEASIBLE, fully nonzero starting point, in closed form.

        Take the minimum-norm solution z_ls = B'(BB')^{-1} b of Bz = b (B has
        full row rank almost surely), then split it as

            x1 = sqrt(z_ls + t),   x2 = sqrt(t),   t = |z_ls| + c,   c > 0,

        so that x1^o2 - x2^o2 = z_ls exactly and hence Bbar x0^o2 = B z_ls = b.
        Feasibility is exact up to the linear solve (~1e-14 in practice).

        The offset c is what keeps every coordinate strictly positive: with
        c = 0 this degenerates to x1 = sqrt(z+), x2 = sqrt(z-), which zeroes
        one coordinate per index, and by the gradient identity in the module
        docstring those stay zero for the whole run. Do not set c = 0.

        This supplies the feasible initialisation FS-P-ALM requires (Table 1)
        without a phase I solve, so the experiment measures the outer
        algorithm rather than a feasibility heuristic.
        """
        if c <= 0:
            raise ValueError("c must be > 0: c=0 gives a starting point whose "
                             "zero coordinates are invariant under the solver "
                             "(see the module docstring)")
        z_ls = self.B.T @ np.linalg.solve(self.B @ self.B.T, self.b)
        t = np.abs(z_ls) + c
        return jnp.array(np.concatenate([np.sqrt(z_ls + t), np.sqrt(t)]))

    def verify_optimum(self, tol=1e-7):
        """Check that z* really is the l1 minimiser, by solving the convex BP
        as an LP.  Returns (is_optimal, ||z_lp||_1, ||z_lp - z*||_inf).

        Recovery is a high-probability statement, so it is checked rather
        than assumed: if it fails, f* = ||z*||_1 is NOT the optimal value and
        every suboptimality curve plotted against it is wrong.
        """
        from scipy.optimize import linprog
        n = self.n
        res = linprog(c=np.ones(2 * n),
                      A_eq=np.hstack([self.B, -self.B]), b_eq=self.b,
                      bounds=[(0, None)] * (2 * n), method="highs")
        if not res.success:
            return False, np.nan, np.nan
        z_lp = res.x[:n] - res.x[n:]
        l1 = float(np.abs(z_lp).sum())
        err = float(np.max(np.abs(z_lp - self.z_star)))
        return (abs(l1 - self.f_star) <= tol * max(1.0, self.f_star)), l1, err


def make_instance(seed, p=None, n=None, k=None, rng=None, sizes=None,
                  krange=None):
    """Generate one random instance.

    The generator varies problem character, not only size: the shape (p, n),
    the undersampling ratio p/n and the sparsity k all move across the test
    set. The pair (p/n, k/p) is what matters -- it
    places the instance relative to the phase transition of l1 recovery, i.e.
    decides whether recovery is easy, marginal or impossible -- so k is drawn
    per instance rather than fixed at 10, and the shape pool spans several
    ratios rather than one.

    Instances near or past the transition are not discarded here: they are
    legitimate problems, and verify_optimum() reports whether f* = ||z*||_1
    is still the optimal value, so callers can skip only those where the
    reference itself would be wrong.

    Pass p, n or k explicitly to pin any of them; pass sizes/krange to change
    the pools they are drawn from.
    """
    rng = rng or np.random.default_rng(seed)
    sizes = tuple(sizes) if sizes else SIZES
    if p is None or n is None:
        p_d, n_d = sizes[int(rng.integers(len(sizes)))]
        p = p if p is not None else p_d
        n = n if n is not None else n_d
    if k is None:
        klo, khi = krange if krange else KRANGE
        k = int(rng.integers(klo, khi + 1))

    B = rng.standard_normal((p, n))
    z_star = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    z_star[support] = rng.standard_normal(k)
    b = B @ z_star
    return Instance(p=p, n=n, k=k, B=B, b=b, z_star=z_star, seed=seed)


def make_instances(n_instances, seed0=0, sizes=None, krange=None):
    return [make_instance(seed0 + i, sizes=sizes, krange=krange)
            for i in range(n_instances)]


# -------------------------------------------------- solver-side builders ---
def fs_palm_problem(inst, fs_palm_mod, jittable=True):
    """Build the fs_palm.Problem for this instance."""
    Bbar, bj = inst.Bbar, inst.bj

    def f1(x):
        return jnp.sum(x ** 2)

    def h(x):
        return Bbar @ (x ** 2) - bj

    return fs_palm_mod.Problem(f1=f1, h=[h], jittable=jittable)


if __name__ == "__main__":
    # Sanity-check the construction, then hand over to the sample plots --
    # the same entry-point convention as qcqp_problem.py.
    print("checking the instance construction\n")
    for seed in range(3):
        inst = make_instance(seed, p=100, n=256)
        x0 = inst.x0()
        ok, l1, err = inst.verify_optimum()
        print(f"  {inst.name:22s} ||h(x0)||_inf={inst.violation(x0):.2e} "
              f"f(x0)={inst.objective(x0):>10.3f} f*={inst.f_star:.6f} "
              f"LP l1={l1:.6f} {'OK' if ok else 'MISMATCH'} "
              f"||z_lp-z*||={err:.1e}")
        assert np.min(np.abs(np.asarray(x0))) > 0, "x0 has a zero coordinate"
    print("\n  x0 is feasible and strictly nonzero; z* is the l1 optimum\n")

    # sample.main() defaults to --inner-solver both, so this writes a
    # SEPARATE set of figures for PANOC and for ProxGrad. The two are not
    # interchangeable: PANOC is quasi-Newton and its L-BFGS directions
    # absorb the subproblem ill-conditioning a growing penalty induces,
    # which is exactly the effect the penalty schedule is meant to control.
    print("\ngenerating sample plots for BOTH inner solvers "
          "(PANOC and ProxGrad)...\n")
    import bp_sample
    bp_sample.main()
