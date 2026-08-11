import numpy as np
import jax.numpy as jnp
import fs_palm
from ..utils.utils import algencan_penalty0

np.NaN = np.nan
import alpaqa as pa
import jax


class PALMInnerTrainer:
    """
    Class to run PALM inner solver with Adam and L-BFGS optimization.
    """
    def __init__(self, train_fun):
        """
        Initialize the PALMInnerTrainer.

        Parameters
        ----------
        train_fun : callable
            Function to run the solver.
            train_fun(palm_obj_fun, x, lbfgs_iters, lbfgs_tol)
                where palm_obj_fun is a function that takes in x and returns the inner objective value.
        """
        self.train_fun = train_fun

class PaProblem(pa.BoxConstrProblem):
    """
    Internal problem class for PANOC solver.
    """
    def __init__(self, obj_fun, x0, f2=None, l1_lbda=None, solver_opts=None, tol=1e-9, max_iter=2000, direction=None, jittable=True):
        super().__init__(x0.shape[0], 0)
        self.jit_loss = jax.jit(obj_fun) if jittable else obj_fun
        self.init_x = x0
        if type(f2) == fs_palm.Box:
            self.variable_bounds.lower = f2.lower
            self.variable_bounds.upper = f2.upper
        if l1_lbda is not None:
            if type(l1_lbda) in [float, int]:
                self.l1_reg = [l1_lbda]
            elif type(l1_lbda) != list:
                raise ValueError(f"Invalid L1 regularization type: {type(l1_lbda)}; expected float, int, or list.")
            elif type(l1_lbda) == list and len(l1_lbda) != x0.shape[0]:
                raise ValueError(f"Invalid L1 regularization list length: {len(l1_lbda)}; expected {x0.shape[0]}.")
            else:
                self.l1_reg = l1_lbda
        elif type(f2) == fs_palm.L1Norm:
            self.l1_reg = [f2.λ]
        else:
            self.l1_reg = [0.0]
        self.pa_solver_opts = solver_opts
        self.pa_direction = direction
        self.pa_tol = tol
        self.pa_max_iter = max_iter
        if jittable:
            self.jit_grad_f = jax.jit(jax.grad(obj_fun))
        else:
            self.jit_grad_f = jax.grad(obj_fun)

    def eval_objective(self, x):
        return self.jit_loss(x)

    def eval_objective_gradient(self, x, grad_f):
        grad_f[:] = self.jit_grad_f(x)

def get_solver_run(f2=None, l1_lbda=None, solver_opts=None, direction=None, print_interval=0, jittable=True):
    """
    Internal function to get the solver run function for PALM inner trainer.
    """
    def solver_run(palm_obj_fun, x0, lbfgs_iters, lbfgs_tol):
        pa_prob = PaProblem(
            obj_fun=palm_obj_fun,
            x0=x0,
            f2=f2,
            l1_lbda=l1_lbda,
            solver_opts=solver_opts,
            tol=lbfgs_tol,
            max_iter=lbfgs_iters,
            direction=direction,
            jittable=jittable
        )
        if pa_prob.pa_direction is None:
            pa_prob.pa_direction = pa.LBFGSDirection({"memory": 20})
        if pa_prob.pa_solver_opts is None:
            pa_prob.pa_solver_opts = {
                    "print_interval": print_interval,
                    "max_iter": pa_prob.pa_max_iter,
                    "stop_crit": pa.ProjGradUnitNorm,
                    # "quadratic_upperbound_tolerance_factor": 1e-12,
                }
        pa_prob.pa_solver = pa.PANOCSolver(pa_prob.pa_solver_opts, pa_prob.pa_direction)
        cnt = pa.problem_with_counters(pa_prob)
        sol, stats = pa_prob.pa_solver(cnt.problem, {"tolerance": pa_prob.pa_tol}, pa_prob.init_x)
        # PaProblem implements eval_objective and eval_objective_gradient
        # separately, so PANOC never calls the *fused* evaluator and alpaqa's
        # `objective_and_gradient` counter stays 0 for every run -- reading it
        # made grad_evals a constant. The populated counters are the separate
        # ones; `objective_gradient` is the gradient count the plots label
        # "grad evals", and `objective` is reported alongside it since PANOC's
        # linesearch evaluates f without a gradient.
        return sol, {"obj_grad_evals": cnt.evaluations.objective_gradient,
                    "obj_evals": cnt.evaluations.objective,
                    "fp_res": stats['ε'], "obj_val": pa_prob.eval_objective(sol), "reg_val": stats['final_h'], "status": str(stats['status'])}
    
    return PALMInnerTrainer(solver_run)

def phase_I_optim(x0, h, g, f2, lbda0, mu0, alphas=(1.01, 2, 5, 10), gamma0="auto",
                  tol=1e-7, max_iter=300, max_iter_inner=100, inner_solver="PANOC",
                  verbose=True):
    """
    Solves the Phase I feasibility problem (Appendix "The phase I problem for
    initialization") to find an initial feasible point, via a single call to
    FS-P-ALM itself on the auxiliary problem.

    Two conditions matter for this to work as intended:
     (1) The starting point z^0=(x^0,s^0) must itself be feasible for the
         auxiliary problem's only constraint g(x)<=s, as FS-P-ALM requires a
         feasible start -- so s^0 = max(g(x^0)), not 0.
     (2) The stopping test must require real progress on the *original*
         h(x), g(x), not just satisfaction of the auxiliary g(x)<=s (which
         holds trivially from a feasible start without s, or h(x) folded
         into f1, ever shrinking). We therefore hand Solution.fs_palm() the
         true-residual callable `true_residual` below and it terminates on
         max(||h(x)||_inf, ||max(g(x),0)||_inf) directly -- the same
         quantity verified on return, and the definition of the feasible
         starting point this routine exists to produce.

    Initialisation is scale-aware rather than hard-coded. nu_0 uses Birgin &
    Martinez's rule against the *original* violation (see below), and gamma_0
    defaults to "auto", i.e. kappa/L_hat with L_hat a finite-difference
    estimate of the local curvature of the augmented Lagrangian. Fixed
    constants cannot work here: the right nu_0 has units [f]/[c^2] and the
    right gamma_0 has units [x^2]/[f], so both change with the scaling of the
    problem, and a value tuned on one instance silently fails on the next.

    max_iter_inner caps the inner solver's iterations per outer step; if it
    doesn't reach tau_k within that budget it simply returns its current
    (best) iterate -- a genuine descent method never returns a worse point,
    it just under-converges -- and the outer loop's own stopping test
    (which includes this residual) correctly keeps iterating rather than
    declaring false convergence. The caller (fs_palm.py) passes the same
    per-inner-solver value the main solve loop uses (1000 for PANOC, 5000
    for ProxGrad -- ProxGrad has no curvature information, so it needs a
    substantially larger budget to make comparable progress); the default
    here is only a fallback for calling this function directly.

    On alpha: penalty growth here is nu_k = nu0*phi(k+1) = nu0*(k+1)^alpha
    (FS-P-ALM fixes xi1=xi2=1, so the schedule is the *only* growth mechanism).
    alpha therefore sets the entire trajectory, and too large a value drives
    nu into a range where the inner subproblem is numerically hopeless: with
    the original alpha=20, nu reached ~1e16 within ~10 outer iterations, PANOC
    then reported a (scaled) converged residual without moving x at all, and
    every subsequent iteration was dead time.

    Crucially nu_0 and alpha trade off: measured on one randomised portfolio
    instance, nu_0=1e-3 converges at alpha=4 but fails at alpha=2 (too slow to
    reach a useful magnitude), while nu_0=1 converges at alpha=2 but fails at
    alpha=4 (too fast, ill-conditioned). No single pair is robust across
    problems, so this routine tries several alphas and returns the first that
    reaches the target, keeping the best iterate seen if none does. Unlike
    gamma and nu -- which the algorithm escalates on its own, making an
    external retry loop over them redundant -- alpha is never adapted during a
    run, so sweeping it reaches configurations no single run can.

    On nu0/rho0: deliberately left at the solver's small default. ALGENCAN's
    initialization rule (rho_1 = max{1e-8, min{10, 2|f(x0)|/||c(x0)||^2}}) was
    tried here and made things *worse* -- it is designed for safeguarded
    multiplicative growth (rho <- 10*rho, reactive and slow), where starting at
    the right scale matters. Under FS-P-ALM's open-loop schedule nu0 is not a
    starting scale but a multiplier on the whole trajectory, so scaling it up
    by ~2000x (the rule returns nu0=2 on the portfolio problem vs the 1e-3
    default) merely reaches the ill-conditioned regime ~2000x sooner: every
    alpha failed at nu0=2, whereas nu0=1e-3 converges for alpha in {2,4,6}.
    """
    x_dim = x0.shape[0]

    if g is not None and h is None:
        feas_f = lambda z: jnp.sum(z[-1]**2)
        feas_g = lambda z: g(z[:x_dim]) - z[-1]
    elif g is not None and h is not None:
        feas_f = lambda z: 0.5*(jnp.sum(h(z[:x_dim])**2) + jnp.sum(z[-1]**2))
        feas_g = lambda z: g(z[:x_dim]) - z[-1]
    elif h is not None and g is None:
        feas_f = lambda z: 0.5*jnp.sum(h(z)**2)
        feas_g = None

    feas_prob = fs_palm.Problem(
                    f1=feas_f,
                    g=[feas_g] if feas_g is not None else None,
                    h=None,
                    jittable=True
                )
    if g:
        s0 = jnp.max(g(x0))  # makes z^0=(x0,s0) feasible for g(x)<=s
        z0 = jnp.concatenate([x0, jnp.array([s0])])
    else:
        z0 = x0.copy()

    def orig_residuals(z):
        """(h,g) residuals of the *original* problem at the x-part of z."""
        x_c = z[:x_dim]
        h_res = jnp.linalg.norm(h(x_c), jnp.inf) if h is not None else 0.0
        g_res = jnp.linalg.norm(jnp.maximum(g(x_c), 0), jnp.inf) if g is not None else 0.0
        return h_res, g_res

    def true_residual(z):
        return jnp.maximum(*orig_residuals(z))

    # Scale-aware initial penalty (Birgin & Martinez). The auxiliary
    # constraint g(x)<=s is satisfied at z0 by construction, so its violation
    # is identically zero and cannot set the scale; the *original* violation
    # can, is nonzero exactly when Phase I is invoked, and is the quantity
    # Phase I exists to drive to zero.
    h0_sq = float(jnp.sum(h(x0) ** 2)) if h is not None else 0.0
    g0_sq = float(jnp.sum(jnp.maximum(g(x0), 0.0) ** 2)) if g is not None else 0.0
    nu0 = algencan_penalty0(float(feas_f(z0)), h0_sq + g0_sq)

    target = max(tol, 1e-5)
    best = (jnp.inf, None, None)

    # alpha is the sole driver of penalty growth (FS-P-ALM fixes xi=1) and,
    # unlike gamma and nu, it is *not* adapted during a run -- so trying a few
    # values reaches configurations no single run can. nu_0 and alpha trade
    # off against each other (a small nu_0 needs a larger alpha to reach a
    # useful magnitude in time, and vice versa), which is why one fixed pair
    # cannot be robust across problems.
    for a in alphas:
        prob_a = fs_palm.Problem(f1=feas_f, g=[feas_g] if feas_g is not None else None,
                                 h=None, jittable=True)
        try:
            res_a = fs_palm.solve(prob_a, z0, lbda0=lbda0, mu0=mu0, use_proximal=True,
                                  tol=tol, max_iter=max_iter, max_iter_inner=max_iter_inner,
                                  alpha=a, gamma0=gamma0, nu0=nu0, rho0=nu0,
                                  start_feas=False, inner_solver=inner_solver,
                                  verbosity=0, phase_I_residual=true_residual)
        except Exception as exc:                       # keep trying other alphas
            if verbose:
                print(f"  Phase I: alpha={a} raised {type(exc).__name__}: {exc}")
            continue
        h_res, g_res = orig_residuals(res_a.x)
        r = float(max(h_res, g_res))
        if r < best[0]:
            best = (r, res_a.x[:x_dim], a)
        if r <= target:
            if verbose:
                print(f"Phase I optimization successful (alpha={a}, "
                      f"{len(res_a.total_infeas)} iterations, residual {r:.2e}).")
            return res_a.x[:x_dim]
        if verbose:
            print(f"  Phase I: alpha={a} reached {r:.2e} > {target:.1e}, trying next")

    r, x_best, a_best = best
    if x_best is None:
        raise RuntimeError("Phase I optimization failed: every alpha in "
                           f"{list(alphas)} raised an exception.")
    raise RuntimeError(
        f"Phase I optimization failed to reach target accuracy {target:.1e}. "
        f"Best residual {r:.2e} (alpha={a_best}) over alphas {list(alphas)}. "
        f"nu0={nu0:.3e} (scale-aware), gamma0={gamma0!r}.")

# ---------------------------------------------------------------------------
# Proximal gradient inner solver
# ---------------------------------------------------------------------------
# PANOC is a quasi-Newton method: its L-BFGS directions absorb much of the
# ill-conditioning that a growing penalty induces in the subproblems, which is
# why classic ALM stays cheap even when nu reaches 1e5. A plain first-order
# method has an iteration complexity that depends directly on the condition
# number of the subproblem, so it is the honest instrument for testing whether
# a penalty schedule ill-conditions the inner problems -- the claim the
# paper's boundedness argument is about.
#
# Written here rather than called through a foreign-language solver package
# so that it runs on the same JAX oracles as PANOC and reports the SAME
# counters (obj_grad_evals / obj_evals); routing one solver through a
# language bridge and the other through JAX would confound the algorithmic
# comparison with a language-bridge overhead, exactly the objection raised
# about wall-clock comparisons across a cross-language bridge.

def _make_prox(f2=None, l1_lbda=None, n=None):
    """prox operator of the nonsmooth term, matching PaProblem's handling.

    Returns prox(v, gamma) = argmin_u { f2(u) + ||u-v||^2/(2 gamma) } and a
    callable giving f2(x), for the three regularisers the package supports.
    """
    lower = upper = None
    lam = None
    if isinstance(f2, fs_palm.Box):
        lower = jnp.asarray(f2.lower)
        upper = jnp.asarray(f2.upper)
    if l1_lbda is not None:
        lam = jnp.asarray(l1_lbda, dtype=float)
    elif isinstance(f2, fs_palm.L1Norm):
        lam = jnp.asarray(float(f2.λ))

    def prox(v, gamma):
        u = v
        if lam is not None:                       # soft-thresholding
            t = gamma * lam
            u = jnp.sign(u) * jnp.maximum(jnp.abs(u) - t, 0.0)
        if lower is not None:                     # projection onto the box
            u = jnp.clip(u, lower, upper)
        return u

    def reg(x):
        val = 0.0
        if lam is not None:
            val = val + jnp.sum(lam * jnp.abs(x))
        return val

    return prox, reg


def get_proxgrad_run(f2=None, l1_lbda=None, jittable=True, gamma_init=None,
                     backtrack=0.5, grow=1.1, gamma_min=1e-16,
                     print_interval=0):
    """Inner solver: proximal gradient with a backtracking linesearch.

        x^{j+1} = prox_{gamma f2}( x^j - gamma grad f(x^j) )

    The stepsize is accepted when the descent lemma holds on the SMOOTH part,

        f(x+) <= f(x) + <grad f(x), x+ - x> + ||x+ - x||^2 / (2 gamma),

    which is what guarantees monotone decrease of f + f2; otherwise gamma is
    halved and the trial repeated. Between iterations gamma is allowed to grow
    by `grow` so an early conservative estimate does not persist for the whole
    solve.

    THE STEPSIZE IS WARM STARTED ACROSS OUTER ITERATIONS. gamma lives in the
    closure and each subproblem resumes from the last accepted value times
    `grow`, rather than from a fresh estimate. This matters more than it looks:
    consecutive augmented-Lagrangian subproblems differ only through the
    multipliers and the penalty, so the previous stepsize is a much better
    starting guess than a two-point probe. Re-probing costs two gradient
    evaluations per outer iteration and, worse, restarts from a value that is
    systematically too LARGE, because the two-point estimate of L runs low
    (measured 1.4x-2.4x low on these problem families); every one of those
    over-estimates then has to be backtracked away again from scratch.

    The reported fixed-point residual is
        ||x - prox_{f2}(x - grad f(x))||_inf
    i.e. taken at UNIT stepsize. This is deliberate: it is the same quantity
    the outer loop's epsilon-KKT test uses (Definition 1) and the same as
    alpaqa's ProjGradUnitNorm, so tau_k means the same thing whichever inner
    solver is selected. Using the linesearch's own gamma here instead would
    make the inner tolerance silently stepsize-dependent.
    """
    prox, reg = _make_prox(f2, l1_lbda)
    # Persists across calls: one entry per solver instance, so two solvers
    # built for two different problems do not share a stepsize.
    warm = {"gamma": None}

    def solver_run(palm_obj_fun, x0, max_iter, fp_tol):
        f = jax.jit(palm_obj_fun) if jittable else palm_obj_fun
        fg = jax.jit(jax.value_and_grad(palm_obj_fun)) if jittable \
            else jax.value_and_grad(palm_obj_fun)

        x = jnp.asarray(x0)
        n_grad = 0
        n_obj = 0

        # Stepsize: warm start from the previous subproblem when there is one,
        # otherwise 1/L from a two-point probe (or the supplied value).
        if warm["gamma"] is not None:
            gamma = warm["gamma"] * grow
        elif gamma_init is None:
            f0, g0 = fg(x); n_grad += 1
            ng = float(jnp.linalg.norm(g0))
            d = g0 / ng if (np.isfinite(ng) and ng > 0) else jnp.ones_like(x)
            eps = 1e-6 * max(1.0, float(jnp.linalg.norm(x)))
            _, g1 = fg(x + eps * d); n_grad += 1
            L = float(jnp.linalg.norm(g1 - g0)) / eps
            gamma = 1.0 / L if (np.isfinite(L) and L > 0) else 1.0
        else:
            gamma = float(gamma_init)

        status = "MaxIter"
        fp = np.inf
        for it in range(int(max_iter)):
            fx, gx = fg(x); n_grad += 1
            if not (np.isfinite(float(fx))):
                status = "NaNOrInf"
                break

            # stopping test at unit stepsize (see the docstring)
            fp = float(jnp.max(jnp.abs(x - prox(x - gx, 1.0))))
            if fp <= fp_tol:
                status = "Converged"
                break

            # backtracking on the descent lemma
            while True:
                x_new = prox(x - gamma * gx, gamma)
                d = x_new - x
                lhs = float(f(x_new)); n_obj += 1
                rhs = float(fx + jnp.dot(gx, d) + jnp.dot(d, d) / (2.0 * gamma))
                if not np.isfinite(lhs):
                    pass                      # treat as a failed trial
                elif lhs <= rhs + 1e-14 * max(1.0, abs(float(fx))):
                    break
                gamma *= backtrack
                if gamma < gamma_min:
                    status = "StepsizeTooSmall"
                    break
            if status == "StepsizeTooSmall":
                break

            x = x_new
            gamma *= grow
            if print_interval and it % print_interval == 0:
                print(f"      [proxgrad] it={it} f={float(fx):.6e} "
                      f"fp={fp:.3e} gamma={gamma:.3e}", flush=True)

        # Hand the accepted stepsize to the next subproblem. Not stored if the
        # solve broke down, since the value that caused the breakdown is the
        # last thing the next subproblem should start from.
        if status in ("Converged", "MaxIter") and np.isfinite(gamma) and gamma > 0:
            warm["gamma"] = gamma

        return x, {"obj_grad_evals": n_grad, "obj_evals": n_obj,
                   "fp_res": fp, "obj_val": float(f(x)),
                   "reg_val": float(reg(x)), "status": status,
                   "gamma": gamma}

    return PALMInnerTrainer(solver_run)
