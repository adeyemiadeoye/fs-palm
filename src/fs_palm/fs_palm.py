import numpy as np
np.NaN = np.nan
import jax
from functools import partial
import jax.numpy as jnp
import alpaqa as pa
from .inner_solvers.inner_solvers import phase_I_optim, get_solver_run, get_proxgrad_run
from .utils.utils import params_flatten, update_penalties, GradEvalCounter, algencan_penalty0, rule1_penalty0, rule3_penalty0, lipschitz_estimate, spectral_norm_hess, spectral_norm_jtj, lipschitz_gamma0, proximal_init
import time

reg_classes = [pa.functions.L1Norm, pa.Box, pa.functions.NuclearNorm]

class Solution:
    """
    Holds the solution process and data for FS-P-ALM.
    """
    def __init__(self, problem, x0, inner_solve_runner, lbda0, mu0, rho0, nu0, gamma0, x_shapes=None, x_cumsizes=None, use_proximal=False, beta=0.5, alpha=2.0, alpha_gamma=1.01, delta="auto", xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7, start_feas=True, inner_solver="PANOC", max_iter_inner=None, pa_solver_opts=None, pa_direction=None, verbosity=1, max_runtime=24.0, phi_strategy="pow", feas_reset_interval=None, no_reset=False, adaptive_fp_tol=True, tau0=0.1, kappa_tol=0.5, penalty_horizon=30, penalty_ramp=5.0, penalty_kappa=1e2, delta_rule="objective", gamma_kappa=1e3, penalty_max=1e20, phase_I_residual=None):
        self.problem = problem
        self.x0 = x0
        self.x_shapes = x_shapes
        self.x_cumsizes = x_cumsizes
        self.is_not_flat_x = x_shapes is not None and x_cumsizes is not None
        self.lbda0 = lbda0
        self.mu0 = mu0
        self.rho0 = rho0
        self.nu0 = nu0
        self.beta = beta
        self.alpha = alpha
        self.alpha_gamma = alpha_gamma
        # Validate here rather than let a bad value travel. A string other than
        # "auto" (e.g. passing the delta_rule name "objective" to delta by
        # mistake) survives __init__ untouched and only fails ~500 lines later
        # inside the outer loop, as
        #   TypeError: Only integer scalar arrays can be converted to a scalar
        #              index
        # from `self.delta*(self.x0 - x)**2` -- an error that names neither
        # delta nor the argument that was wrong.
        if isinstance(delta, str):
            if delta != "auto":
                raise ValueError(
                    f"delta must be a positive float or 'auto', got {delta!r}. "
                    f"Did you mean delta_rule={delta!r}?")
        elif not (float(delta) > 0.0):
            raise ValueError(f"delta must be > 0, got {delta!r}")
        if delta_rule not in ("objective", "displacement"):
            raise ValueError(f"delta_rule must be 'objective' or "
                             f"'displacement', got {delta_rule!r}")
        self.delta = delta
        self.xi1 = xi1
        self.xi2 = xi2
        self.tol = tol
        self.fp_tol = fp_tol if fp_tol is not None else tol
        self.adaptive_fp_tol = adaptive_fp_tol
        self.tau0 = tau0            # tau_k = max(tau0*kappa_tol^k, tol)
        self.kappa_tol = kappa_tol
        self.penalty_horizon = penalty_horizon   # expected outer iterations, NOT max_iter (rule1)
        self.penalty_ramp = penalty_ramp         # short ramp to the target scale (rule3)
        self.penalty_kappa = penalty_kappa       # dimensionless multiplier in rule3
        # Upper bound on rho and nu, applied to EVERY variant so the comparison
        # stays symmetric (the fixed-schedule variants never approach it, so it
        # is a no-op for them). Uncapped, the multiplicative update drives the
        # penalty to ~1e290 on the harder instances, where the augmented
        # Lagrangian rho*||h||^2 retains no usable precision in double and the
        # run is reporting a numerical breakdown rather than an algorithmic
        # one. Beyond ~1e16 the subproblem Hessian grad^2 f + rho J'J puts the
        # objective's O(1) curvature below the rounding error of a double, so
        # anything "converging" there is the penalty crushing the problem while
        # the objective is numerically invisible. The cap is still generous:
        # reaching ||h|| ~ 1e-6 by penalty alone needs rho ~ 1e14 on these
        # problems, so 1e20 leaves six orders of real headroom.
        #
        # This is a numerical guard of ours, NOT standard practice -- do not
        # cite it as such. Classic ALM implementations typically do not cap the
        # penalty at all; they instead clip the MULTIPLIERS, i.e. the
        # safeguarding mechanism this method replaces with the feasible-start
        # argument. Measured, the value does not matter here anyway: on basis
        # pursuit, caps of 1e20 and 1e300 gave identical cost (113886 gradient
        # evaluations) and the same outcome.
        self.penalty_max = float(penalty_max)
        self.penalty_capped_at = None
        self.delta_rule = delta_rule
        # gamma_0 = gamma_kappa / L_hat when gamma0 == "auto". gamma is the
        # proximal REGULARISATION, not a stepsize: the term (1/2gamma)||x-xhat||^2
        # adds 1/gamma to the curvature, so gamma_kappa = 1 (i.e. gamma_0 = 1/L)
        # makes the proximal term as strong as the problem's own curvature and
        # over-damps the iteration. gamma_kappa >> 1 keeps it a mild stabiliser.
        self.gamma_kappa = gamma_kappa
        # Resolved per inner solver, because the right cap differs by an order
        # of magnitude. A proximal-gradient subproblem needs many more cheap
        # iterations than a quasi-Newton one, so 1000 truncates it routinely;
        # but handing PANOC the same 5000 makes it grind on subproblems it
        # would otherwise abandon, which measured 4x WORSE on basis pursuit
        # (FS-ALM: 1351 -> 5375 gradient evaluations). Pass an explicit value
        # to override either default.
        if max_iter_inner is None:
            max_iter_inner = (5000 if str(inner_solver).upper()
                              in ("PROXGRAD", "PROXIMALGRADIENT", "PG") else 1000)
        self.max_iter_inner = max_iter_inner
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.x = x0
        self.start_feas = start_feas
        self.inner_solver = inner_solver
        self.phase_I_tol = phase_I_tol
        self.fp_res = []
        self.kkt_res = []
        self.total_infeas = []
        self.f_hist = []
        self.rho_hist = []
        self.nu_hist = []
        self.gamma_hist = []
        self.prox_hist = []
        self.solve_status = None
        self.pa_direction = pa_direction
        self.pa_solver_opts = pa_solver_opts
        self.use_proximal = use_proximal
        self.gamma0 = gamma0          # may be "auto"; resolved below
        self.gamma_k = None
        self.phi_strategy = phi_strategy
        self.max_runtime = max_runtime * 3600 if max_runtime is not None else 24.0 * 3600
        self.total_runtime = None
        self.solve_runtime = None
        self.feas_reset_interval = feas_reset_interval
        self.reset_x0 = x0
        self.no_reset = no_reset
        self.alm_grad_fn = None
        self.inner_solve_runner = inner_solve_runner
        if self.inner_solve_runner is None:
            if self.problem.h is None and self.problem.g is None:
                print_interval = 5*self.verbosity
            else:
                print_interval = 0
            if str(self.inner_solver).upper() in ("PROXGRAD", "PROXIMALGRADIENT", "PG"):
                # plain first-order inner solver: no quasi-Newton direction, so
                # the cost of a subproblem reflects its conditioning directly
                self.inner_solve_runner = get_proxgrad_run(f2=self.problem.f2, l1_lbda=self.problem.f2_lbda, jittable=self.problem.jittable, print_interval=print_interval)
            else:
                self.inner_solve_runner = get_solver_run(f2=self.problem.f2, l1_lbda=self.problem.f2_lbda, solver_opts=pa_solver_opts, direction=pa_direction, print_interval=print_interval, jittable=self.problem.jittable)
        else:
            self.inner_solve_runner = inner_solve_runner
        # rho0/nu0/gamma0 may be the string "auto"; resolve in this order,
        # since the gamma_0 estimate needs an augmented Lagrangian, which in
        # turn needs the penalties.
        self._resolve_auto_penalties()
        if self.problem.h is not None:
            h0 = self.problem.h(self.x0)
            self.rho_vec = jnp.ones_like(h0) * self.rho0
        else:
            self.rho_vec = None
        if self.problem.g is not None:
            g0 = self.problem.g(self.x0)
            self.nu_vec = jnp.ones_like(g0) * self.nu0
        else:
            self.nu_vec = None
        self._resolve_auto_gamma(lbda0, mu0)
        self.gamma_k = self.gamma0 if use_proximal else jnp.nan

        # Phase I is signalled by supplying the true-residual callable itself:
        # when set, the stopping test uses it instead of this (auxiliary)
        # problem's own infeasibility. See the stopping test in fs_palm().
        self.phase_I_residual = phase_I_residual
        self.is_phase_I = phase_I_residual is not None
        # E^k baseline for the nu-update test (state:proxALM:nu) must be evaluated
        # with (mu^{k-1}, nu_{k-1}), i.e. the E computed at the *previous* iteration --
        # cached here rather than recomputed with the just-updated (mu, nu).
        self._prev_E_cache = None
        # inner solves that returned without reaching tau_k (see the main loop)
        self.n_inner_incomplete = 0
        # per-iteration outcome of the beta-progress test, which is what
        # actually triggers a penalty increase (see the main loop)
        self.beta_test_fired = []


    def _resolve_auto_penalties(self):
        """Resolve a symbolic rho0/nu0 into a number.

        Named strategies, so that alternatives can be compared rather than one
        being hard-coded:

          "rule3" (DEFAULT)   penalty_kappa*L_f/||J_c||^2/(ramp+1)^alpha --
                              curvature matched, with the dimensionless
                              constant calibrated across all three problem
                              families this method was tested on, including
                              for classic ALM (phi_strategy="linear"): rule3
                              detects phi==0 and sets growth=1 so ALM starts
                              AT the target rather than below it (see
                              rule3_penalty0's docstring for why dividing
                              anyway manufactures a runaway). Measured against
                              rule1 on classic ALM, basis pursuit, PANOC: same
                              solution, 385 vs 2941 gradient evaluations.
          "rule1"             rho_base/(horizon+1)^alpha -- value-based, no
                              curvature estimate. Used internally as the
                              fallback when rule3's curvature estimate is
                              unusable (linear objective, degenerate
                              Jacobian); also selectable directly. See
                              rule1_penalty0.
          "auto"              Birgin & Martinez's rho = 2|f(x0)|/||c(x0)||^2,
                              clipped to [1e-8, 10]. A literature baseline,
                              not tuned for this method; note the clip binds
                              on most well-scaled problems, in which case this
                              simply returns 10.

        A float is of course still accepted and used verbatim.
        """
        strategies = ("rule1", "rule3", "auto")
        # A name missing from this tuple is not a no-op: the string survives
        # unresolved and only fails later as `jnp.ones_like(g0) * "rule3"`,
        # which numpy reads as string repetition and reports as
        #   TypeError: Only integer scalar arrays can be converted to a scalar
        #              index
        # -- naming neither the penalty nor the strategy. Reject unknown names
        # here instead, while the argument is still in hand.
        for name, spec in (("rho0", self.rho0), ("nu0", self.nu0)):
            if isinstance(spec, str) and spec not in strategies:
                raise ValueError(f"{name}={spec!r} is not a known strategy; "
                                 f"expected a float or one of {strategies}")
        if self.rho0 not in strategies and self.nu0 not in strategies:
            return
        f0 = float(self.problem.f1(self.x0))
        viol_sq = 0.0
        if self.problem.h is not None:
            viol_sq += float(jnp.sum(self.problem.h(self.x0) ** 2))
        if self.problem.g is not None:
            viol_sq += float(jnp.sum(jnp.maximum(self.problem.g(self.x0), 0.0) ** 2))

        def curvature_ratio():
            """(L of grad f1, ||J_c(x0)||^2) for rule3.

            Both are curvatures, so the ratio is the penalty at which the
            constraints are felt as strongly as the objective -- unlike the
            value-based rules, this stays informative at a feasible x0, where
            ||c(x0)|| = 0.
            """
            gf1 = jax.grad(lambda x: self.problem.f1(x))
            L_obj = spectral_norm_hess(gf1, self.x0) or lipschitz_estimate(gf1, self.x0)
            # ||J_c||^2 via power iteration on J'(Jv): one jvp and one vjp per
            # iteration, so J is never assembled (forming it costs m gradient
            # evaluations). NOT the Hessian of 1/2||c||^2 -- that is
            # J'J + sum_i c_i grad^2 c_i, and the second term vanishes only
            # where c(x0)=0. It does for equalities at a feasible point, but
            # NOT for inequalities, where feasibility means g(x0)<0 strictly:
            # measured on a QCQP started at x0=0 the contamination is exactly
            # -sum_i r_i P_i and shifts the estimate by 16-23%.
            # PER BLOCK, not summed. rho penalises h and nu penalises g, and
            # the two are independent parameters: rescaling the equalities
            # alone should move rho_0 by s^-2 and leave nu_0 alone. Summing
            # ||J_h||^2 + ||J_g||^2 into one number makes the rule invariant
            # only under a COMMON rescaling of both blocks, and ties rho_0 to
            # constraints it does not penalise. On the three families here
            # exactly one block is present in each (QCQP and the convex QPs
            # have inequalities only, basis pursuit equalities only), so this
            # reproduces every previously measured number; it differs only on
            # problems with both.
            def block(cf):
                if cf is None:
                    return None
                val = spectral_norm_jtj(cf, self.x0)
                if val is None:
                    J = jnp.atleast_2d(jnp.asarray(jax.jacfwd(cf)(self.x0)))
                    val = float(jnp.linalg.norm(J, 2)) ** 2
                return val
            return L_obj, block(self.problem.h), block(self.problem.g)

        cached = {}

        def resolve(spec, block):
            """`block` is "h" for rho_0 and "g" for nu_0.

            The curvature rules use that block's own Jacobian, so rho_0 is set
            by the equalities and nu_0 by the inequalities. When a problem has
            only one of the two, the other parameter is never read; falling
            back to the other block's norm would only invent a number.
            """
            if spec == "rule1":
                return rule1_penalty0(f0, viol_sq, self.alpha,
                                       horizon=self.penalty_horizon)
            if spec == "rule3":
                if "cr" not in cached:
                    cached["cr"] = curvature_ratio()
                L_obj, Jh_sq, Jg_sq = cached["cr"]
                J_sq = Jh_sq if block == "h" else Jg_sq
                if J_sq is None:
                    # this block is absent, so the parameter is unused; keep
                    # the other block's scale rather than a bare default, so a
                    # caller that inspects it still sees something sensible
                    J_sq = Jg_sq if block == "h" else Jh_sq
                if L_obj is None or J_sq is None or J_sq <= 0.0:
                    # no usable curvature (e.g. a linear objective): fall back
                    # rather than return a meaningless number
                    return rule1_penalty0(f0, viol_sq, self.alpha,
                                          horizon=self.penalty_horizon)
                # Divide by the growth THIS variant actually has, so that every
                # variant reaches kappa*nu_star after `ramp` outer iterations.
                # _get_phi_i returns 0 for phi_strategy="linear" (classical
                # ALM), where the penalty grows only via the xi-multiplication
                # triggered by a failed beta-test -- so there is no schedule to
                # ramp up and nu_0 must start AT the target, not below it. See
                # rule3_penalty0 for why dividing anyway manufactures a runaway.
                phi_ramp = float(self._get_phi_i(self.penalty_ramp + 1.0))
                return rule3_penalty0(L_obj, J_sq, self.alpha,
                                      kappa=self.penalty_kappa,
                                      ramp=self.penalty_ramp,
                                      growth=phi_ramp if phi_ramp > 0.0 else 1.0)
            if spec == "auto":
                return algencan_penalty0(f0, viol_sq)
            return spec

        self.rho0 = resolve(self.rho0, "h")
        self.nu0 = resolve(self.nu0, "g")

    def _resolve_auto_gamma(self, lbda0, mu0):
        """Resolve gamma0 and/or delta == "auto" via proximal_init.

        The two are set together because they are the two branches of the same
        gamma_k update; see proximal_init for the rules and their units.

        Passing x_prev = x makes the proximal term identically zero, so
        _get_prox_L_aug below is the augmented Lagrangian *without* the prox
        term for any gamma -- exactly the curvature gamma_0 should match.
        """
        if self.gamma0 != "auto" and self.delta != "auto":
            return
        if not self.use_proximal:
            if self.gamma0 == "auto":
                self.gamma0 = 1e-1
            if self.delta == "auto":
                self.delta = 1.0
            return

        def al(x):
            return self._get_prox_L_aug(x, x, lbda0, mu0,
                                        self.rho_vec, self.nu_vec, 1.0)

        g0, d0 = proximal_init(jax.grad(al), self.x0,
                               f0=float(self.problem.f1(self.x0)),
                               kappa=self.gamma_kappa,
                               delta_rule=self.delta_rule)
        if self.gamma0 == "auto":
            self.gamma0 = g0
        if self.delta == "auto":
            self.delta = d0

    def _get_phi_i(self, i):
        if self.phi_strategy == "log":
            return i*jnp.log(i)
        elif self.phi_strategy == "pow":
            return i**self.alpha
        elif self.phi_strategy == "linear":
            return 0
        else:
            raise ValueError(f"Unknown phi strategy: {self.phi_strategy}")

    def _get_phi_gamma_i(self, i):
        # separate growth function phi_gamma for the proximal parameter gamma_k
        # (Assumption on growth functions in the paper): keep alpha_gamma close
        # to 1 so gamma_k grows much more slowly than rho_k, nu_k
        if self.phi_strategy == "log":
            return i*jnp.log(i)
        elif self.phi_strategy == "pow":
            return i**self.alpha_gamma
        elif self.phi_strategy == "linear":
            return 0
        else:
            raise ValueError(f"Unknown phi strategy: {self.phi_strategy}")

    def _is_feasible(self, x):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])
        h_feas = (self.problem.h is None or jnp.all(jnp.isclose(h_x, 0, atol=self.tol, rtol=self.tol)))
        g_feas = (self.problem.g is None or jnp.all(g_x <= self.tol))
        return (h_feas and g_feas)
    
    def _eval_prox_term(self, x, x_prev, gamma_k):
        if self.use_proximal:
            return jnp.sum((1/(2*gamma_k))*(x - x_prev)**2)
        else:
            return 0.0

    def fs_palm(self):
        total_start_time = time.time()
        start_time = time.time()
        warmup_end_time = None
        self.grad_evals = []
        last_grad_eval = 0
        if self.problem.h is None and self.problem.g is None:
            if self.verbosity > 0:
                print("Solving problem without constraints")
            self.use_proximal = False
            @jax.jit
            def obj_fun(x):
                return self._get_palm_obj_fun(x, x, self.lbda0, self.mu0, self.rho0, self.nu0, self.gamma_k)
            x = self.x0.copy()
            x_new, state = self.inner_solve_runner.train_fun(obj_fun, x, self.max_iter_inner, self.tol)
            if self.is_not_flat_x:
                x_new = params_flatten(x_new)

            self.x = x_new
            self.f_hist.append(self.problem.f1(self.x))
            self.total_runtime = time.time() - total_start_time
            self.solve_runtime = self.total_runtime
            if self.verbosity > 0:
                print(f"Unconstrained optimization completed after {self.solve_runtime:.6f} seconds.")
                print(f"{'Objective value:':<25} {self.f_hist[-1]:.6e}")
            # status is state, not output -- see the note in the main loop
            self.solve_status = state['status']
            if self.solve_status.startswith("SolverStatus."): # match fs_palm style
                self.solve_status = self.solve_status[len("SolverStatus."):]
            return

        if self._is_feasible(self.x0):
            if self.verbosity > 0:
                print("Initial point is feasible.")
            x = self.x0.copy()
            self.start_feas = False
            warmup_end_time = time.time()
        else:
            if self.start_feas:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Finding a feasible point...")
                # Phase I finds a feasible x^0 via its own auxiliary problem; it is
                # initialisation infrastructure, not part of what inner_solver
                # comparisons (PANOC vs ProxGrad) are about, so it always uses
                # PANOC regardless of self.inner_solver. Observed directly: every
                # Phase I call tonight that used PANOC succeeded quickly; the one
                # case forced onto ProxGrad (DUALC1, kappa~1.1e6) failed outright
                # at every alpha tried -- the same ill-conditioning sensitivity
                # ProxGrad shows throughout, just hitting the initialisation step
                # instead of the main solve it's supposed to be isolated to.
                self.x0 = phase_I_optim(self.x0, self.problem.h, self.problem.g, self.problem.f2, self.lbda0, self.mu0, tol=self.phase_I_tol, inner_solver="PANOC", max_iter_inner=self.max_iter_inner)
                self.reset_x0 = self.x0.copy()
                warmup_end_time = time.time()
            else:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Starting from infeasible point.")
                warmup_end_time = time.time()
            x = self.x0.copy()
        x_prev = x.copy()
        lbda = self.lbda0
        mu = self.mu0
        rho_vec = self.rho_vec.copy() if self.rho_vec is not None else None
        nu_vec = self.nu_vec.copy() if self.nu_vec is not None else None

        if self.verbosity > 0:
            print(f"{'iter':<5} | {'f':<10} | {'p. term':<10} | {'total infeas':<10} | {'rho':<10} | {'nu':<10} | {'gamma':<10}")
            print("-" * 90)

        prox_term_i = jnp.nan
        solve_start_time = warmup_end_time if warmup_end_time is not None else total_start_time
        
        grad_evals = 0
        if self.use_proximal:
            @jax.jit
            def L_aug(x):
                return self._get_prox_L_aug(x, x_prev, lbda, mu, rho_vec, nu_vec, self.gamma_k)
        else:
            @jax.jit
            def L_aug(x):
                return self._get_prox_L_aug(x, None, lbda, mu, rho_vec, nu_vec, self.gamma_k)
            
        self.alm_grad_fn = GradEvalCounter(jax.grad(L_aug)) if not self.problem.jittable else GradEvalCounter(jax.jit(jax.grad(L_aug)))

        ####### main outer ALM loop #######
        for i in range(self.max_iter):
            fp_tol = self.fp_tol
            phi_i = self._get_phi_i(i+1)
            if self.max_runtime is not None and (time.time() - start_time) > self.max_runtime:
                if self.verbosity > 0:
                    print("-" * 80)
                    print(f"Maximum runtime of {self.max_runtime:.2e} seconds reached. Exiting loop.")
                self.solve_status = "MaxRuntimeExceeded"
                break
            if self.problem.callback is not None:
                self.problem.callback(
                    iter=i,
                    x=x,
                    x_prev=x_prev,
                    lbda=lbda,
                    mu=mu,
                    rho=rho_vec,
                    nu=nu_vec,
                    gamma_k=self.gamma_k,
                    x0=self.x0
                )
            if i == 0:
                self.grad_evals.append(0)
                h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
                g_x = self.problem.g(x) if self.problem.g else jnp.array([0])
                nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
                nrm_E = 0
                if self.problem.g:
                    E_x = jnp.minimum(-g_x, 1/nu_vec*mu)
                    nrm_E = jnp.linalg.norm(E_x, jnp.inf)
                    self._prev_E_cache = E_x
                L0_grad = self.alm_grad_fn(x)
                if self.grad_evals is not None:
                    grad_evals += self.alm_grad_fn.count
                if type(self.problem.f2) in reg_classes:
                    p_z = pa.prox(self.problem.f2, x - L0_grad)[1]
                    if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                        p_z = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), p_z - L0_grad)[1]
                    fp_res_i = jnp.linalg.norm(x - p_z, jnp.inf)
                elif type(self.problem.f2) == type(None): # this won't be the case though (case handled above)
                    fp_res_i = jnp.linalg.norm(L0_grad, jnp.inf)
                stopping_terms = jnp.array([fp_res_i, nrm_h, nrm_E])
                # include fp_res, as in the main loop below, so that kkt_res[0]
                # measures the same thing as every later entry
                eps_kkt_res = jnp.max(stopping_terms)
                f_x = self.problem.f1(x)
                self.kkt_res.append(eps_kkt_res)
                self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
                self.fp_res.append(stopping_terms[0])
                self.f_hist.append(f_x)
                self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
                self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
                self.gamma_hist.append(self.gamma_k)
                self.prox_hist.append(prox_term_i)
                if self.verbosity > 0:
                    print(
                        f"{i:<5} | {f_x:<10.4e} | {'-':^10} | {self.total_infeas[-1]:<10.4e} | {f"{jnp.max(rho_vec):<10.4e}" if rho_vec is not None else f"{'-':^10}"} | {f"{jnp.max(nu_vec):<10.4e}" if nu_vec is not None else f"{'-':^10}"} | {f"{self.gamma_k:<10.4e}" if self.use_proximal else f"{'-':^10}"}")

                # Step 1: to reset xhat?
                L_aug_val = L_aug(x)
                if self.no_reset:
                    x_hat = x
                else:
                    f1_reset_x0 = self.problem.f1(self.reset_x0)
                    if self.problem.f2:
                        pz_val = pa.prox(self.problem.f2, x)[0]
                        L_aug_val += pz_val
                        if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                            pz_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), x)[0]
                            L_aug_val += pz_val
                        pz_reset_val = pa.prox(self.problem.f2, self.reset_x0)[0]
                        f1_reset_x0 += pz_reset_val
                        if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                            pz_reset_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), self.reset_x0)[0]
                            f1_reset_x0 += pz_reset_val
                    
                    prox_term = self._eval_prox_term(self.reset_x0, x, self.gamma_k)
                    L_func_val = L_aug_val + prox_term
                    if L_func_val > f1_reset_x0 + prox_term:
                        x_hat = self.reset_x0
                    else:
                        x_hat = x

            if self.adaptive_fp_tol:
                if callable(self.fp_tol):
                    fp_tol = self.fp_tol(i)
                else:
                    # tau_k = max(tau0 * kappa^k, tol): geometric decay that
                    # SATURATES at the requested accuracy.
                    #
                    # The previous schedule, tau_k = 0.1/(k+1)^1.1, decays far
                    # too slowly to be useful: it is still ~1e-4 after 500 outer
                    # iterations, and reaching 1e-8 would take ~2.3e6 of them. So
                    # the subproblems were never solved to anything near the
                    # requested tolerance, which capped the attainable accuracy
                    # regardless of `tol` (measured: the objective plateaued at
                    # ~1.5e-6 whether tol was 1e-6 or 1e-8).
                    #
                    # Saturating at tol is also the better match to the theory:
                    # the analysis assumes tau_k -> tau in R_+ and delivers an
                    # eps-KKT point for any eps > tau, so tau = tol says exactly
                    # "you asked for tol, you get eps-KKT just above tol". The
                    # old schedule implied tau = 0, promising eps-KKT for every
                    # eps > 0 -- a promise no finite run keeps.
                    fp_tol = max(self.tau0 * self.kappa_tol**i, self.tol)

            def palm_obj_fun(x):
                return self._get_palm_obj_fun(x, x_prev, lbda, mu, rho_vec, nu_vec, self.gamma_k)
            
            x_new, state = self.inner_solve_runner.train_fun(palm_obj_fun, x_hat, self.max_iter_inner, fp_tol)
            if self.is_not_flat_x:
                x_new = params_flatten(x_new)

            isnanfpres = False if self.problem.f2 is None else (jnp.isnan(fp_res_i) or jnp.isinf(fp_res_i))
            isnannu = False if self.problem.g is None else (jnp.isnan(nu_vec).any() or jnp.isinf(nu_vec).any())
            isnanrho = False if self.problem.h is None else (jnp.isnan(rho_vec).any() or jnp.isinf(rho_vec).any())
            isnangamma = False if self.use_proximal is False else (jnp.isnan(self.gamma_k) or jnp.isinf(self.gamma_k))
            if jnp.isnan(self.problem.f1(x_new)) or jnp.isinf(self.problem.f1(x_new)) or isnanfpres or isnannu or isnanrho or isnangamma:
                if self.verbosity > 0:
                    stop_eps = self.total_infeas[-1]
                    print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {stop_eps:<10.4e} | {f"{jnp.max(rho_vec):<10.4e}" if rho_vec is not None else f"{'-':^10}"} | {f"{jnp.max(nu_vec):<10.4e}" if nu_vec is not None else f"{'-':^10}"} | {f"{self.gamma_k:<10.4e}" if self.use_proximal else f"{'-':^10}"}")
                    print("-" * 90)
                    print("One or more functions returned NaN or Inf. Stopping optimization.")
                    print(f"{'f1 value:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {f"{prox_term_i:.6e}" if self.use_proximal else f"{'-':.6}"}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {stop_eps:.6e}")
                    print(f"{'rho:':<25} {f"{jnp.max(rho_vec):.6e}" if rho_vec is not None else f"{'-':.6}"}")
                    print(f"{'nu:':<25} {f"{jnp.max(nu_vec):.6e}" if nu_vec is not None else f"{'-':.6}"}")
                    print(f"{'gamma:':<25} {f"{self.gamma_k:.6e}" if self.use_proximal else f"{'-':.6}"}")
                self.solve_status = "NaNOrInf"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )
                break

            h_x = self.problem.h(x_new) if self.problem.h else jnp.array([0])
            g_x = self.problem.g(x_new) if self.problem.g else jnp.array([0])

            if self.use_proximal:
                x_prev = x.copy()

            lbda = lbda + jnp.multiply(rho_vec, h_x) if self.problem.h else lbda
            mu_new = jnp.maximum(0, mu + jnp.multiply(nu_vec, g_x)) if self.problem.g else mu

            nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
            prev_h = self.problem.h(x) if self.problem.h else jnp.array([0])

            # E^{k+1} = min{-g(x^{k+1}), mu^k/nu_k}, using the pre-update (mu, nu_vec)
            E_x = jnp.minimum(-g_x, jnp.divide(1, nu_vec)*mu) if self.problem.g else jnp.array([0])
            nrm_E = jnp.linalg.norm(E_x, jnp.inf)

            # E^k baseline for the beta-test (state:proxALM:nu) must use (mu^{k-1},
            # nu_{k-1}) -- the cached E from the previous iteration, not a recompute
            # with the just-updated (mu, nu_vec) (which would use mu^k, nu_k instead).
            prev_E = self._prev_E_cache if self._prev_E_cache is not None else E_x

            # At k=0, x^0 is feasible so h(x^0)=0 and E^0_i=0 for any active g_i:
            # the beta-test baseline is degenerate and would force a spurious penalty
            # increase regardless of how good x^1 is. Skip the test for k=0 only;
            # rho_1=rho_0, nu_1=nu_0 in that case.
            # Record whether the beta-progress test FIRED this iteration. This
            # is the event that drives the penalty update, and it is not the
            # same as an inner solve missing tau_k: a subproblem can be solved
            # inexactly and still reduce the constraint residual by the factor
            # beta, in which case the penalty does not move at all. Conflating
            # the two misreads the mechanism entirely, so the trigger is now
            # recorded directly rather than inferred from the penalty history
            # (which cannot distinguish "test passed" from "test fired but the
            # schedule floor was already below the current penalty").
            _fired = False
            if i > 0:
                if self.problem.h is not None and prev_h is not None:
                    _fired |= bool(jnp.linalg.norm(h_x, jnp.inf)
                                   > self.beta * jnp.linalg.norm(prev_h, jnp.inf))
                if self.problem.g is not None and prev_E is not None:
                    _fired |= bool(jnp.linalg.norm(E_x, jnp.inf)
                                   > self.beta * jnp.linalg.norm(prev_E, jnp.inf))
            self.beta_test_fired.append(_fired)
            rho_vec, nu_vec = update_penalties(self.problem.lbda_sizes, self.problem.mu_sizes, rho_vec, nu_vec, self.rho0, self.nu0, E_x, prev_E, h_x, prev_h, self.beta, self.xi1, self.xi2, phi_i, skip_test=(i == 0))
            if self.penalty_max is not None and np.isfinite(self.penalty_max):
                _hit = False
                if rho_vec is not None:
                    _hit |= bool(jnp.max(rho_vec) > self.penalty_max)
                    rho_vec = jnp.minimum(rho_vec, self.penalty_max)
                if nu_vec is not None:
                    _hit |= bool(jnp.max(nu_vec) > self.penalty_max)
                    nu_vec = jnp.minimum(nu_vec, self.penalty_max)
                if _hit and self.penalty_capped_at is None:
                    self.penalty_capped_at = i
                    if self.verbosity > 0:
                        print(f"  penalty reached the cap {self.penalty_max:.1e} "
                              f"at iteration {i}; held there for the rest of the run")

            self._prev_E_cache = E_x

            x = x_new
            mu = mu_new

            if self.use_proximal:
                prox_term_i = jnp.sum(1/(2*self.gamma_k)*(x - x_prev)**2)
                # residual used by the stopping test, matching the (1/gamma_k)||x^{k+1}-x^k||
                # quantity shown (thm:prim-res) to vanish -- not the squared/2*gamma
                # objective contribution prox_term_i, which is on a different scale.
                prim_res_i = (1/self.gamma_k)*jnp.linalg.norm(x - x_prev, jnp.inf)
            else:
                prox_term_i = 0.0
                prim_res_i = 0.0

            fp_res_i = state["fp_res"]
            # The inner solver may return without reaching tau_k (it exhausts
            # max_iter_inner and hands back its best iterate). That is NOT an
            # error and must not abort the run: the outer loop carries on and
            # the next subproblem, with updated multipliers, is usually solved
            # to tolerance -- fp_res is observed to oscillate rather than to
            # stall. It does mean this iterate carries no eps-KKT certificate,
            # so it is counted and warned about instead of passing silently.
            if fp_res_i > fp_tol:
                self.n_inner_incomplete += 1
                if self.verbosity > 0 and self.n_inner_incomplete == 1:
                    print(f"  [warning] inner solver returned fp_res={fp_res_i:.2e} "
                          f"> tau_k={fp_tol:.2e} at iteration {i}; continuing "
                          f"(further occurrences counted, reported at the end)")
            stopping_terms = jnp.array([fp_res_i, nrm_h, nrm_E])
            if self.use_proximal:
                stopping_terms = jnp.concatenate([stopping_terms, jnp.array([prim_res_i])])

            # fp_res is INCLUDED: the theory bounds the true stationarity gap by
            # the inner tolerance tau_k plus a vanishing term, so a residual that
            # omits the inner inexactness understates the gap -- previously this
            # took max over stopping_terms[1:], silently dropping fp_res. With
            # the saturating tau_k schedule the inner solves actually reach `tol`,
            # so requiring fp_res <= tol as well costs little and makes
            # eps_kkt_res <= tol a genuine eps-KKT certificate: every term
            # (||h||, ||E||, the primal residual, and the inner residual) is then
            # below tol, rather than all but one of them.
            eps_kkt_res = jnp.max(stopping_terms)
            f_x = self.problem.f1(x)
            self.kkt_res.append(eps_kkt_res)
            self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
            self.f_hist.append(f_x)
            self.fp_res.append(stopping_terms[0])
            self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
            self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
            self.gamma_hist.append(self.gamma_k)
            self.prox_hist.append(prox_term_i)
            # counted here, alongside the other per-iteration histories and
            # *before* the stopping test below: the inner solve for this
            # iteration has already happened, so terminating now must not
            # discard its evaluations (it previously did, which is why the
            # list had to be back-filled with a stale value at the end).
            grad_evals += state["obj_grad_evals"]
            self.grad_evals.append(grad_evals)
            last_grad_eval = grad_evals

            if self.use_proximal:
                self.gamma_k = jnp.maximum(jnp.sum(self.delta*(self.x0 - x)**2), self.gamma0*self._get_phi_gamma_i(i+1))

            # Phase I terminates on the true residual of the *original* problem,
            # max(||h(x)||_inf, ||max(g(x),0)||_inf), supplied by phase_I_optim
            # as a callable (the auxiliary problem this loop actually sees has
            # h=None and the single constraint g(x)<=s, so neither its own
            # infeasibility nor its objective is the quantity we care about).
            # This is exactly the condition phase_I_optim verifies on return and
            # exactly what "a feasible starting point" means, so it is checked
            # directly rather than through a surrogate.
            stop_eps = self.phase_I_residual(x) if self.is_phase_I else self.total_infeas[-1]
            if self.verbosity > 0 and ((i + 1) % int(20/self.verbosity) == 0 or (i + 1) == self.max_iter):
                print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {stop_eps:<10.4e} | {f"{jnp.max(rho_vec):<10.4e}" if rho_vec is not None else f"{'-':^10}"} | {f"{jnp.max(nu_vec):<10.4e}" if nu_vec is not None else f"{'-':^10}"} | {f"{self.gamma_k:<10.4e}" if self.use_proximal else f"{'-':^10}"}")

            if ((stop_eps if self.is_phase_I else eps_kkt_res) <= self.tol):
                if self.verbosity > 0:
                    print(
                        f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {stop_eps:<10.4e} | {f"{jnp.max(rho_vec):<10.4e}" if rho_vec is not None else f"{'-':^10}"} | {f"{jnp.max(nu_vec):<10.4e}" if nu_vec is not None else f"{'-':^10}"} | {f"{self.gamma_k:<10.4e}" if self.use_proximal else f"{'-':^10}"}")
                    print("-" * 90)
                    print(f"Convergence achieved after {i + 1} iterations and {(time.time() - solve_start_time):.2f} seconds.")
                    print(f"{'Optimal f1 value found:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {f"{prox_term_i:.6e}" if self.use_proximal else f"{'-':.6}"}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {stop_eps:.6e}")
                    print(f"{'rho:':<25} {f"{jnp.max(rho_vec):.6e}" if rho_vec is not None else f"{'-':.6}"}")
                    print(f"{'nu:':<25} {f"{jnp.max(nu_vec):.6e}" if nu_vec is not None else f"{'-':.6}"}")
                    print(f"{'gamma:':<25} {f"{self.gamma_k:.6e}" if self.use_proximal else f"{'-':.6}"}")
                # must not sit inside the verbosity guard above: the reported
                # status is state, not output, and callers running silently
                # (benchmarks) need it too
                self.solve_status = "Converged"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )
                break
            # verbosity must gate printing only, never control flow: with
            # verbosity=0 this branch used to be skipped entirely, leaving
            # solve_status=None and the user's callback unfired on a run that
            # exhausted max_iter. The inner guard below does the printing.
            elif (i + 1) == self.max_iter:
                if self.verbosity > 0:
                    print("-" * 90)
                    print("Maximum iterations reached without convergence.")
                    print(f"{'f1 value:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {f"{prox_term_i:.6e}" if self.use_proximal else f"{'-':.6}"}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {stop_eps:.6e}")
                    print(f"{'rho:':<25} {f"{jnp.max(rho_vec):.6e}" if rho_vec is not None else f"{'-':.6}"}")
                    print(f"{'nu:':<25} {f"{jnp.max(nu_vec):.6e}" if nu_vec is not None else f"{'-':.6}"}")
                    print(f"{'gamma:':<25} {f"{self.gamma_k:.6e}" if self.use_proximal else f"{'-':.6}"}")
                self.solve_status = "Stopped"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )

            if self.feas_reset_interval is not None and self.feas_reset_interval > 0 and (i + 1) % self.feas_reset_interval == 0:
                if self._is_feasible(x):
                    self.reset_x0 = x.copy() if hasattr(x, 'copy') else jnp.array(x)

            # to reset xhat? (step 1, decides x_hat for the *next* iteration k+1, so
            # must use the just-updated lbda=lambda^{k+1}, mu=mu^{k+1}, rho_vec=rho_{k+1},
            # nu_vec=nu_{k+1}, self.gamma_k=gamma_{k+1} -- state["obj_val"] instead
            # reflects the stale (lambda^k, mu^k, rho_k, nu_k) the inner solve at
            # iteration k was minimized under.
            if self.no_reset:
                x_hat = x
            else:
                f1_reset_x0 = self.problem.f1(self.reset_x0)
                # self-referential prox center (x_prev=x) makes the prox term vanish,
                # matching L_{rho_{k+1},nu_{k+1},gamma_{k+1}}(x^{k+1},...;x^{k+1})
                L_func_val = self._get_prox_L_aug(x, x, lbda, mu, rho_vec, nu_vec, self.gamma_k) + state["reg_val"]
                if self.problem.f2:
                    pz_reset_val = pa.prox(self.problem.f2, self.reset_x0)[0]
                    f1_reset_x0 += pz_reset_val
                    if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                        pz_reset_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), self.reset_x0)[0]
                        f1_reset_x0 += pz_reset_val

                prox_term = self._eval_prox_term(self.reset_x0, x, self.gamma_k)
                if L_func_val > f1_reset_x0 + prox_term:
                    x_hat = self.reset_x0
                else:
                    x_hat = x

        if self.grad_evals is not None:
            target_len = len(self.f_hist)
            while len(self.grad_evals) < target_len:
                self.grad_evals.append(last_grad_eval)

        self.x = x
        self.total_runtime = time.time() - total_start_time
        self.solve_runtime = time.time() - solve_start_time
        if self.verbosity > 0 and self.n_inner_incomplete:
            print(f"{'inner solves < tau_k:':<25} {self.n_inner_incomplete} of "
                  f"{len(self.f_hist)} (best iterate returned; see the note in fs_palm())")
        return

    # @partial(jax.jit, static_argnums=0)
    def _get_prox_L_aug(self, x, x_prev, lbda, mu, rho, nu, gamma_k):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])

        L_aug_val = self.problem.f1(x)

        if self.problem.h:
            eq_term = jnp.sum(rho*0.5*h_x**2) + jnp.dot(lbda,h_x)
            L_aug_val += eq_term

        if self.problem.g:
            nu_g_x_mu = nu*g_x + mu
            ineq_term = jnp.sum((1/(2*nu)) * jnp.where(nu_g_x_mu > 0, nu_g_x_mu**2, 0)) - jnp.sum((1/(2*nu))*mu**2)
            L_aug_val += ineq_term

        prox_term = self._eval_prox_term(x, x_prev, gamma_k)
        L_aug_val += prox_term

        return L_aug_val
    
    @partial(jax.jit, static_argnums=0)
    def _get_palm_obj_fun(self, x_curr, x_prev, lbda, mu, rho, nu, gamma_k):
        if self.is_not_flat_x:
            x = params_flatten(x_curr)
        else:
            x = x_curr.copy()
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])

        L_aug_val = self.problem.f1(x)

        if self.problem.h:
            eq_term = jnp.sum(rho*0.5*h_x**2) + jnp.dot(lbda,h_x)
            L_aug_val += eq_term

        if self.problem.g:
            nu_g_x_mu = nu*g_x + mu
            ineq_term = jnp.sum((1/(2*nu)) * jnp.where(nu_g_x_mu > 0, nu_g_x_mu**2, 0)) - jnp.sum((1/(2*nu))*mu**2)
            L_aug_val += ineq_term

        prox_term = self._eval_prox_term(x, x_prev, gamma_k)
        L_aug_val += prox_term

        return L_aug_val




