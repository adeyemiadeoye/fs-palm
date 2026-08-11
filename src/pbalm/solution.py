from .problem import GradEvalCounter
from .result import Result
from .pbalm import Solution
from jax import jacfwd
import jax
import jax.numpy as jnp
import numpy as np
import time

def solve(problem, x0, inner_solve_runner=None, lbda0=None, mu0=None, rho0="rule3", nu0="rule3", use_proximal=True,
            gamma0="auto", x_shapes=None, x_cumsizes=None, beta=0.5, alpha=2, alpha_gamma=1.01, delta="auto", xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7,
            start_feas=True, inner_solver=None, pa_direction=None, pa_solver_opts=None, verbosity=1, max_runtime=24.0,
            phi_strategy="pow", feas_reset_interval=None, no_reset=False, adaptive_fp_tol=True, tau0=0.1, kappa_tol=0.5, penalty_horizon=30, penalty_ramp=5.0, penalty_kappa=1e2, delta_rule="objective", gamma_kappa=1e3, penalty_max=1e20, max_iter_inner=None, phase_I_residual=None):
    r"""
    Calls the PBALM solver on the given problem instance.

    Parameters:
        problem: An instance of the Problem class defining the optimization problem.
        x0: Initial guess for the decision variables.
        inner_solve_runner: (Optional) Class that holds the inner solver routine `train_fun`; train_fun(palm_obj_fun, x, max_iter_inner, fp_tol).
        lbda0: Initial Lagrange multipliers for equality constraints. Defaults to zero.
        mu0: Initial Lagrange multipliers for inequality constraints. Defaults to zero, and is
            clipped at zero if supplied. NOTE: earlier versions defaulted to a seeded RANDOM
            standard-normal draw; see the comment at the initialisation site for why that was
            replaced (it cost 8-53% more gradient evaluations and broke the scale invariance of
            gamma_0). Experiments run with a previous version were using random dual starts.
        rho0: Initial penalty for equality constraints. A float, or one of:
            "rule3"  (DEFAULT) penalty_kappa*L_f/||J_c(x0)||**2/(penalty_ramp+1)**alpha, curvature
                     matched: ||J_c||**2 is what the penalty contributes to the Hessian, so the
                     ratio is the penalty at which constraints are felt as strongly as the
                     objective. Correctly handles phi==0 (classic ALM) by not dividing, so it is
                     the recommended choice there too -- measured on basis pursuit under PANOC,
                     same solution as "rule1" at 385 vs 2941 gradient evaluations. Validated on
                     both sides of the ratio; see rule3_penalty0.
            "rule1"  value-based, no curvature estimate; rho_base/(penalty_horizon+1)**alpha.
                     Degenerates at a feasible x0, where ||c(x0)||=0 leaves it carrying no
                     information about the constraints. Used internally as rule3's fallback when
                     the curvature estimate is unusable; also selectable directly.
            "auto"   Birgin & Martinez's 2|f(x0)|/||c(x0)||**2. A literature baseline, not tuned
                     for this method.
        nu0: Initial penalty for inequality constraints; same options as rho0.
        penalty_kappa: Dimensionless multiplier in rule3 (default 1e2), chosen as the best single
            value across all three families rather than the best on any one. Per-family optima
            disagree -- about 1e3 on QCQP and basis pursuit, about 1e1 on the convex QPs -- so no
            constant is optimal everywhere; the question a default has to answer is whether some
            constant is REASONABLE everywhere. Scored against each instance's own best penalty,
            over ten instances spanning the three families:

                kappa   geo-mean  worst  |  QCQP  basis pursuit  convex QP
                 1e1      1.68     3.3   |  1.33      2.76         1.00
                 1e2      1.38     2.8   |  1.02      1.82         1.47
                 1e3      1.61     3.4   |  1.10      1.89         2.50
                 1e4      1.58     5.0   |  1.62      1.01         3.60

            1e2 is the only value within 2x of every family's own optimum. Convex QP rows are
            restricted to cond(Q) <= 1e2, above which the eps-KKT residual stalls at the
            large-nu_0 end and the comparison selects small nu_0 mechanically.
        use_proximal: Boolean indicating whether to use proximal ALM or standard ALM.
        gamma0: Initial proximal parameter, or "auto" for gamma_kappa/L_hat of the augmented Lagrangian at x0 (see proximal_init).
        gamma_kappa: Multiplier in gamma_0 = gamma_kappa/L_hat when gamma0 == "auto" (default 1e3).
            gamma is the proximal REGULARISATION, not a stepsize: (1/2gamma)||x-xhat||^2 adds 1/gamma
            to the curvature, so gamma_kappa=1 makes the proximal term as strong as the local curvature
            of the augmented Lagrangian and over-damps the iteration. gamma_kappa is therefore the
            INVERSE FRACTION of the local curvature the prox contributes: 1e3 adds about 0.1%.
            Calibrated under the ProxGrad inner solver, which is the discriminating instrument here --
            PANOC's quasi-Newton direction absorbs a mis-set gamma and barely separates the values
            (1.43 / 1.07 / 1.07 / 1.05 for 1e2 / 1e3 / 1e4 / 1e5), whereas under ProxGrad 1e2 costs
            1.9x on one instance and 48.7x on another. The cost is FLAT from 1e3 to 1e6, and 1e3 is
            chosen as the smallest value on that plateau: gamma_kappa -> infinity means gamma_0 ->
            infinity, i.e. no proximal term at all and FS-P-ALM degenerating to FS-ALM, so the
            smallest plateau value is the strongest regularisation that costs nothing.
        x_shapes: (Optional) Shapes of decision variable blocks (for structured variables).
        x_cumsizes: (Optional) Cumulative sizes of decision variable blocks (for structured variables).
        beta: Penalty update parameter (0 < beta < 1).
        alpha: Penalty update parameter (alpha > 1); exponent of the growth function phi for rho and nu.
        alpha_gamma: Exponent of the separate growth function phi_gamma for the proximal parameter gamma (alpha_gamma > 1, close to 1 so gamma grows much more slowly than rho, nu).
        delta: Proximal parameter update factor (delta > 0), or "auto" to set it together with gamma0 via proximal_init.
        xi1: Penalty update scaling factor for equality constraints (xi1 >= 1, xi1=1 for the paper algorithm).
        xi2: Penalty update scaling factor for inequality constraints (xi2 >= 1, xi2=1 for the paper algorithm).
        tol: Tolerance for convergence.
        fp_tol: Fixed-point tolerance for inner solver (if None, defaults to tol). Can be provided as a float or a function of iteration number k.
        max_iter: Maximum number of PBALM iterations.
        phase_I_tol: Tolerance for Phase I feasibility problem.
        start_feas: Boolean indicating whether to start with Phase I feasibility optimization.
        inner_solver: "PANOC" (default, via alpaqa: forward-backward with an L-BFGS direction) or
            "ProxGrad" (plain proximal gradient with a backtracking linesearch on the descent lemma).
            The choice is not neutral for what an experiment shows: PANOC's quasi-Newton direction
            absorbs ill-conditioned subproblems, so it understates the difference between the
            penalty schedules. Under ProxGrad the classical ALM enters a feedback loop -- a missed
            inner tolerance fails the beta-test, which multiplies the penalty, which worsens the
            conditioning -- and the fixed-schedule variants cannot, because their penalty is capped
            by the schedule. Report which one was used.
        pa_direction: (Optional) Direction method for PANOC inner solver. See PANOC documentation for details.
        pa_solver_opts: (Optional) Additional options for the PANOC inner solver.
        verbosity: Level of verbosity for logging (<=0: silent).
        max_runtime: Maximum runtime in hours.
        phi_strategy: Strategy for updating the minimum penalty parameter ("pow", "log", "linear").
        feas_reset_interval: Interval for resetting \hat{x} to a recent feasible point (step 1 in the algorithm).
        no_reset: Boolean indicating whether to disable resetting \hat{x}.
        adaptive_fp_tol: Boolean indicating whether to adaptively update the inner solver's fixed-point tolerance. If True (default), uses fp_tol if provided as a function, otherwise the schedule tau_k = max(tau0*kappa_tol**k, tol).
        tau0: Initial inner tolerance for the adaptive schedule (default 0.1).
        kappa_tol: Geometric decay factor of the inner tolerance per outer iteration (default 0.5). The schedule saturates at `tol`, so the subproblems are eventually solved to the requested accuracy; setting tau0=tol**(1/3), kappa_tol=0.1 reproduces a common augmented-Lagrangian tolerance-decay rule.
        penalty_horizon: Expected number of OUTER iterations (default 30), used by the "rule1" initialization. Not max_iter: using the iteration cap would make the penalty grow far too slowly to enforce the constraints.
        delta_rule: Which rule "auto" uses for delta -- "objective" (1/max(1,|f(x0)|), the default) or
            "displacement" (gamma0/max(1,||x0||)^2).
            WHAT DELTA ACTUALLY DOES. gamma_k = max{delta*||x0-x^k||^2, gamma_0*phi_gamma(k+1)}, so
            delta only matters when the first branch wins. Instrumented over 19-21 outer iterations
            at the recommended gamma_0: on the QCQPs, delta = 1 and delta = 1e-6 both leave the branch
            NEVER active (0/19) and give bit-identical evaluation counts, so any sane delta is inert
            there. On basis pursuit the branch IS active at delta = 1 (17/19 iterations) and changes
            cost by up to 1.3x in either direction. The "objective" rule returns 1.0 on the QCQPs
            (f(x0)=0) and ~2e-3 on basis pursuit, landing in the inert regime on BOTH -- which is
            what the manuscript's per-problem delta = 1 / delta = 1e-6 split was achieving by hand.
        penalty_max: Upper bound on rho and nu (default 1e20), applied to every variant so the
            comparison stays symmetric. Uncapped, the multiplicative update reaches ~1e290 on the
            harder instances, where rho*||h||^2 carries no usable precision in double and the run
            reports a numerical breakdown rather than an algorithmic one. A numerical guard of
            ours, not standard practice: classic ALM implementations typically do not cap the
            penalty and instead clip the MULTIPLIERS. Generous by design -- reaching ||h|| ~ 1e-6
            by penalty alone needs rho ~ 1e14 here -- and measured to be immaterial: caps of
            1e20 and 1e300 gave identical cost and outcome on basis pursuit.
        max_iter_inner: Maximum iterations for the inner solver. Defaults per solver when None:
            1000 for PANOC, 5000 for ProxGrad. The two need caps an order of magnitude apart --
            a proximal-gradient subproblem takes many more cheap iterations, while giving PANOC
            the larger cap makes it grind on subproblems it would otherwise abandon (measured 4x
            worse on basis pursuit).
    """
    
    # solve() replaces problem.h/problem.g with a single jitted function that
    # concatenates the user's list of constraints. Keep the original lists so
    # that passing the same Problem to solve() more than once (as the example
    # scripts do, to compare ALM/BALM/P-BALM on one problem) rebuilds from the
    # lists instead of trying to iterate the already-wrapped function.
    if problem.h is not None and not hasattr(problem, "_h_list"):
        problem._h_list = problem.h
    if problem.g is not None and not hasattr(problem, "_g_list"):
        problem._g_list = problem.g

    lbda_sizes = []
    mu_sizes = []
    if problem.h is not None:
        lbda_sizes = [h(x0).flatten().shape[0] for h in problem._h_list]
        problem.h = get_constraint_function(problem._h_list)
        problem.h = jax.jit(problem.h) if problem.jittable else problem.h
        if problem.jittable:
            problem.h_grad = GradEvalCounter(jax.jit(jacfwd(problem.h)))
        else:
            problem.h_grad = GradEvalCounter(jacfwd(problem.h))
        h0 = problem.h(x0)
        if lbda0 is None:
            # Zero, not a random draw. See the note on mu0 below.
            lbda0 = jnp.zeros_like(h0)
    if problem.g is not None:
        mu_sizes = [g(x0).flatten().shape[0] for g in problem._g_list]
        problem.g = get_constraint_function(problem._g_list)
        problem.g = jax.jit(problem.g) if problem.jittable else problem.g
        if problem.jittable:
            problem.g_grad = GradEvalCounter(jax.jit(jacfwd(problem.g)))
        else:
            problem.g_grad = GradEvalCounter(jacfwd(problem.g))
        g0 = problem.g(x0)
        if mu0 is None:
            # Zero, not a random draw. The previous default was
            #     rng = np.random.default_rng(1234)
            #     mu0 = jnp.array(rng.standard_normal(g0.shape))
            # i.e. half zeros and half positive standard-normal values. Three
            # reasons that was wrong, in increasing order of consequence.
            #
            # 1. It costs. Measured over five QCQP instances at otherwise
            #    identical settings, mu0 = 0 was cheaper on all five (277 vs
            #    298, 195 vs 298, 161 vs 216, 184 vs 203, 154 vs 209 gradient
            #    evaluations) at the same objective value.
            #
            # 2. It has the wrong units, so it BREAKS THE SCALE INVARIANCE of
            #    the rest of the initialisation. mu has units [f]/[c]; multiply
            #    every constraint by s and the correct multiplier scales by
            #    1/s, but a standard-normal draw does not move. Since mu enters
            #    the augmented Lagrangian whose curvature sets gamma_0, the
            #    damage propagates: on a controlled pair differing only by a
            #    1e-3 rescaling of the constraints, the random start gave
            #    gamma_0 = 2.87e1 and 1.00e3 for what is the same problem,
            #    while mu0 = 0 gives 1.00e3 for both.
            #
            # 3. It falsifies the premise the penalty rule is derived from. At
            #    a strictly feasible x0 with mu0 = 0 the inequality penalty
            #    term max(0, mu + nu*g)^2 is identically zero in a
            #    neighbourhood, so it contributes no curvature there. With
            #    mu0 > 0 it is active even though g(x0) < 0: measured, the
            #    augmented Lagrangian Hessian at x0 had norm 36.4 against the
            #    2.894 of the objective alone.
            mu0 = jnp.zeros_like(g0)
        mu0 = jnp.maximum(mu0, 0.0)
    problem.lbda_sizes = lbda_sizes
    problem.mu_sizes = mu_sizes
    solution = Solution(problem, x0, inner_solve_runner, lbda0, mu0, rho0, nu0, gamma0,
                        x_shapes, x_cumsizes, use_proximal=use_proximal, beta=beta, alpha=alpha, alpha_gamma=alpha_gamma, delta=delta,
                        xi1=xi1, xi2=xi2, tol=tol, fp_tol=fp_tol, max_iter=max_iter, phase_I_tol=phase_I_tol, start_feas=start_feas,
                        inner_solver=inner_solver, pa_direction=pa_direction, pa_solver_opts=pa_solver_opts, verbosity=verbosity,
                        max_runtime=max_runtime, phi_strategy=phi_strategy, feas_reset_interval=feas_reset_interval,
                        no_reset=no_reset, adaptive_fp_tol=adaptive_fp_tol, tau0=tau0, kappa_tol=kappa_tol, penalty_horizon=penalty_horizon, penalty_ramp=penalty_ramp, delta_rule=delta_rule,
                        gamma_kappa=gamma_kappa, penalty_kappa=penalty_kappa, penalty_max=penalty_max, max_iter_inner=max_iter_inner, phase_I_residual=phase_I_residual)
    solution.pbalm()
    res = Result(solution.x, solution.fp_res, solution.kkt_res, solution.total_infeas, solution.f_hist, solution.rho_hist, solution.nu_hist, solution.gamma_hist, solution.prox_hist, solution.solve_status, solution.total_runtime, solution.solve_runtime, grad_evals=getattr(solution, 'grad_evals', None),
                 n_inner_incomplete=getattr(solution, 'n_inner_incomplete', 0),
                 beta_test_fired=getattr(solution, 'beta_test_fired', None),
                 penalty_capped_at=getattr(solution, 'penalty_capped_at', None),
                 n_outer=len(getattr(solution, 'f_hist', []) or []) or None,
                 # the RESOLVED values, after "rule1"/"rule3"/"auto" have been
                 # turned into numbers -- see Result for why these are reported
                 rho0=getattr(solution, 'rho0', None),
                 nu0=getattr(solution, 'nu0', None),
                 gamma0=getattr(solution, 'gamma0', None),
                 delta=getattr(solution, 'delta', None))
    
    return res

def get_constraint_function(constraints):
    @jax.jit
    def con_fun(x):
        return jnp.concatenate([con(x).flatten() for con in constraints])
    return con_fun