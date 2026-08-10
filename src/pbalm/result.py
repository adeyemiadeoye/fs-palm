class Result:
    """
    Class to store the results of the PBALM solver.
    """
    def __init__(self, x, fp_res, kkt_res, total_infeas, f_hist, rho_hist, nu_hist, gamma_hist, prox_hist, solve_status, total_runtime, solve_runtime, grad_evals=None, n_inner_incomplete=0, n_outer=None, beta_test_fired=None, penalty_capped_at=None, rho0=None, nu0=None, gamma0=None, delta=None):
        """
        Initializes the Result object.

        Attributes:
            x: The solution found by the solver.
            fp_res: History of fixed-point residuals.
            kkt_res: History of KKT residuals.
            total_infeas: Total infeasibility at the solution.
            f_hist: History of objective function values.
            rho_hist: History of penalty parameter rho (for equality constraints).
            nu_hist: History of penalty parameter nu (for inequality constraints).
            gamma_hist: History of the proximal parameter gamma.
            prox_hist: History of proximal term.
            solve_status: Status of the solver ('Converged', 'Stopped', 'MaxRuntimeExceeded', 'NanOrInf').
            total_runtime: Total runtime of the solver.
            solve_runtime: Runtime of the solving phase.
            grad_evals: Number of gradient evaluations.
            n_inner_incomplete: Number of outer iterations whose inner solve returned with
                fp_res > tau_k, i.e. PANOC did not reach the requested inner accuracy. This is
                the direct measure of the inner subproblems becoming hard (typically because the
                penalty has grown large enough to ill-condition them); a run can still converge
                with this nonzero, so it is reported rather than treated as a failure.
            n_outer: Number of outer iterations performed.
            rho0, nu0, gamma0, delta: The RESOLVED initialisation actually used. These are
                reported because they are usually not what the caller passed: rho0/nu0 may
                arrive as "rule1"/"rule3"/"auto" and gamma0/delta as "auto", and the numbers
                the rule produced are what has to be quoted when reporting an experiment or
                comparing two runs. Without them a run is not reproducible from its own output.
        """
        self.x = x
        self.fp_res = fp_res
        self.kkt_res = kkt_res
        self.total_infeas = total_infeas
        self.f_hist = f_hist
        self.rho_hist = rho_hist
        self.nu_hist = nu_hist
        self.gamma_hist = gamma_hist
        self.prox_hist = prox_hist
        self.solve_status = solve_status
        self.total_runtime = total_runtime
        self.solve_runtime = solve_runtime
        self.grad_evals = grad_evals
        self.n_inner_incomplete = n_inner_incomplete
        # Per outer iteration: did the beta-progress test on the constraint
        # residual fire? That test, not an incomplete inner solve, is what
        # triggers a penalty increase, and the two are routinely different.
        self.beta_test_fired = beta_test_fired
        # Outer iteration at which rho/nu first hit penalty_max, or None if the
        # cap never bound. Reported because a run that ends at the cap has
        # stopped being governed by the algorithm.
        self.penalty_capped_at = penalty_capped_at
        self.n_outer = n_outer
        self.rho0 = rho0
        self.nu0 = nu0
        self.gamma0 = gamma0
        self.delta = delta