import jax
import jax.numpy as jnp
import numpy as np
import fs_palm

def params_flatten(params):
    return jnp.concatenate([p.flatten() if isinstance(p, jnp.ndarray) else jnp.array([p]) for p in params])
def params_shape(params):
    shapes = [p.shape if isinstance(p, jnp.ndarray) else () for p in params]
    cumsizes = np.cumsum([np.prod(s) for s in shapes])
    # cumsizes = jnp.array(cumsizes, dtype=int)
    return shapes, cumsizes
def params_unflatten(params_flattened, shapes, cumsizes):
    params_split = jnp.split(params_flattened, cumsizes)
    params_unflattened = [array.reshape(shape) for array, shape in zip(params_split, shapes)]
    return params_unflattened

def algencan_penalty0(f0, viol_sq, lo=1e-8, hi=1e1):
    """Birgin & Martinez's scale-aware initial penalty parameter.

        rho_0 = max{lo, min{hi, 2|f(x0)| / ||c(x0)||^2}}

    chosen so the penalty term (rho/2)||c||^2 is comparable in magnitude to
    the objective at the starting point. A fixed constant cannot do this: the
    right value has units of [f]/[c^2], so it changes with the scaling of
    either, which is why one hard-coded default works on one problem and
    fails on the next.

    `viol_sq` is ||c(x0)||^2. If it vanishes (a starting point that is already
    feasible, as in the phase-I auxiliary problem by construction) the ratio
    is undefined and the upper clip is returned; callers that have a more
    meaningful scale should pass it instead.
    """
    f0 = float(f0)
    viol_sq = float(viol_sq)
    if not np.isfinite(f0) or not np.isfinite(viol_sq) or viol_sq <= 0.0:
        return float(hi)
    return float(min(hi, max(lo, 2.0 * abs(f0) / viol_sq)))


def guarded_penalty_base(f0, viol_sq, lo=1e-8, hi=1e8):
    """Scale-aware base penalty with max(1, .) guards.

    Adapted from a common augmented-Lagrangian initialization scheme, which
    stores mu = 1/rho and sets

        mu = 0.1 * max(1, ||c(x0)||^2/2) / max(1, |f(x0)|),  clipped to [1e-8, 1e8]

    which has the same shape as Birgin & Martinez's rho ~ |f|/||c||^2 but with
    max(1, .) guards on both numerator and denominator. Those guards matter:
    they keep the rule finite when the starting point is feasible and the
    objective vanishes (f=0, ||c||=0), where the ungarded ratio is 0/0 --
    exactly the case for a QCQP started at x0 = 0.

    Be aware that on well-scaled problems (|f| and ||c||^2 both O(1)) this
    returns its fallback of 10 and is not adapting to anything; the scale
    awareness only bites when the two are far apart.
    """
    f0, viol_sq = float(f0), float(viol_sq)
    if not np.isfinite(f0) or not np.isfinite(viol_sq):
        return 10.0
    mu = 0.1 * max(1.0, 0.5 * viol_sq) / max(1.0, abs(f0))
    mu = min(max(mu, lo), hi)
    return float(1.0 / mu)


def rule1_penalty0(f0, viol_sq, alpha, horizon=30):
    """Rule 1: scale-aware base, corrected for this method's growth schedule.

        rho_0 = rho_base / (horizon+1)^alpha

    FS-P-ALM grows the penalty as rho_k = rho_0 (k+1)^alpha, so rho_0 and alpha
    are not independent: what matters is the magnitude the trajectory reaches
    by the time the method is expected to converge. Anchoring at a scale-aware
    rho_base (guarded_penalty_base) and dividing by the growth factor at the
    horizon makes rho_k arrive at rho_base after `horizon` iterations, whatever
    alpha is.

    `horizon` is the EXPECTED number of outer iterations, not max_iter: using
    the iteration cap would make the penalty grow far too slowly ever to
    enforce the constraints.

    Measured on a nonconvex QCQP (n=30, m=10, tol=1e-6) this removes the
    dependence on alpha that otherwise dominates. With the fixed rho_0=1e-3,
    alpha=12 fails outright (relative objective gap 0.75 after 1.5e6 gradient
    evaluations); with this rule every alpha in {2,4,12} converges, and the
    gap/violation come out at ~5e-9/1e-7 with a few hundred evaluations.
    """
    return guarded_penalty_base(f0, viol_sq) / (horizon + 1.0) ** alpha


def rule3_penalty0(L_obj, J_con_sq, alpha, kappa=1e2, ramp=5.0,
                   lo=1e-14, hi=1e14, growth=None):
    """Rule 3 (the default; see resolve() in fs_palm.py): a curvature-ratio
    anchor reached over a SHORT ramp, on corrected curvature estimates, with
    the dimensionless constant actually calibrated.

        nu_star = L_obj / ||J_c(x0)||^2
        nu_0    = kappa * nu_star / growth(ramp)

    Two ideas, both still doing real work relative to rule1 (value-based,
    anchored at the full outer-iteration horizon rather than a curvature
    ratio reached early -- see rule1_penalty0).

    THE SCALE. At a strictly feasible x0 -- what \\PFS-ALM{} requires -- a
    value-based anchor like Birgin & Martinez's 2|f|/||c||^2 degenerates:
    ||c(x0)||=0 leaves it a pure objective magnitude, carrying no information
    about the constraints. A ratio of CURVATURES does not degenerate: the
    penalty term (nu/2)||c(x)||^2 contributes nu*||J_c||^2 to the Hessian, so
    nu = L_obj/||J_c||^2 is the penalty at which the constraints are felt as
    strongly as the objective's own curvature -- well defined at a feasible
    point, and invariant to rescaling either f or c.

    THE HORIZON. Anchoring at the full expected iteration count K (rule1 uses
    K=30) means the penalty only reaches a useful magnitude at the very END
    of the run -- with alpha=12 the factor 31^12~8e17 leaves the constraints
    essentially unenforced for the whole budget. `ramp` (a handful of
    iterations, not the horizon) is when the penalty should instead reach the
    target; the schedule's job from then on is only to keep growing it.

    The target is the same for every variant -- the penalty should REACH
    kappa*nu_star after `ramp` outer iterations -- so the divisor has to be
    the growth that variant actually has, and `growth` supplies it:

        phi(k) = (k+1)^alpha   ->  growth = (ramp+1)^alpha    [FS-P-ALM, FS-ALM]
        phi == 0               ->  growth = 1                 [classical ALM]

    DO NOT APPLY THE SCHEDULE CORRECTION TO A VARIANT THAT HAS NO SCHEDULE.
    Classical ALM sets phi == 0 and grows the penalty only by nu <- xi*nu when
    the beta-test on the constraint residual fails. Dividing its nu_0 by
    (ramp+1)^alpha starts it 6^alpha below the target -- 36x at alpha=2, 1296x
    at alpha=4 -- with nothing that reliably climbs back. Worse, the way it
    climbs back is precisely the pathology the fixed-schedule variants are
    meant to avoid: too small a penalty fails the beta-test, which multiplies
    nu, which worsens the subproblem, which makes the inner solve miss its
    tolerance, which fails the test again. Handicapping ALM this way manufactures
    the very runaway a comparison would then report as a finding. Measured
    before this argument was added: under a plain proximal-gradient inner solver,
    ALM so initialised drove nu to 6.4e286 with 489 of 501 inner solves
    incomplete, on a problem where it converges in 1505 evaluations when started
    at the target scale.

    `growth=None` keeps the (ramp+1)^alpha behaviour for backward compatibility;
    fs_palm passes the correct one for the configured phi_strategy.

    Three things separate this from rule2 (removed from this module -- it had
    no remaining callers once this rule replaced it everywhere; kept only by
    name below since the comparison is what justifies the current constants).

    (1) THE ESTIMATES. rule2 was fed a two-point finite-difference probe that
        measures ||H d|| along one direction, not ||H||; on these families it
        came out 1.4x-2.4x low, and the bias is instance dependent, so it does
        not cancel in the ratio. L_obj now comes from spectral_norm_hess and
        ||J_c||^2 from spectral_norm_jtj, both verified against exact
        references to <=2e-8 relative on the QCQP, basis-pursuit and convex-QP
        families.

    (2) THE CONSTANT. rule2 had kappa = 1 implicitly, i.e. it asserted that the
        best penalty is exactly the curvature-matching one. Measured, it is not:
        sweeping the dimensionless multiplier at CONSTANT penalty over nine
        instances (five QCQP, four basis pursuit), no multiplier below 1e3
        converged on all of them, and 1e3 costs a geometric-mean 1.97x
        (worst 5x) against tuning the penalty per instance. So rule2 was
        starting three orders of magnitude too low.

    (3) WHY THE FORM IS RIGHT, WHICH IS THE PART THAT GENERALISES. nu_0 has
        units [f]/[c]^2, so any fixed default is wrong by whatever scaling the
        user happened to choose for the constraints. The controlled test:
        multiplying every constraint function of a QCQP by 1e-3 leaves the
        feasible set, the solution and the problem unchanged, yet moves the
        measured optimal penalty by exactly 1e6 (2.915e-01 -> 2.915e+05) --
        and the multiplier nu_opt/nu_star is 1e2 for BOTH, ratio 1.000. Across
        five QCQP instances the form removes 1.2e4 of the 1.2e6 spread in the
        optimal penalty.

    REJECTED ALTERNATIVE. Classical ALM theory ties the dual contraction rate
    to 1/(rho*sigma_min(J_A)^2), suggesting nu_star = L_obj/sigma_min(J)^2
    instead. That form is 15x WORSE here (spread 1.5e5 vs 1.0e4): on a pair of
    instances differing only in how nearly parallel the constraint normals are
    (cond(J)^2 of 7.1 vs 2554) it predicts a 361x change in the multiplier,
    where the measurement shows 1x. The dominant direction, not the worst
    conditioned one, is what sets the scale in this method.

    ON THE PRECISION REQUIRED. Not much, which is what makes a single published
    constant defensible. With the growth schedule active the cost is flat in
    kappa over several decades, because an UNDER-estimate is erased by
    nu_k = nu_0*(k+1)^alpha within a few outer iterations while an OVER-estimate
    is permanent -- the schedule only pushes nu up. Erring low is therefore the
    safe direction.

    WHY THIS IS NOT THE DEFAULT: kappa IS FAMILY DEPENDENT. On the convex QP
    family the best multiplier is 10, not 1e3, so this rule starts nu_0 about
    two decades too high there and loses to rule1 (288 vs 816 gradient
    evaluations at cond(Q)=1e2; at cond(Q)=1e6 it starts from nu_0 = 5.2e6 and
    ends with a 10% objective gap). Since 10 and the 1e2-1e4 plateau measured
    on QCQP and basis pursuit are one to three orders apart, no single constant
    covers the three families, and picking one silently would be a per-problem
    tuning knob wearing the costume of a rule.

    The FORM survives that, on both sides of the ratio:

      denominator  a controlled 1e-3 rescaling of the constraints (above) moves
                   the optimal penalty by exactly 1e6 and leaves the multiplier
                   unchanged, ratio 1.000;
      numerator    on convex QPs with A -- hence ||J||^2 -- held fixed and only
                   lambda_max(Q) varied, the implied multiplier is 10.00 at
                   cond(Q)=1e1 and 10.00 at cond(Q)=1e2, i.e. the optimal
                   penalty tracks lambda_max exactly over the range where the
                   measurement is clean.

    "Clean" is doing real work in that sentence. Above cond(Q) ~ 1e2 the
    large-nu_0 runs stop converging -- not by diverging, but by the eps-KKT
    residual stalling near 1e-4 while the objective gap and the violation are
    ~1e-10 -- so restricting to converged runs selects small nu_0 mechanically
    and the apparent optimum collapses. Do not calibrate a penalty on that
    family above cond(Q) ~ 1e2 until the stopping test is fixed.
    """
    nu_star = float(L_obj) / max(float(J_con_sq), 1e-300)
    g = (ramp + 1.0) ** alpha if growth is None else max(float(growth), 1e-300)
    return float(min(hi, max(lo, float(kappa) * nu_star / g)))


def _power_iteration(matvec, x0, iters, tol, seed):
    """Shared power-iteration core. Returns (value, converged, used_iters).

    `value` is None when the iteration cannot produce a usable number (the
    operator annihilates the iterate, or a matrix-vector product cannot be
    formed). `converged` says whether the relative change in the estimate met
    `tol` before the iteration cap -- WITHOUT it the returned number is a lower
    bound on the true spectral norm, not an estimate of it, since power
    iteration approaches the dominant eigenvalue from below. Callers must not
    silently treat an unconverged value as accurate: a clustered spectrum
    (lambda_2/lambda_1 close to 1) converges at rate (lambda_2/lambda_1)^k and
    can be far short after `iters` steps.
    """
    x0 = jnp.asarray(x0)
    v = jnp.asarray(np.random.default_rng(seed).standard_normal(x0.shape),
                    dtype=x0.dtype)
    nv = float(jnp.linalg.norm(v))
    if not np.isfinite(nv) or nv == 0.0:
        return None, False, 0
    v = v / nv
    lam = 0.0
    for it in range(1, int(iters) + 1):
        # Deliberately narrow: a matrix-vector product may legitimately be
        # unformable (TypeError/ValueError from a non-differentiable oracle),
        # in which case the caller falls back. Anything else is a bug and must
        # not be silently converted into "estimator unavailable" -- a bare
        # except here previously hid a missing `import jax`.
        try:
            w = matvec(v)
        except (TypeError, ValueError):
            return None, False, it
        nw = float(jnp.linalg.norm(w))
        if not np.isfinite(nw) or nw == 0.0:
            return None, False, it
        v = w / nw
        # Relative test. An absolute floor (the earlier max(1, nw)) makes the
        # test vacuous for small spectra: the "flat" QCQP instance has
        # ||J||^2 ~ 3e-4, where a 1e-10 absolute change is 7 significant
        # digits and passes on the first step regardless of accuracy.
        if abs(nw - lam) <= tol * nw:
            return float(nw), True, it
        lam = nw
    return (float(lam), False, int(iters)) if (np.isfinite(lam) and lam > 0.0) \
        else (None, False, int(iters))


def spectral_norm_hess(grad_fn, x0, iters=300, tol=1e-9, seed=0,
                       return_info=False):
    """Largest |eigenvalue| of the Jacobian of `grad_fn` at x0 (i.e. of the
    Hessian, when grad_fn is a gradient), by power iteration on exact
    Hessian-vector products.

    With `return_info=True` returns (value, converged, iters) instead of the
    bare value; see _power_iteration on why an unconverged value is a lower
    bound rather than an estimate.

    WHY NOT A FINITE-DIFFERENCE PROBE. lipschitz_estimate below perturbs along
    a single direction d and returns ||H d||, which is a Rayleigh-type quantity
    and NOT ||H||. Two ways this misleads, both measured on the QCQP family:

      * for the objective, d is the normalised gradient, which carries no
        relation to the dominant eigenvector: the probe returned 2.86e-01 on
        three instances whose true ||grad^2 f_1|| is 1.0, i.e. 3.5x low;
      * for the penalty curvature H = J'J at a feasible point, J has rank at
        most m while d lives in R^n, so for m << n an arbitrary d lands mostly
        in the (n-m)-dimensional null space of J'J. Measured: 2.9x to 3.6x low
        whenever m/n was small, and accurate only on instances where J'J
        happened to be effectively rank one.

    The bias is systematic and instance-dependent, so it does not cancel when
    forming ratios such as L_f/L_c. Power iteration instead converges to the
    eigenvalue of largest magnitude -- exactly the local Lipschitz modulus of
    grad_fn -- using exact Hessian-vector products (a jvp through grad_fn), so
    there is no step size and no probe-direction sensitivity.

    Costs one Hessian-vector product per iteration; typically a few tens.
    Returns None if the iteration cannot produce a usable value, which happens
    only when the Hessian annihilates the iterate -- a genuine degeneracy (the
    function is flat to second order at x0), not an artefact.
    """
    x0 = jnp.asarray(x0)

    def Hv(v):
        return jnp.asarray(jax.jvp(grad_fn, (x0,), (v,))[1])

    out = _power_iteration(Hv, x0, iters, tol, seed)
    return out if return_info else out[0]


def spectral_norm_jtj(c_fn, x0, iters=300, tol=1e-9, seed=0,
                      return_info=False):
    """Largest eigenvalue of J'J, where J is the Jacobian of `c_fn` at x0 --
    i.e. ||J||_2^2 -- by power iteration on J'(Jv).

    This is the curvature the penalty term (nu/2)||c(x)||^2 would contribute if
    the constraints were active, and it is the quantity the initialisation rule
    calls for.

    DO NOT compute it as the Hessian of 0.5||c||^2. That Hessian is

        J'J + sum_i c_i grad^2 c_i ,

    and the second term vanishes only where c(x0) = 0. For EQUALITY constraints
    at a feasible point it does; for INEQUALITY constraints it does not, since
    feasibility there means g(x0) < 0 strictly, not g(x0) = 0. Measured on a
    QCQP started at x0 = 0, where g_i(x0) = -r_i, the contamination is exactly
    -sum_i r_i P_i and moves the estimate by 16-23%.

    Each iteration costs one jvp and one vjp, so J is never assembled -- which
    matters since forming J costs m gradient evaluations.

    With `return_info=True` returns (value, converged, iters) instead of the
    bare value.
    """
    x0 = jnp.asarray(x0)

    def JtJv(v):
        _, Jv = jax.jvp(c_fn, (x0,), (v,))
        _, vjp = jax.vjp(c_fn, x0)
        return jnp.asarray(vjp(Jv)[0])

    out = _power_iteration(JtJv, x0, iters, tol, seed)
    return out if return_info else out[0]


def lipschitz_estimate(grad_fn, x0, rel_eps=1e-6):
    """Two-point finite-difference estimate of the local Lipschitz constant of
    `grad_fn` at x0. Deterministic (the probe direction is the normalised
    gradient, or a fixed vector where the gradient vanishes) and costs two
    gradient evaluations. Returns None if the estimate is not usable.

    Retained as a FALLBACK for the case where a Hessian-vector product cannot
    be formed; prefer spectral_norm_hess above, which is exact where this one
    is systematically low (see its docstring).
    """
    x0 = jnp.asarray(x0)
    g0 = jnp.asarray(grad_fn(x0))
    ng = float(jnp.linalg.norm(g0))
    if np.isfinite(ng) and ng > 0.0:
        d = g0 / ng
    else:
        d = jnp.ones_like(x0) / jnp.sqrt(jnp.asarray(x0.size, dtype=x0.dtype))
    eps = rel_eps * max(1.0, float(jnp.linalg.norm(x0)))
    g1 = jnp.asarray(grad_fn(x0 + eps * d))
    L = float(jnp.linalg.norm(g1 - g0)) / eps
    return L if (np.isfinite(L) and L > 0.0) else None


def lipschitz_gamma0(grad_fn, x0, kappa=1e3, lo=1e-8, hi=1e4, rel_eps=1e-6,
                     fallback=1e-1):
    """Initial proximal parameter gamma_0 = kappa / L_hat.

    L_hat is a two-point finite-difference estimate of the local Lipschitz
    constant of the gradient of the augmented Lagrangian at x0. This makes the
    proximal regularisation commensurate with the local curvature instead of
    an arbitrary constant, and it carries the right units: [gamma] = [x^2]/[f],
    so (1/2gamma)||dx||^2 has the units of f, and the proximal term scales with
    the objective and is invariant to a rescaling of the variables.

    ON THE VALUE OF kappa. gamma is the proximal REGULARISATION, not a
    stepsize, and the distinction sets the scale. The term
    (1/2gamma)||x-xhat||^2 adds 1/gamma to the curvature seen by the inner
    solver, so kappa = 1 -- i.e. gamma_0 = 1/L_hat, the natural choice if one
    thinks of gamma as a stepsize -- makes the proximal term exactly as strong
    as the local curvature of the augmented Lagrangian, doubling it and
    over-damping the iteration. kappa is therefore the INVERSE FRACTION of the
    local curvature the proximal term contributes: kappa = 1e3 (the default)
    means the prox adds about 0.1% of L_hat, a mild stabiliser.

    Measured, gradient evaluations, kappa = 1 -> 1e2:

        basis pursuit  p=60,  n=150, alpha=4    4045 -> 914   (FS-P-ALM)
        basis pursuit  p=60,  n=150, alpha=6   18879 -> 1747
        basis pursuit  p=100, n=256, alpha=4    5078 ->  753
        QCQP           5 instances, alpha=2,4   0.26x to 0.82x of kappa=1

    On basis pursuit this is the difference between FS-P-ALM losing to FS-ALM
    (1.8-9.4x worse) and beating it (2.2-2.4x better); on QCQP it is never
    worse. hi is 1e4 rather than 1e2 so that the clip does not silently undo
    the kappa scaling on well-conditioned problems.

    Costs two gradient evaluations. Deterministic (the probe direction is the
    normalised gradient, or a fixed vector where the gradient vanishes), so
    repeated runs give identical results.
    """
    # Power iteration first: it is exact where the finite-difference probe is
    # systematically low (by 2.1x-3.5x on the QCQP family). Fall back to the
    # probe only if a Hessian-vector product cannot be formed.
    L = spectral_norm_hess(grad_fn, x0)
    if L is None:
        L = lipschitz_estimate(grad_fn, x0, rel_eps=rel_eps)
    if L is None:
        return float(fallback)
    return float(min(hi, max(lo, kappa / L)))


def proximal_init(grad_al, x0, f0=0.0, kappa=1e3, delta_rule="objective"):
    """Initialise the proximal parameters (gamma_0, delta) together.

    They belong together because they are the two branches of the same update,

        gamma_k = max(delta*||x^0 - x^k||^2,  gamma_0 * phi_gamma(k+1)),

    and are only meaningful relative to one another: delta sets how fast the
    proximal term relaxes as the iterate travels, gamma_0 sets where the
    schedule branch starts.

    gamma_0 = kappa / L_hat  (see lipschitz_gamma0, which documents why kappa is
    ~1e2 and not 1) is scale-correct on its own.
    Note it needs no alpha correction, unlike the penalties: gamma uses the
    separate exponent alpha_gamma (fixed, ~1.01), so its trajectory does not
    move when alpha is swept.

    delta has units 1/[f] -- from [gamma] = [x^2]/[f] and [||dx||^2] = [x^2] --
    so a fixed delta is not scale invariant. That shows in practice: the
    manuscript uses delta=1 for the QP experiments and delta=1e-6 for basis
    pursuit, a six-order spread across two problems, which is the signature of
    a quantity that wants a scale rule rather than per-problem tuning.

    Two candidate rules, selected by `delta_rule`:

      "objective"     delta = 1 / max(1, |f(x^0)|)              [default]
                      Uses the objective scale directly, which is what the
                      units call for, without reference to a displacement.
      "displacement"  delta = gamma_0 / max(1, ||x^0||)^2
                      Makes the two branches of the max equal when the iterate
                      has moved a distance ~||x^0|| from the start, i.e. the
                      distance branch takes over only after real travel.

    Measured over 4 nonconvex QCQP instances x alpha in {2,4,12}: both always
    converge with comparable accuracy; "objective" is cheaper in 6 of 9
    comparisons (up to 2.9x), "displacement" in 2, and "displacement" gives
    slightly tighter feasibility. The margin is modest and family specific, so
    "objective" is the default on grounds of cost and simplicity rather than
    demonstrated superiority.

    Returns (gamma_0, delta).
    """
    gamma0 = lipschitz_gamma0(grad_al, x0, kappa=kappa)
    if delta_rule == "displacement":
        D = max(1.0, float(jnp.linalg.norm(jnp.asarray(x0))))
        delta = gamma0 / D ** 2
    elif delta_rule == "objective":
        delta = 1.0 / max(1.0, abs(float(f0)))
    else:
        raise ValueError(f"unknown delta_rule: {delta_rule!r}")
    return float(gamma0), float(delta)


def update_penalties(lbda_sizes, mu_sizes,
                        rho, nu, rho0, nu0,
                        E_x, prev_E, h_x, prev_h,
                        beta, xi1, xi2, phi_i, skip_test=False):
    # At k=0, x^0 is feasible so h(x^0)=0 and E^0=0 for any active inequality:
    # the beta-progress test is degenerate (unpassable) and would force a
    # spurious penalty increase regardless of x^1. Skip it for k=0.
    if skip_test:
        return rho, nu

    m_1 = 0
    m_n = 0
    nu_new = nu
    for mu_i in mu_sizes:
        m_n += mu_i
        nu_i = nu[m_1:m_n]
        nrm_Ei = jnp.linalg.norm(E_x[m_1:m_n], jnp.inf)
        prev_nrm_Ei = jnp.linalg.norm(prev_E[m_1:m_n], jnp.inf)
        if nrm_Ei > beta*prev_nrm_Ei:
            nu_i_new = jnp.maximum(xi2*nu_i, jnp.full_like(nu_i, nu0*phi_i))
            nu_new = nu_new.at[m_1:m_n].set(nu_i_new)
        m_1 = m_n
    
    l_1 = 0
    l_n = 0
    rho_new = rho
    for lbda_i in lbda_sizes:
        l_n += lbda_i
        hxi = h_x[l_1:l_n]
        rho_i = rho[l_1:l_n]
        nrm_hi = jnp.linalg.norm(hxi, jnp.inf)
        prev_nrm_hi = jnp.linalg.norm(prev_h[l_1:l_n], jnp.inf)
        if nrm_hi > beta*prev_nrm_hi:
            rho_i_new = jnp.maximum(xi1*rho_i, jnp.full_like(rho_i, rho0*phi_i))
            rho_new = rho_new.at[l_1:l_n].set(rho_i_new)
        l_1 = l_n

    return rho_new, nu_new

class GradEvalCounter:
    """
    Wrapper class to count the number of gradient evaluations.
    """
    def __init__(self, fn):
        self.fn = fn
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.fn(*args, **kwargs)
    def reset(self):
        self.count = 0