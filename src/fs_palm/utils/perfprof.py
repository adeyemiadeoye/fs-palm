"""Performance profiles, shifted geometric means, and a solver-independent
success criterion, for benchmarking FS-P-ALM against other solvers.

Protocol follows the two references we settled on:

  [BM]  Birgin & Martinez, "Complexity and performance of an Augmented
        Lagrangian algorithm" -- success is re-checked by the *driver* from
        the returned point, never read from a solver's own status flag:
            (61)  max{||h(x)||_inf, ||[g(x)]_+||_inf} <= eps_feas
            (63)  f_i <= f_min + ftol * max{1, |f_min|},  f_min = min_i f_i
        Robustness is reported as a table over a range of ftol, separately
        from the efficiency profile.

  [H]   Hermans, "QPALM" thesis Sec. 9.1 -- Dolan-More profile with the
        r = inf convention for failures, plus the shifted geometric mean
        as a scalar summary.

  [GS]  Gould & Scott, ACM TOMS 43(2), 2016 -- performance profiles
        misrepresent performance when more than two solvers are compared
        at once, because r_{p,s} is normalised by the best solver *on each
        problem*, so adding or removing a solver moves every other curve.
        Hence `pairwise_profiles` below refuses to plot more than two.

Nothing here imports a solver; it operates on a plain results table, so the
run step and the plot step stay independent (re-plot without re-solving).
"""
import numpy as np

INF = np.inf


# ---------------------------------------------------------------- success ---
def feasibility_violation(h_val, g_val):
    """max{||h(x)||_inf, ||[g(x)]_+||_inf} -- criterion (61) of [BM].

    Both arguments may be None (no such constraints) or array-like.
    """
    v = 0.0
    if h_val is not None and np.size(h_val) > 0:
        v = max(v, float(np.max(np.abs(np.asarray(h_val)))))
    if g_val is not None and np.size(g_val) > 0:
        v = max(v, float(np.max(np.maximum(np.asarray(g_val), 0.0))))
    return v


def success_mask(viol, fval, eps_feas=1e-6, ftol=1e-1, f_ref=None):
    """Solver-independent success test, criteria (61)+(63) of [BM].

    Parameters
    ----------
    viol : (n_prob, n_solv) array
        Feasibility violation of each returned point, from
        `feasibility_violation`. NaN marks a run that produced no point at
        all (crash / time-out); it is treated as a failure.
    fval : (n_prob, n_solv) array
        Objective value at each returned point. NaN treated as failure.
    eps_feas, ftol : float
        Tolerances for (61) and (63) respectively.
    f_ref : (n_prob,) array, optional
        KNOWN optimal value per problem. When given, (63) is applied against
        it instead of against the best value the compared solvers attained.
        Use this whenever the true optimum is available: the default
        best-found reference cannot detect the case where EVERY solver stops
        at the same spurious stationary point, since that point then defines
        the reference and all of them "pass". NaN entries fall back to the
        best-found rule for that problem.

    Returns
    -------
    (n_prob, n_solv) bool array.

    With the default reference the (63) comparison is *relative to the best
    objective attained on that problem by the solvers being compared*, so it
    is only meaningful within a fixed comparison set -- which is another
    reason the pairwise rule of [GS] matters: change the solver set and the
    mask changes too. Passing f_ref removes that coupling entirely.
    """
    viol = np.asarray(viol, dtype=float)
    fval = np.asarray(fval, dtype=float)

    feasible = np.isfinite(viol) & (viol <= eps_feas)
    # f_min over solvers that are *feasible* on that problem; problems where
    # nobody is feasible give f_min = nan and nothing passes.
    f_masked = np.where(feasible & np.isfinite(fval), fval, np.nan)
    with np.errstate(invalid="ignore"):
        f_min = np.nanmin(f_masked, axis=1, keepdims=True)
    if f_ref is not None:
        ref = np.asarray(f_ref, dtype=float).reshape(-1, 1)
        if ref.shape[0] != fval.shape[0]:
            raise ValueError(f"f_ref has {ref.shape[0]} entries, expected "
                             f"{fval.shape[0]}")
        f_min = np.where(np.isfinite(ref), ref, f_min)
    good_obj = f_masked <= f_min + ftol * np.maximum(1.0, np.abs(f_min))
    return feasible & np.isfinite(fval) & np.nan_to_num(good_obj, nan=False)


def robustness_table(viol, fval, solver_names, eps_feas=1e-6,
                     ftols=(1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 0.0), f_ref=None):
    """Table 3 of [BM]: #problems solved by each solver as ftol is tightened.

    Returns (ftols, counts) with counts of shape (len(ftols), n_solv).
    f_ref is forwarded to `success_mask`.
    """
    counts = np.array([success_mask(viol, fval, eps_feas, ft,
                                    f_ref=f_ref).sum(axis=0)
                       for ft in ftols], dtype=int)
    return list(ftols), counts, list(solver_names)


def format_robustness_table(ftols, counts, solver_names, n_prob):
    w = max(len(s) for s in solver_names) + 2
    lines = [f"{'ftol':>10}" + "".join(f"{s:>{w}}" for s in solver_names)
             + f"   (out of {n_prob})"]
    lines.append("-" * len(lines[0]))
    for ft, row in zip(ftols, counts):
        lines.append(f"{ft:>10.0e}" + "".join(f"{c:>{w}d}" for c in row))
    return "\n".join(lines)


# ---------------------------------------------------------------- profile ---
def performance_ratios(cost, solved):
    """r_{p,s} = t_{p,s} / min_s t_{p,s}, with r = inf where s failed ([H]).

    Problems that *no* solver solved are dropped: their best cost is
    undefined, and Dolan-More profiles conventionally exclude them.
    Returns (ratios, kept_mask_over_problems).
    """
    cost = np.asarray(cost, dtype=float)
    solved = np.asarray(solved, dtype=bool)
    if cost.shape != solved.shape:
        raise ValueError(f"cost {cost.shape} and solved {solved.shape} differ")

    keep = solved.any(axis=1)
    cost, solved = cost[keep], solved[keep]
    if cost.size == 0:
        return np.zeros((0, solved.shape[1])), keep

    # a failed run has no meaningful cost, so it must not influence the min
    eff = np.where(solved & np.isfinite(cost), cost, INF)
    best = eff.min(axis=1, keepdims=True)
    if np.any(best <= 0):
        raise ValueError("non-positive best cost; costs must be > 0 "
                         "(use e.g. max(t, 1e-6) for sub-resolution timings)")
    return eff / best, keep


def performance_profile(cost, solved, taus=None, n_tau=512):
    """rho_s(tau) = (1/n_p) #{p : r_{p,s} <= tau}, Dolan-More / [H] Sec 9.1.2.

    Returns (taus, rho) with rho of shape (len(taus), n_solv).
    """
    r, _ = performance_ratios(cost, solved)
    if r.size == 0:
        raise ValueError("no problem was solved by any solver")
    n_p = r.shape[0]

    if taus is None:
        finite = r[np.isfinite(r)]
        r_max = finite.max() if finite.size else 2.0
        taus = np.logspace(0, np.log10(max(r_max, 2.0)) * 1.02, n_tau)
    taus = np.asarray(taus, dtype=float)

    rho = (r[None, :, :] <= taus[:, None, None]).sum(axis=1) / n_p
    return taus, rho


def shifted_geometric_mean(cost, solved, shift=1.0, fail_value=None):
    """sgm_s = prod_p (t_{p,s} + shift)^(1/n_p) - shift,  [H] Sec 9.1.1.

    Failures take `fail_value` (Hermans uses the time limit); defaults to the
    worst observed cost over all runs, which plays the same role when there
    is no explicit limit.
    """
    cost = np.asarray(cost, dtype=float)
    solved = np.asarray(solved, dtype=bool)
    finite = cost[np.isfinite(cost)]
    if fail_value is None:
        fail_value = finite.max() if finite.size else 1.0
    eff = np.where(solved & np.isfinite(cost), cost, fail_value)
    return np.exp(np.mean(np.log(eff + shift), axis=0)) - shift


# ----------------------------------------------------------------- plotting --
def pairwise_profiles(cost, solved, solver_names, xlabel, outfile=None,
                      title=None, ax=None):
    """Plot ONE performance profile comparing exactly two solvers.

    Refuses three or more, per [GS]: with r normalised by the per-problem
    best, a 3-way plot cannot be read as three pairwise comparisons.
    """
    if len(solver_names) != 2:
        raise ValueError(
            f"performance profiles compare exactly two solvers at a time "
            f"(got {len(solver_names)}: {list(solver_names)}). "
            "Gould & Scott (2016) show profiles misrepresent performance "
            "with more than two; build one figure per pair instead.")
    import matplotlib.pyplot as plt

    taus, rho = performance_profile(cost, solved)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, name in enumerate(solver_names):
        ax.step(taus, rho[:, s], where="post", label=name)
    # tau is a dimensionless RATIO to the best solver on each problem, not an
    # absolute cost, so the axis is labelled as a factor. Reading rho_s(tau) as
    # "fraction of problems on which s came within tau times the best" is the
    # phrasing used by Hermans (QPALM thesis, Fig. 9.1) and is what the curve
    # actually means; labelling the axis with the raw cost unit invites the
    # reader to misread it as an absolute count.
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"fraction of problems within $\tau\times$ best")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(taus[0], taus[-1])
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    if outfile and created:
        fig.tight_layout()
        fig.savefig(outfile, bbox_inches="tight")
        print(f"wrote {outfile}")
    return ax


# --------------------------------------------------------------- self-test ---
if __name__ == "__main__":
    # 4 problems x 2 solvers, hand-checkable.
    cost = np.array([[1.0, 2.0],     # A best (ratio 1 vs 2)
                     [4.0, 1.0],     # B best (ratio 4 vs 1)
                     [1.0, 1.0],     # tie
                     [5.0, 9.0]])    # A best, but B fails below
    solved = np.array([[True, True], [True, True], [True, True], [True, False]])

    r, keep = performance_ratios(cost, solved)
    assert keep.all()
    assert np.allclose(r[:, 0], [1, 4, 1, 1]), r[:, 0]
    assert np.allclose(r[:, 1], [2, 1, 1, INF]), r[:, 1]

    taus, rho = performance_profile(cost, solved, taus=np.array([1.0, 2.0, 4.0, 1e9]))
    # at tau=1: A best on 3 of 4 (probs 0,2,3); B best on 2 of 4 (probs 1,2)
    assert np.allclose(rho[0], [0.75, 0.50]), rho[0]
    # at tau=2: A still 3/4; B gains problem 0 -> 3/4
    assert np.allclose(rho[1], [0.75, 0.75]), rho[1]
    # at tau=4: A gains problem 1 -> 4/4; B unchanged (prob 3 failed)
    assert np.allclose(rho[2], [1.00, 0.75]), rho[2]
    # as tau->inf the curve plateaus at the fraction solved, never reaching 1 for B
    assert np.allclose(rho[3], [1.00, 0.75]), rho[3]
    assert np.all(np.diff(rho, axis=0) >= 0), "profiles must be non-decreasing"

    # a problem nobody solves is dropped
    c2 = np.vstack([cost, [7.0, 7.0]])
    s2 = np.vstack([solved, [False, False]])
    r2, keep2 = performance_ratios(c2, s2)
    assert r2.shape[0] == 4 and not keep2[-1]

    # success criterion (61)+(63)
    viol = np.array([[1e-9, 1e-9], [1e-9, 1e-3], [1e-9, 1e-9]])
    fval = np.array([[1.00, 1.05], [2.00, 2.00], [10.0, 5.00]])
    m = success_mask(viol, fval, eps_feas=1e-6, ftol=1e-1)
    # p0: both feasible, f_min=1.0, cutoff 1.0+0.1*1=1.1 -> both pass
    assert m[0].tolist() == [True, True], m[0]
    # p1: solver B infeasible -> fails regardless of objective
    assert m[1].tolist() == [True, False], m[1]
    # p2: f_min=5, cutoff 5+0.1*5=5.5 -> A's 10.0 fails
    assert m[2].tolist() == [False, True], m[2]

    sgm = shifted_geometric_mean(cost, solved, shift=1.0, fail_value=10.0)
    assert sgm.shape == (2,) and np.all(sgm > 0)

    try:
        pairwise_profiles(cost, solved, ["A", "B", "C"], "x")
    except ValueError as e:
        assert "exactly two" in str(e)
    else:
        raise AssertionError("should have refused 3 solvers")

    print("perfprof self-test: all assertions passed")
