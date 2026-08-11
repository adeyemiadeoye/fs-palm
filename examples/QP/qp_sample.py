"""Illustrative per-problem plots for the convex QPs.

Compares FS-ALM/FS-P-ALM against classic ALM under different levels of
complexity via the condition number kappa(Q), with alpha = 12 and xi = 10.
Two modes:

  --source synthetic  (default)  one column per CONDITION NUMBER, with
                      everything else about the instance held fixed. Named
                      MM problems vary kappa only incidentally -- DUALC1,
                      CVXQP2_M and LOTSCHD differ in size, sparsity,
                      constraint count and conditioning all at once, so an
                      effect attributed to kappa cannot be separated from
                      the rest. Sweeping kappa on otherwise identical
                      instances is what a claim about conditioning actually
                      requires.

  --source cutest     three named Maros-Meszaros problems (DUALC1, CVXQP2_M,
                      LOTSCHD) instead of synthetic instances. Requires
                      CUTEst compiled locally; see qp_problem.load_cutest.

Conventions:

    ALM-xi        xi > 1, phi(k) = 0, no Step-1 reset      (classic ALM)
    FS-ALM-alpha    xi = 1, phi(k) = (k+1)^alpha
    FS-P-ALM-alpha  as FS-ALM plus the proximal term

ALM is a heuristic baseline NOT covered by the paper's theory (phi = 0 falls
outside the penalty assumption); it is kept as an ablation.

Outputs (PDF, house style), per problem tag:
    qp_sample_gradevals_infeas_<tag>.pdf   grad evals vs total infeasibility
    qp_sample_gradevals_subopt_<tag>.pdf   grad evals vs suboptimality
    qp_sample_penalty_iter_<tag>.pdf       penalty nu_k vs iteration
    qp_sample_legend.pdf                   shared legend (multi-parameter mode)

These problems are CONVEX, so the suboptimality is measured against the true
global optimum from an interior-point solver, not against the best value any
run happened to reach.

Run from this directory:

    python qp_sample.py                              # condition-number sweep
    python qp_sample.py --conds 1e2 1e6              # pick condition numbers
    python qp_sample.py --source cutest              # the named MM problems
    python qp_sample.py --alphas 4 12 --xis 4 10     # parameter sweep
"""
import argparse

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")     # these scripts only write PDFs; the default
                          # interactive backend crashes at interpreter
                          # teardown ("main thread is not in main loop")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# Prefer this repo's solver over any installed fs_palm. A pip-installed fs_palm
# in the same environment would otherwise shadow it and the script would
# benchmark the published package instead of the working copy.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "src")
if _os.path.isdir(_os.path.join(_SRC, "fs_palm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import fs_palm

from fs_palm.utils.plotting import setup_matplotlib, inner_solvers
from qp_problem import (make_instance, load_cutest, fs_palm_problem,
                        MM_COND_PROBLEMS, CONDS)


def style():
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'p', 'H']
    colors = np.concatenate([plt.cm.Dark2(np.linspace(0, 1, plt.cm.Dark2.N)),
                             plt.cm.Set1(np.linspace(0, 1, plt.cm.Set1.N))])
    linestyles = [(5, (10, 3)), (0, (5, 1)), (0, (3, 1, 1, 1)),
                  (0, (3, 1, 1, 1, 1, 1)), '--', '-', '-.', ':']
    return markers, colors, linestyles, 9


_RULES = ("rule1", "rule3", "auto")


def _as_rule(v):
    """Accept a named rule or a numeric value from the CLI.

    Kept in sync with the strategies fs_palm.solve accepts; listing them once
    means a rule added to the solver does not silently become "not a number"
    at the command line.
    """
    if isinstance(v, str) and v not in _RULES:
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"expected one of {_RULES} or a number, "
                             f"got {v!r}")
    return v


def run_variants(inst, alphas, xis, tol, max_iter, delta=1.0, gamma0="auto",
                 gamma_kappa=1e2, alm_rho0="rule3", penalty_rule="rule3",
                 penalty_ramp=5.0, seed=1234, verbose=True,
                 inner_solver="PANOC", only_proximal=False):
    """One curve per algorithm/parameter combination, all from the same x0.

    NOTE ON WHERE EACH CURVE STARTS. "rule1" sets nu_0 = nu_bar/(K+1)^alpha so
    that the open-loop schedule nu_k = nu_0 (k+1)^alpha passes through the
    scale-aware base at the horizon K; that couples nu_0 to alpha by design,
    so FS-ALM/FS-P-ALM curves at different alpha legitimately start at different
    penalties. It is meaningless for classic ALM, whose penalty grows
    multiplicatively and never uses alpha -- left alone ALM silently inherits
    the solver's DEFAULT alpha=2. `alm_rho0` names ALM's rule explicitly.
    """
    problem = fs_palm_problem(inst, fs_palm, jittable=True)
    x0 = inst.x0()
    # Multipliers start at ZERO, not at a random draw; see fs_palm.solve's
    # mu0 documentation.

    common = dict(tol=tol, max_iter=max_iter, start_feas=True,
                  inner_solver=inner_solver, beta=0.5, adaptive_fp_tol=True,
                  verbosity=0)
    runs = []

    def go(label, **kw):
        sol = fs_palm.solve(problem, x0, **kw, **common)
        runs.append((label, sol))
        problem.reset_counters()
        x = np.asarray(sol.x)
        if verbose:
            print(f"  {label:16s} status={str(sol.solve_status):11s} "
                  f"nu0={float(sol.nu_hist[0]):.2e} "
                  f"f={inst.objective(x):.10f} "
                  f"gap={abs(inst.objective(x) - inst.f_star):.2e} "
                  f"viol={float(sol.total_infeas[-1]):.2e} "
                  f"evals={int(sol.grad_evals[-1]):>8d}", flush=True)

    if not only_proximal:
        for xi in xis:
            go(r"\texttt{ALM}-" + str(xi), use_proximal=False,
               phi_strategy="linear", xi1=float(xi), xi2=float(xi), no_reset=True,
               rho0=alm_rho0, nu0=alm_rho0, penalty_ramp=penalty_ramp)
        for a in alphas:
            go(r"\texttt{FS-ALM}-" + str(a), use_proximal=False, phi_strategy="pow",
               xi1=1.0, xi2=1.0, alpha=a, rho0=penalty_rule, nu0=penalty_rule,
               penalty_ramp=penalty_ramp)
    for a in alphas:
        go(r"\texttt{FS-P-ALM}-" + str(a), use_proximal=True, phi_strategy="pow",
           xi1=1.0, xi2=1.0, alpha=a, delta=delta, gamma0=gamma0,
           gamma_kappa=gamma_kappa, rho0=penalty_rule, nu0=penalty_rule,
           penalty_ramp=penalty_ramp)
    return runs


def _plot(xs, ys, labels, xlabel, ylabel, fname, loc="lower left",
          logx=True, logy=True, ncol=1, title=None):
    markers, colors, linestyles, ms = style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        plt.plot(x, y, label=lab, marker=markers[i % len(markers)],
                 markevery=0.1, markerfacecolor='none',
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 markersize=ms, linewidth=2.5)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if labels[0] is not None:
        plt.legend(fontsize=13, loc=loc, ncol=ncol)
    # Suppress MINOR tick labels on log axes. When the data spans little more
    # than a decade -- routine here, since these curves cover a few thousand
    # gradient evaluations -- matplotlib labels the minor ticks as well, and
    # "2x10^3 4x10^3 6x10^3 ..." overprints into an unreadable smear.
    if logx:
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if logy:
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # Set log-axis limits EXPLICITLY from the finite data. A runaway penalty
    # reaches ~1e290 here; matplotlib's autoscale then pads that ~290-decade
    # range by its default margin, the upper limit overflows past 1e308, and
    # the tick formatter dies with "cannot convert float infinity to integer".
    # The data are finite throughout -- it is the padded AXIS that overflows --
    # so capping the locator alone does not help; the limits have to be pinned.
    def _pin(axis, vals, is_log):
        v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in vals]) \
            if len(vals) else np.array([])
        v = v[np.isfinite(v)]
        if is_log:
            v = v[v > 0]
        if v.size == 0:
            return
        lo, hi = float(v.min()), float(v.max())
        if is_log:
            lo, hi = max(lo / 3.0, 1e-300), min(hi * 3.0, 1e300)
            axis.set_major_locator(mticker.LogLocator(numticks=8))
        else:
            pad = 0.05 * max(hi - lo, 1.0)
            lo, hi = lo - pad, hi + pad
        (ax.set_xlim if axis is ax.xaxis else ax.set_ylim)(lo, hi)

    _pin(ax.yaxis, ys, logy)
    _pin(ax.xaxis, xs, logx)
    plt.tight_layout()
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


def _legend_file(labels, fname):
    markers, colors, linestyles, ms = style()
    fig, ax = plt.subplots(figsize=(7, 1), dpi=300)
    handles = []
    for i, lab in enumerate(labels):
        hd, = ax.plot([], [], marker=markers[i % len(markers)],
                      markerfacecolor='none', color=colors[i % len(colors)],
                      linestyle=linestyles[i % len(linestyles)], markersize=ms,
                      linewidth=2.5)
        handles.append(hd)
    ax.legend(handles=handles, labels=labels, fontsize=12, loc='center',
              ncol=max(1, len(labels) // 2))
    ax.axis('off')
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {fname}")


def _penalty_history(runs):
    """Penalty trajectories, picking whichever of nu_k / rho_k this problem
    has. These QPs are inequality-constrained, so nu_k is the live one and
    rho_hist is a list of None."""
    def clean(hist):
        return np.asarray([float(v) for v in hist if v is not None])

    nu = [clean(s.nu_hist) for _, s in runs]
    rho = [clean(s.rho_hist) for _, s in runs]
    if all(len(v) for v in nu):
        return nu, r'$\nu_k$'
    if all(len(v) for v in rho):
        return rho, r'$\rho_k$'
    raise ValueError("neither nu_hist nor rho_hist is populated for every run")


def figure_for_instance(inst, args, show_labels, inner):
    tag = f"{inst.name}_{inner.lower()}"
    print(f"\ninstance {tag}: n={inst.n}, m={inst.m}, "
          f"kappa(Q)={inst.cond:.2e}, tol={args.tol:.0e}")
    x0 = np.asarray(inst.x0())
    print(f"  x0: violation={inst.violation(x0):.2e}, f(x0)={inst.objective(x0):.6f}, "
          f"f*={inst.f_star:.10f}"
          + ("" if inst.violation(x0) <= 1e-9 else "   (infeasible -> phase I runs)"))

    runs = run_variants(inst, args.alphas, args.xis, args.tol, args.max_iter,
                        delta=args.delta, gamma0=args.gamma0,
                        gamma_kappa=args.gamma_kappa,
                        alm_rho0=_as_rule(args.alm_rho0),
                        penalty_rule=_as_rule(args.penalty_rule),
                        penalty_ramp=args.penalty_ramp, seed=args.seed,
                        inner_solver=inner, only_proximal=args.only_proximal)

    labels = [lab for lab, _ in runs] if show_labels else [None] * len(runs)
    f0_gap = max(abs(inst.objective(x0) - inst.f_star), 1e-16)
    suffix = "" if show_labels else "_multi"

    _plot([s.grad_evals for _, s in runs],
          [np.maximum(np.asarray(s.total_infeas), 1e-12) for _, s in runs],
          labels, r'$\textbf{grad evals}$', r'$\textbf{total infeas}$',
          f"qp_sample_gradevals_infeas_{tag}{suffix}.pdf", ncol=2)

    _plot([s.grad_evals for _, s in runs],
          [np.maximum(np.abs(np.asarray(s.f_hist) - inst.f_star) / f0_gap,
                      1e-16) for _, s in runs],
          labels, r'$\textbf{grad evals}$',
          r'$\frac{|f_1(x^k) - f_1^\star|}{|f_1(x^0) - f_1^\star|}$',
          f"qp_sample_gradevals_subopt_{tag}{suffix}.pdf", ncol=2,
          title=rf'$\kappa(Q) = {inst.cond:.0e}$' if show_labels else None)

    pen, sym = _penalty_history(runs)
    _plot([np.arange(len(v)) for v in pen], pen, labels,
          r'$\textbf{n. iterations}$', sym,
          f"qp_sample_penalty_iter_{tag}{suffix}.pdf", loc="lower right",
          logx=False, ncol=2)
    return [lab for lab, _ in runs]


def main():
    ap = argparse.ArgumentParser(
        description="Sample convergence plots on convex QPs.")
    ap.add_argument("--source", choices=["synthetic", "cutest"],
                    default="synthetic",
                    help='"synthetic" (default) sweeps kappa(Q) with '
                         'everything else fixed; "cutest" uses '
                         "named Maros-Meszaros problems")
    ap.add_argument("--conds", type=float, nargs="+", default=None,
                    help=f"condition numbers for --source synthetic "
                         f"(default {' '.join(f'{c:.0e}' for c in CONDS)})")
    ap.add_argument("--problems", type=str, nargs="+", default=None,
                    help=f"MM problem names for --source cutest "
                         f"(default {' '.join(MM_COND_PROBLEMS)})")
    ap.add_argument("--n", type=int, default=100, help="synthetic: variables")
    ap.add_argument("--m", type=int, default=50,
                    help="synthetic: two-sided linear constraints (2m rows)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.01, 2, 8],
                    help="alpha for FS-ALM and FS-P-ALM (default 12)")
    ap.add_argument("--xis", type=int, nargs="+", default=[10],
                    help="xi for classic ALM (default 10)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--delta", type=float, default=1.0,
                    help="proximal delta")
    ap.add_argument("--gamma0", default="auto")
    ap.add_argument("--gamma-kappa", type=float, default=1e3,
                    help="gamma_0 = gamma_kappa/L_hat when --gamma0 auto; "
                         "the inverse fraction of local curvature the "
                         "proximal term contributes")
    ap.add_argument("--penalty-rule", default="rule3",
                    help='rho_0/nu_0 rule for FS-ALM and FS-P-ALM: "rule3" '
                         '(default, curvature-matched with a short ramp), '
                         '"rule1", "auto", or a number')
    ap.add_argument("--penalty-ramp", type=float, default=5.0,
                    help="rule3: iterations allowed to ramp up to the "
                         "curvature-matched penalty (default 5)")
    ap.add_argument("--alm-rho0", default="rule3",
                    help='rho_0/nu_0 rule for classic ALM only: "rule3" '
                         '(default), "rule1", "auto", or a number such as 1e-3')
    ap.add_argument("--only-proximal", action="store_true",
                    help="skip ALM-xi and FS-ALM, run only FS-P-ALM "
                         "(diagnostic sweeps that don't need the baselines)")
    ap.add_argument("--inner-solver", default="both",
                    choices=["PANOC", "ProxGrad", "both"],
                    help="inner solver: PANOC (quasi-Newton), ProxGrad "
                         "(proximal gradient with backtracking), or both "
                         "(default) which writes a separate set of figures "
                         "for each. "
                         "PANOC's L-BFGS directions absorb subproblem "
                         "ill-conditioning, so ProxGrad is the instrument "
                         "for seeing whether a penalty schedule "
                         "ill-conditions the inner problems")
    args = ap.parse_args()

    if args.source == "cutest":
        names = args.problems or list(MM_COND_PROBLEMS)
        insts = [load_cutest(nm, seed=args.seed) for nm in names]
    else:
        conds = args.conds or list(CONDS)
        insts = [make_instance(args.seed, n=args.n, m=args.m, cond=cd)
                 for cd in conds]

    # One parameter value per algorithm -> label the curves in place; several
    # -> the figure gets crowded, so labels move to a separate legend file
    # (the manuscript's layout for the multi-parameter figures).
    show_labels = len(args.alphas) == 1 and len(args.xis) == 1

    setup_matplotlib()
    for inner in inner_solvers(args.inner_solver):
        print(f"\n=== inner solver: {inner} " + "=" * 40)
        labels = None
        for inst in insts:
            try:
                labels = figure_for_instance(inst, args, show_labels, inner)
            except Exception as e:
                # One instance's failure (e.g. Phase I or a variant not
                # converging) should not cost every other instance's already-
                # completed work, or every instance still queued behind it --
                # this script has no CSV/resume mechanism, so previously this
                # was an uncaught crash that lost the rest of the run outright.
                print(f"  [FAILED] {inst.name} ({inner}): "
                      f"{type(e).__name__}: {e}"[:200], flush=True)
        if not show_labels and labels:
            _legend_file(labels, f"qp_sample_legend_{inner.lower()}.pdf")


if __name__ == "__main__":
    main()
