"""Illustrative single-instance plots for nonconvex basis pursuit.

Two modes:

  --mode multi   one instance, p=400, n=1024, sweeping the parameters: ALM
                 at several xi, FS-ALM and FS-P-ALM at several alpha. Curves are
                 unlabelled and a separate legend file is written.

  --mode single  alpha = 4 and xi = 4 fixed, one column per problem size,
                 over (p, n) in {(200,512), (400,1024), (600,2048)}. Each
                 size gets its own pair of plots.

Conventions:

    ALM-xi        xi > 1, phi(k) = 0, no Step-1 reset      (classic ALM)
    FS-ALM-alpha    xi = 1, phi(k) = (k+1)^alpha
    FS-P-ALM-alpha  as FS-ALM plus the proximal term

xi = 1 always for FS-ALM and FS-P-ALM -- that is what defines them (Table 1);
the penalty grows through the schedule phi, not a multiplicative safeguard.
Only classic ALM uses xi > 1. ALM is a heuristic here and is NOT supported by
the paper's theory; it is retained as an ablation baseline.

Outputs (PDF, house style), per size tag:
    bp_sample_gradevals_infeas_<tag>.pdf   grad evals vs total infeasibility
    bp_sample_gradevals_subopt_<tag>.pdf   grad evals vs objective gap
    bp_sample_penalty_iter_<tag>.pdf       penalty rho_k vs iteration
    bp_sample_legend.pdf                   shared legend (multi mode only)

The suboptimality gap here is measured against the
TRUE optimum f* = ||z*||_1, not against the best value any solver found: the
planted signal is the l1 minimiser (checked per instance with an LP), so the
vertical axis is a genuine suboptimality and not a relative ranking.

All runs start from the same strictly feasible x0 (closed form, see
bp_problem.Instance.x0) so no phase I solve is involved and the plots measure
the outer algorithm only.

Run from this directory:

    python bp_sample.py                             # one column per size
    python bp_sample.py --mode multi                # parameter sweep, one size
    python bp_sample.py --sizes 200,512 400,1024    # pick the sizes
    python bp_sample.py --mode multi --alphas 4 6 9 12 --xis 2 4 7 10
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
from bp_problem import make_instance, fs_palm_problem


def style():
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'p', 'H']
    colors = np.concatenate([plt.cm.Dark2(np.linspace(0, 1, plt.cm.Dark2.N)),
                             plt.cm.Set1(np.linspace(0, 1, plt.cm.Set1.N))])
    linestyles = [(5, (10, 3)), (0, (5, 1)), (0, (3, 1, 1, 1)),
                  (0, (3, 1, 1, 1, 1, 1)), '--', '-', '-.', ':']
    return markers, colors, linestyles, 9


def run_variants(inst, alphas, xis, tol, max_iter, c0=1.0, delta="auto",
                 gamma0="auto", seed=1234, verbose=True, alm_rho0="rule3",
                 inner_solver="PANOC", beta=0.5):
    """One curve per algorithm/parameter combination, all from the same x0.

    NOTE ON WHERE EACH CURVE STARTS. The default penalty rule "rule3" is
    curvature matched (see rule3_penalty0) and, unlike "rule1", detects
    phi==0 (classic ALM's multiplicative-only growth) and does not divide by
    the ramp correction in that case -- so ALM starts AT its target rather
    than below it, rather than needing a separate rule. `alm_rho0` still
    names ALM's rule explicitly, in case a different choice is wanted for it
    specifically. Measured on this instance (p=400, n=1024), classic ALM-4,
    PANOC, same converged solution either way: rule1 costs 2941 gradient
    evaluations, rule3 costs 385.
    """
    problem = fs_palm_problem(inst, fs_palm, jittable=True)
    x0 = inst.x0(c=c0)                  # strictly feasible by construction
    # Multipliers start at ZERO, not at a random draw. The earlier random
    # start is not scale invariant (lambda carries units [f]/[c]) and, since it
    # enters the augmented Lagrangian whose curvature sets gamma_0, it made
    # gamma_0 differ between two problems that are the same up to a rescaling
    # of the constraints. It was also measurably more expensive on every
    # instance tested. `seed` is kept in the signature for the instance draw.

    # rho0/nu0 stay at the solver default ("rule1"), gamma0/delta at "auto".
    # Hard-coding them would be wrong for a figure that sweeps alpha: rho_0
    # and alpha are coupled through rho_k = rho_0 (k+1)^alpha, so a value
    # tuned at one alpha starves or explodes the others.
    common = dict(tol=tol, max_iter=max_iter, start_feas=True,
                  inner_solver=inner_solver, beta=beta, adaptive_fp_tol=True,
                  verbosity=0)
    runs = []

    def go(label, **kw):
        sol = fs_palm.solve(problem, x0, **kw, **common)
        runs.append((label, sol))
        problem.reset_counters()
        x = np.asarray(sol.x)
        if verbose:
            # rho_0 is printed because the curves legitimately start at
            # different penalties (see the docstring); silent disagreement
            # here reads as a bug in the plot.
            print(f"  {label:14s} status={str(sol.solve_status):11s} "
                  f"rho0={float(sol.rho_hist[0]):.2e} "
                  f"f={float(sol.f_hist[-1]):.8f} "
                  f"gap={abs(float(sol.f_hist[-1]) - inst.f_star):.2e} "
                  f"viol={float(sol.total_infeas[-1]):.2e} "
                  f"nnz={inst.nnz(x):>4d}/{inst.k} "
                  f"|z-z*|={inst.recovery_error(x):.1e} "
                  f"evals={int(sol.grad_evals[-1]):>8d}", flush=True)

    for xi in xis:
        go(r"\texttt{ALM}-" + str(xi), use_proximal=False,
           phi_strategy="linear", xi1=float(xi), xi2=float(xi), no_reset=True,
           rho0=alm_rho0, nu0=alm_rho0)
    for a in alphas:
        go(r"\texttt{FS-ALM}-" + str(a), use_proximal=False, phi_strategy="pow",
           xi1=1.0, xi2=1.0, alpha=a)
    for a in alphas:
        go(r"\texttt{FS-P-ALM}-" + str(a), use_proximal=True, phi_strategy="pow",
           xi1=1.0, xi2=1.0, alpha=a, delta=delta, gamma0=gamma0)
    return runs


def _plot(xs, ys, labels, xlabel, ylabel, fname, loc="lower left",
          logx=True, logy=True, ncol=1, title=None, sci_x=True):
    markers, colors, linestyles, ms = style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        # Drop non-finite and (on a log axis) non-positive points before
        # plotting. A run that diverges can leave inf/nan in its history --
        # here an FS-P-ALM run at alpha=8 whose penalty starts at 5e-9 never
        # moved off x^0 -- and matplotlib then fails deep inside tight_layout
        # with "cannot convert float infinity to integer", which names neither
        # the curve nor the quantity. Better to drop the unplottable points and
        # say so than to lose the whole figure.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        keep = np.isfinite(x) & np.isfinite(y)
        if logy:
            keep &= y > 0
        if logx:
            keep &= x > 0
        if not keep.all():
            print(f"    note: {lab}: dropped {int((~keep).sum())} of {n} "
                  f"non-plottable points from {fname}")
        if not keep.any():
            continue
        plt.plot(x[keep], y[keep], label=lab, marker=markers[i % len(markers)],
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
    if not logx and sci_x:
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
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


_RULES = ("rule1", "rule3", "auto")


def _as_rule(v):
    """Accept a named rule or a numeric value from the CLI.

    Kept in sync with the strategies fs_palm.solve accepts; listing them in one
    place here means a rule added to the solver does not silently become
    "not a number" at the command line.
    """
    if isinstance(v, str) and v not in _RULES:
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"expected one of {_RULES} or a number, "
                             f"got {v!r}")
    return v


def _penalty_history(runs):
    """Penalty trajectories, picking whichever of rho_k / nu_k this problem has.

    rho_hist tracks the equality penalty and nu_hist the inequality one; the
    unused one is a list of None, so it cannot simply be plotted. Basis
    pursuit is equality-only and therefore uses rho_k.
    """
    def clean(hist):
        return np.asarray([float(v) for v in hist if v is not None])

    rho = [clean(s.rho_hist) for _, s in runs]
    nu = [clean(s.nu_hist) for _, s in runs]
    if all(len(v) for v in rho):
        return rho, r'$\rho_k$'
    if all(len(v) for v in nu):
        return nu, r'$\nu_k$'
    raise ValueError("neither rho_hist nor nu_hist is populated for every run")


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


def figure_for_size(p, n, k, args, show_labels, seed, inner):
    inst = make_instance(seed, p=p, n=n, k=k)
    tag = f"bp_p{p}_n{n}_k{k}_{inner.lower()}"
    print(f"\ninstance {inst.name}: p={p}, n={n}, k={k}, tol={args.tol:.0e}")

    if args.verify:
        ok, l1, err = inst.verify_optimum()
        verdict = ("z* IS the l1 optimum" if ok else
                   "MISMATCH -- f* is NOT the optimal value, so the "
                   "suboptimality axis below is meaningless")
        print(f"  LP check: ||z_lp||_1={l1:.8f} vs ||z*||_1={inst.f_star:.8f} "
              f"({verdict}), ||z_lp-z*||_inf={err:.1e}")
    x0 = inst.x0(c=args.c0)
    print(f"  x0: ||h(x0)||_inf={inst.violation(x0):.2e}, "
          f"f(x0)={inst.objective(x0):.3f}, f*={inst.f_star:.8f}")

    runs = run_variants(inst, args.alphas, args.xis, args.tol, args.max_iter,
                        c0=args.c0, delta=args.delta, gamma0=args.gamma0,
                        seed=seed, alm_rho0=_as_rule(args.alm_rho0),
                        inner_solver=inner, beta=args.beta)

    labels = [lab for lab, _ in runs] if show_labels else [None] * len(runs)
    f0_gap = max(abs(inst.objective(x0) - inst.f_star), 1e-16)
    suffix = "" if show_labels else "_multi"
    # keep beta out of the name at the reported default, so the existing
    # figure names keep resolving to the existing figures
    if abs(getattr(args, "beta", 0.5) - 0.5) > 1e-12:
        suffix += f"_beta{args.beta:g}"

    _plot([s.grad_evals for _, s in runs],
          [np.maximum(np.asarray(s.total_infeas), 1e-12) for _, s in runs],
          labels, r'$\textbf{grad evals}$', r'$\textbf{total infeas}$',
          f"bp_sample_gradevals_infeas_{tag}{suffix}.pdf", ncol=2)

    _plot([s.grad_evals for _, s in runs],
          [np.maximum(np.abs(np.asarray(s.f_hist) - inst.f_star) / f0_gap,
                      1e-16) for _, s in runs],
          labels, r'$\textbf{grad evals}$',
          r'$\frac{|f_1(x^k) - f_1^\star|}{|f_1(x^0) - f_1^\star|}$',
          f"bp_sample_gradevals_subopt_{tag}{suffix}.pdf", ncol=2,
          title=rf'$p = {p}, n = {n}$' if show_labels else None)

    # This problem has ONLY equality constraints, so the penalty that moves is
    # rho_k (attached to h); nu_hist is a list of None here because there are
    # no inequality constraints. Not every problem in this family populates
    # the same history field, hence the selection below rather than a
    # hard-coded choice.
    pen, sym = _penalty_history(runs)
    # Non-fatal. A runaway penalty reaches ~1e290 on this family, and a log
    # axis spanning ~290 decades still breaks matplotlib's tick formatter even
    # with the limits pinned. This panel is diagnostic, so losing it must not
    # cost the whole figure set -- the cost/accuracy panels above are the ones
    # the manuscript uses.
    try:
        _plot([np.arange(len(v)) for v in pen], pen, labels,
              r'$\textbf{n. iterations}$', sym,
              f"bp_sample_penalty_iter_{tag}{suffix}.pdf", loc="lower right",
              logx=False, ncol=2)
    except (OverflowError, ValueError) as exc:
        print(f"    note: penalty panel skipped for {tag}{suffix} "
              f"({type(exc).__name__}: penalty range too wide to render)",
              flush=True)
    return [lab for lab, _ in runs]


def main():
    ap = argparse.ArgumentParser(
        description="Sample convergence plots on nonconvex basis pursuit.")
    ap.add_argument("--mode", choices=["single", "multi"], default="single",
                    help='"single": alpha and xi fixed, one column '
                         'per size. "multi": one size, parameters '
                         'swept, curves unlabelled plus a legend file.')
    ap.add_argument("--sizes", type=str, nargs="+", default=None,
                    help='problem shapes as "p,n" (default: 200,512 400,1024 '
                         '600,2048 for single mode; 400,1024 for multi)')
    ap.add_argument("--k", type=int, default=10, help="number of nonzeros")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="alpha for FS-ALM/FS-P-ALM (default 4 for single mode, "
                         "4 6 9 12 for multi)")
    ap.add_argument("--xis", type=int, nargs="+", default=None,
                    help="xi for classic ALM (default 4 for single mode, "
                         "2 4 7 10 for multi)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="sufficient-progress fraction in the penalty test "
                         "||h(x^k)|| > beta ||h(x^(k-1))||. 0.5 is ALGENCAN's "
                         "default; larger means the test fires less often "
                         "and the penalty is raised less")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--c0", type=float, default=1.0,
                    help="offset of the feasible start; must be > 0 (c=0 "
                         "gives invariant zero coordinates)")
    ap.add_argument("--delta", default="auto")
    ap.add_argument("--gamma0", default="auto")
    ap.add_argument("--alm-rho0", default="rule3",
                    help='rho_0 rule for classic ALM only: "rule3" (default, '
                         'curvature matched), "rule1", "auto", or a number. '
                         "ALM's penalty is multiplicative and never uses "
                         "alpha, so without this it inherits the solver's "
                         "default alpha=2 -- see run_variants.__doc__")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="LP-check that z* is the l1 optimum (default on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
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

    multi = args.mode == "multi"
    if args.alphas is None:
        args.alphas = [4, 6, 9, 12] if multi else [4]
    if args.xis is None:
        args.xis = [2, 4, 7, 10] if multi else [4]
    if args.sizes is None:
        args.sizes = ["400,1024"] if multi else ["200,512", "400,1024",
                                                 "600,2048"]
    sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]

    setup_matplotlib()
    for inner in inner_solvers(args.inner_solver):
        print(f"\n=== inner solver: {inner} " + "=" * 40)
        labels = None
        for p, n in sizes:
            labels = figure_for_size(p, n, args.k, args, show_labels=not multi,
                                     seed=args.seed, inner=inner)
        if multi and labels:
            _legend_file(labels, f"bp_sample_legend_{inner.lower()}.pdf")


if __name__ == "__main__":
    main()
