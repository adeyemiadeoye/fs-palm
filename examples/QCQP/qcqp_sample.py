"""Illustrative single-instance plots for the sparse nonconvex QCQP, in the
same style as basis_pursuit.py / all_MM_QPs.py.

Compares, on one representative instance:

    ALM-xi        xi = 10, phi(k) = 0, no Step-1 reset   (classic ALM)
    BALM-alpha    xi = 1,  phi(k) = (k+1)^alpha          alpha in {2, 4, 8}
    P-BALM-alpha  as BALM plus the proximal term         alpha in {2, 4, 8}

xi = 1 always for BALM and P-BALM -- that is what defines them (Table 1);
the penalty grows through the schedule phi, not a multiplicative safeguard.
Only classic ALM uses xi > 1, at the value standard in the literature
(Bertsekas; ALGENCAN; LANCELOT) and found best in the manuscript's own
experiments. One xi is plotted deliberately: more curves make the figure
unreadable. Note the suffix differs by algorithm: ALM-10 is xi=10, while
BALM-4 / P-BALM-4 are alpha=4.

Outputs (PDF, house style):
    qcqp_sample_gradevals_infeas_<tag>.pdf   grad evals vs total infeasibility
    qcqp_sample_gradevals_subopt_<tag>.pdf   grad evals vs objective gap
    qcqp_sample_nu_iter_<tag>.pdf            penalty nu_k vs iteration

The problem is NONCONVEX, so there is no global reference value. The gap is
therefore measured against the best objective found by any of the runs
(including IPOPT), which is what Birgin & Martinez's criterion (63) does as
well. Solvers reaching different local minima are reported explicitly rather
than hidden, since a smaller "gap" is meaningless if the points differ.

Run from this directory:

    python qcqp_sample.py                      # defaults n=50, m=10, seed=0
    python qcqp_sample.py --n 80 --m 20
    python qcqp_sample.py --c 0.5              # sparser solutions
    python qcqp_sample.py --alphas 2 4 8 --xi 10 --tol 1e-8
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")     # only writes PDFs; the interactive backend
                          # crashes at teardown in this environment
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# Prefer this repo's solver over any installed pbalm. A pip-installed pbalm
# in the same environment would otherwise shadow it and the script would
# benchmark the published package instead of the working copy.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "src")
if _os.path.isdir(_os.path.join(_SRC, "pbalm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import pbalm

from pbalm.utils.plotting import setup_matplotlib, inner_solvers
from qcqp_problem import make_instance, pbalm_problem, solve_ipopt


def style():
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']
    colors = np.concatenate([plt.cm.Dark2(np.linspace(0, 1, plt.cm.Dark2.N)),
                             plt.cm.Set1(np.linspace(0, 1, plt.cm.Set1.N))])
    linestyles = [(5, (10, 3)), (0, (5, 1)), (0, (3, 1, 1, 1)),
                  (0, (3, 1, 1, 1, 1, 1)), '--', '-', '-.', ':']
    return markers, colors, linestyles, 9


class ObjectiveRecorder:
    """Records the full objective F = f1 + lmbda||x||_1 per iteration.

    Solution.f_hist stores f1 only, so it omits the l1 term; here that term
    is a genuine part of the objective (unlike the portfolio problem, where
    x >= 0 makes it linear), and f1 alone can sit below the optimal F.
    """

    def __init__(self, inst):
        self.inst, self.xs = inst, []

    def __call__(self, **kw):
        self.xs.append(np.asarray(kw["x"], dtype=float))

    def aligned_objectives(self, sol):
        later = self.xs[1:] + [np.asarray(sol.x, dtype=float)]
        n = len(sol.f_hist)
        if len(later) < n:
            later += [later[-1]] * (n - len(later))
        return np.array([self.inst.objective(x) for x in later[:n]])


def run_variants(inst, alphas, xi, tol, max_iter, delta="auto", gamma0="auto",
                 inner_solver="PANOC", beta=0.5):
    """One curve per algorithm/parameter combination, all from x0 = 0."""
    problem = pbalm_problem(inst, pbalm, jittable=True)
    x0 = inst.x0()                      # strictly feasible by construction
    # rho0/nu0 are left at the solver default ("rule3", curvature matched);
    # gamma0/delta default to "auto". Hard-coding any of these would be wrong
    # for a figure that sweeps alpha: rho_0 and alpha are not independent
    # (rho_k = rho_0 (k+1)^alpha), so a value tuned for one alpha starves or
    # explodes the others. rule3 reaches its target after `penalty_ramp`
    # iterations for every alpha, sidestepping that coupling; with the
    # previously hard-coded rho0=1e-3, large alpha failed outright (relative
    # objective gap 0.5-0.75 after >1e6 gradient evaluations).
    common = dict(tol=tol, max_iter=max_iter, start_feas=True,
                  inner_solver=inner_solver, beta=beta, adaptive_fp_tol=True,
                  verbosity=0)
    runs = []

    def go(label, **kw):
        rec = ObjectiveRecorder(inst)
        problem.callback = rec
        sol = pbalm.solve(problem, x0, **kw, **common)
        F = rec.aligned_objectives(sol)
        runs.append((label, sol, F))
        problem.callback = None
        problem.reset_counters()
        x = np.asarray(sol.x)
        print(f"  {label:24s} status={str(sol.solve_status):11s} "
              f"F={F[-1]:+.8e} viol={sol.total_infeas[-1]:.2e} "
              f"nnz={inst.nnz(x):>3d}/{inst.n} "
              f"grad_evals={sol.grad_evals[-1]:>7d}", flush=True)

    go(r"\texttt{ALM}-" + f"{xi:g}", use_proximal=False, phi_strategy="linear",
       xi1=xi, xi2=xi, no_reset=True)
    for a in alphas:
        go(r"\texttt{FS-ALM}-" + f"{a:g}", use_proximal=False, phi_strategy="pow",
           xi1=1.0, xi2=1.0, alpha=a)
    for a in alphas:
        go(r"\texttt{FS-P-ALM}-" + f"{a:g}", use_proximal=True, phi_strategy="pow",
           xi1=1.0, xi2=1.0, alpha=a, delta=delta, gamma0=gamma0)
    return runs


def _plot(xs, ys, labels, xlabel, ylabel, fname, loc="lower left",
          logx=True, logy=True, ncol=1):
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
    """Write the legend as its own file, separate from the panel axes.

    The QCQP panels carry seven curves; an inline legend covers the part of
    the axes where the curves separate. A shared legend strip above the row
    lets the figure be read with one key.
    """
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


def main():
    ap = argparse.ArgumentParser(
        description="Sample convergence plots on a sparse nonconvex QCQP.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--m", type=int, default=10, help="number of quadratic "
                    "inequality constraints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c", type=float, default=0.5,
                    help="lmbda = c*||q0||_inf; larger c gives sparser solutions")
    # float, not int: Assumption II only requires alpha > 1, and the measured
    # optimum on this family is at the bottom of the admissible range
    # (alpha = 1.01-1.25 beats alpha = 2 on every instance under both inner
    # solvers, and alpha >= 4 fails outright under ProxGrad on 3 of 4). An int
    # type silently excluded exactly the values that turn out to matter.
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.01, 2, 8],
                    help="alpha for FS-ALM and FS-P-ALM (xi is always 1)")
    ap.add_argument("--xi", type=int, default=10, help="xi for classic ALM only")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="sufficient-progress fraction in the penalty test "
                         "||h(x^k)|| > beta ||h(x^(k-1))||. 0.5 is ALGENCAN's "
                         "default; larger means the test fires less often "
                         "and the penalty is raised less")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--delta", default="auto",
                    help='proximal delta; "auto" (default) sets it with gamma0 via proximal_init')
    ap.add_argument("--gamma0", default="auto",
                    help='initial proximal parameter; "auto" (default) uses kappa/L_hat')
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
    a = ap.parse_args()

    inst = make_instance(a.seed, n=a.n, m=a.m, c=a.c)
    ev = np.linalg.eigvalsh(np.asarray(inst.P0))
    print(f"instance {inst.name}: n={inst.n} m={inst.m} lmbda={inst.lmbda:.4f} "
          f"(c={inst.c}), tol={a.tol:.0e}")
    print(f"  P0 eigenvalues in [{ev.min():.3f}, {ev.max():.3f}] "
          f"-> {'nonconvex' if ev.min() < 0 else 'convex'}")
    x_ip, info = solve_ipopt(inst)
    F_ip = inst.objective(x_ip)
    print(f"  IPOPT local solution F={F_ip:+.8e} "
          f"(violation {inst.violation(x_ip):.2e}, nnz {inst.nnz(x_ip)}/{inst.n})\n")

    setup_matplotlib()
    for inner in inner_solvers(a.inner_solver):
        print(f"\n--- inner solver: {inner} " + "-" * 40)
        runs = run_variants(inst, a.alphas, a.xi, a.tol, a.max_iter,
                            delta=a.delta, gamma0=a.gamma0,
                            inner_solver=inner, beta=a.beta)
        _figures(inst, runs, F_ip, inner, beta=a.beta)


def _figures(inst, runs, F_ip, inner, beta=0.5):
    """Write one set of figures for a given inner solver.

    The inner solver is part of the filename because the two are NOT
    interchangeable views of the same experiment: PANOC is quasi-Newton and its
    L-BFGS directions absorb subproblem ill-conditioning, while the proximal
    gradient method's cost reflects that conditioning directly. Overwriting one
    with the other would silently mix them.
    """
    # nonconvex: no global optimum, so reference = best value anyone found
    F_ref = min([F[-1] for _, _, F in runs] + [F_ip])
    spread = max(F[-1] for _, _, F in runs) - min(F[-1] for _, _, F in runs)
    print(f"\n  reference F* (best found) = {F_ref:+.8e}")
    if spread > 1e-6 * max(1.0, abs(F_ref)):
        print(f"  NOTE: final objectives differ by {spread:.2e} -- the runs did "
              f"not all reach the same local minimum, so the gap curves below "
              f"are not directly comparable between them.")

    labels = [lab for lab, _, _ in runs]
    tag = f"qcqp_n{inst.n}_m{inst.m}_s{inst.seed}_{inner.lower()}"
    if abs(beta - 0.5) > 1e-12:
        tag += f"_beta{beta:g}"
    print()

    _plot([s.grad_evals for _, s, _ in runs],
          [np.maximum(np.asarray(s.total_infeas), 1e-12) for _, s, _ in runs],
          [None] * len(labels), r'$\textbf{grad evals}$', r'$\textbf{total infeas}$',
          f"qcqp_sample_gradevals_infeas_{tag}.pdf", ncol=2)

    f0_gap = max(abs(runs[0][2][0] - F_ref), 1e-16)
    _plot([s.grad_evals for _, s, _ in runs],
          [np.maximum(np.abs(F - F_ref) / f0_gap, 1e-16) for _, _, F in runs],
          [None] * len(labels), r'$\textbf{grad evals}$', r'$\textbf{objective gap}$',
          f"qcqp_sample_gradevals_subopt_{tag}.pdf", ncol=2)

    _legend_file(labels, f"qcqp_sample_legend_{inner.lower()}.pdf")

    _plot([np.arange(len(s.nu_hist)) for _, s, _ in runs],
          [np.asarray([float(v) for v in s.nu_hist]) for _, s, _ in runs],
          labels, r'$\textbf{n. iterations}$', r'$\nu_k$',
          f"qcqp_sample_nu_iter_{tag}.pdf", loc="lower right", logx=False,
          ncol=2)


if __name__ == "__main__":
    main()
