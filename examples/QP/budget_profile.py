"""Budget- and accuracy-resolved profiles from the recorded histories.

Two figures, both computed by post-processing the per-iteration histories
written by alpaqa_alm_bench.py. Neither requires re-running a solver, and
neither depends on wall-clock time.

WHY NOT WALL CLOCK. The two solvers cost very different amounts per gradient
evaluation, measured at 2.39e-3 s/eval for fs_palm against 6.09e-5 for alpaqa,
a factor of about 39, because fs_palm is Python and JAX while alpaqa is
compiled C++. Any shared wall-clock cap therefore hands one solver a much
larger effective budget than the other, which is the implementation gap the
gradient-evaluation metric exists to remove. Capping evaluations instead is
exact and needs no calibration.

EQUIVALENCE TO A HARD CAP. Requiring the target to be met at some history
entry with evals <= B is exactly what running with a hard budget of B
evaluations would have produced, provided the recorded history extends to at
least B evaluations (or the solver stopped earlier on its own). Runs whose
history ends before B without meeting the target are censored rather than
failed, and are reported separately rather than silently scored as failures.

    data profile     fraction of instances solved against the budget B,
                     at a fixed accuracy tau (More and Wild)
    accuracy profile fraction of instances reaching gap <= tau against tau,
                     at a fixed budget B

Usage:
    python budget_profile.py <tag> [more tags ...]
        e.g. python budget_profile.py alpha4_ag4 proxgrad_alpha4_ag4
"""
import csv
import json
import os
import sys

import numpy as np

SOLVERS = ["fs_palm", "alpaqa_alm"]
LABELS = [r"\texttt{FS-P-ALM}", r"\texttt{alpaqa ALM}"]
FTOL = 1e-1


def load(tag):
    """Per-instance final results and per-iteration histories for one tag."""
    csv_path = f"alpaqa_alm_bench_fair_{tag}.csv"
    hist_dir = f"alpaqa_alm_bench_history_{tag}"
    rows = list(csv.DictReader(open(csv_path)))
    insts = sorted({r["instance"] for r in rows})
    idx = {nm: i for i, nm in enumerate(insts)}

    fval = np.full((len(insts), 2), np.nan)
    viol = np.full((len(insts), 2), np.nan)
    for r in rows:
        if r["solver"] not in SOLVERS or not r["grad_evals"]:
            continue
        i, j = idx[r["instance"]], SOLVERS.index(r["solver"])
        fval[i, j] = float(r["objective"])
        viol[i, j] = float(r["violation"])

    # Reference objective per instance. No certified optimum exists for most
    # of these, so we use the best value either solver reached at a point that
    # is feasible to the tightest tolerance considered (Birgin and Martinez).
    ok = viol <= 1e-5
    f_best = np.nanmin(np.where(ok, fval, np.inf), axis=1)

    # A run is CENSORED only when it was stopped from outside, that is when
    # the wall-clock guard killed it. Those never reach save_history, since
    # that runs after solve returns, so they leave no history at all. A run
    # that ended early on its own stopping test is NOT censored, it is a
    # determinate result whose achieved accuracy we know, and scoring it as
    # censored would wrongly excuse every solver that simply stopped short.
    killed = set()
    for r in rows:
        if r["solver"] in SOLVERS and r["status"].startswith("ERROR"):
            killed.add((r["instance"], r["solver"]))

    hist = {}
    for name in insts:
        for j, solver in enumerate(SOLVERS):
            p = os.path.join(hist_dir, f"{name}__{solver}.json")
            if not os.path.exists(p):
                continue
            try:
                h = json.load(open(p))
            except json.JSONDecodeError:
                continue          # truncated by a crash, treated as missing
            hist[(name, solver)] = (
                np.asarray(h["evals"], dtype=float),
                np.asarray(h["f"], dtype=float),
                np.asarray(h["viol"], dtype=float),
            )
    return insts, f_best, hist, killed


def solved_within(hist, killed, insts, f_best, budget, tau):
    """(solved, censored) masks under a budget of `budget` evals at accuracy `tau`.

    solved[i, j]   the target was met at some entry with evals <= budget
    censored[i, j] no history, or the history stops short of `budget`
                   without meeting the target, so the run is undetermined
                   at this budget rather than known to have failed
    """
    n = len(insts)
    solved = np.zeros((n, 2), dtype=bool)
    censored = np.zeros((n, 2), dtype=bool)
    for i, name in enumerate(insts):
        for j, solver in enumerate(SOLVERS):
            h = hist.get((name, solver))
            if h is None:
                # no history means the run was killed before it could write
                # one, so its outcome at this budget is unknown
                censored[i, j] = True
                continue
            evals, f, viol = h
            if not np.isfinite(f_best[i]):
                # no solver ever reached a feasible point on this instance, so
                # there is no reference objective and the run is undetermined
                # rather than failed
                censored[i, j] = True
                continue
            with np.errstate(invalid="ignore"):
                gap = np.maximum(viol, np.abs(f - f_best[i])
                                 / max(1.0, abs(f_best[i])))
            gap = np.where(np.isfinite(gap), gap, np.inf)
            within = evals <= budget
            hit = within & (gap <= tau)
            if hit.any():
                solved[i, j] = True
            elif (name, solver) in killed and evals[-1] < budget:
                # cut off from outside before exhausting the budget, so we
                # cannot tell whether it would have got there
                censored[i, j] = True
    return solved, censored


def main(tags):
    from fs_palm.utils.plotting import setup_matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_matplotlib(font_scale=1.8)
    for tag in tags:
        insts, f_best, hist, killed = load(tag)
        n = len(insts)
        print(f"\n=== {tag} ({n} instances) ===")

        budgets = np.unique(np.round(np.logspace(1, 6.5, 200)).astype(int))
        tau = 1e-5
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        for j, (lab, style) in enumerate(zip(LABELS, ["-", "--"])):
            frac = []
            for B in budgets:
                s, _ = solved_within(hist, killed, insts, f_best, B, tau)
                frac.append(s[:, j].sum() / n)
            ax.plot(budgets, frac, style, label=lab, linewidth=2.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"gradient evaluation budget $B$")
        ax.set_ylabel(r"fraction solved at $\tau=10^{-5}$")
        ax.legend(fontsize=13, loc="lower right")
        fig.tight_layout()
        out = f"data_profile_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

        for B in [1e3, 1e4, 1e5, 1e6, 2.5e6]:
            s, c = solved_within(hist, killed, insts, f_best, B, tau)
            print(f"  B={B:9.0e}  fs_palm {s[:,0].sum():2d}/{n}"
                  f" (censored {c[:,0].sum():2d})   "
                  f"alpaqa {s[:,1].sum():2d}/{n} (censored {c[:,1].sum():2d})")

        # Accuracy profile, same machinery, budget held at alpaqa's own
        # iteration limit so neither solver is truncated by the budget, and
        # tau swept instead. The loose end of the sweep reproduces the
        # Birgin and Martinez reading, where landing near the best known
        # objective suffices, and the tight end asks for the accuracy the
        # stopping test in Definition 1 is stated at. Sweeping avoids having
        # to defend one threshold.
        B_full = 2.5e6
        taus = np.logspace(1, -10, 220)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        for j, (lab, style) in enumerate(zip(LABELS, ["-", "--"])):
            frac = []
            for t in taus:
                s, _ = solved_within(hist, killed, insts, f_best, B_full, t)
                frac.append(s[:, j].sum() / n)
            ax.plot(taus, frac, style, label=lab, linewidth=2.5)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"accuracy threshold $\tau$")
        ax.set_ylabel(r"fraction of instances reaching gap $\leq\tau$")
        ax.axvline(1e-5, color="gray", linestyle=":", linewidth=1)
        ax.legend(fontsize=13, loc="lower left")
        fig.tight_layout()
        out = f"accuracy_profile_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
        for t in [1e-1, 1e-3, 1e-5, 1e-8]:
            s, c = solved_within(hist, killed, insts, f_best, B_full, t)
            print(f"  tau={t:.0e}  fs_palm {s[:,0].sum():2d}/{n}"
                  f" (censored {c[:,0].sum():2d})   "
                  f"alpaqa {s[:,1].sum():2d}/{n} (censored {c[:,1].sum():2d})")


if __name__ == "__main__":
    sys.path.insert(0, "../../src")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
