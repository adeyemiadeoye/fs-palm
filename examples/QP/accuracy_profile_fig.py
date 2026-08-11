"""Side-by-side accuracy profiles, PANOC and ProxGrad, for the manuscript.

For each instance and solver we take the point the solver actually RETURNS and
measure the accuracy it attains,

    gap = max( constraint violation,  |f - f_best| / max(1,|f_best|) )

where f_best is the best objective either solver reached at a feasible point.
The curve is the fraction of instances with gap <= tau, swept over tau.

WHY THE RETURNED POINT rather than the best point visited. A user only ever
receives the final iterate, so that is the accuracy actually delivered. Scoring
the whole trajectory instead credits a solver for points it merely passed
through and did not stop at, which is the More and Wild data-profile convention
and answers a different question.

WHY SWEEP tau rather than fix it. The two methods do not stop on the same test
(ours additionally requires the proximal residual to fall below the tolerance),
so any single threshold privileges one of them. In particular the threshold
equal to the requested tolerance flatters whichever method overshoots it.
Sweeping shows the whole picture and commits to nothing.

COMMON SET. An instance is included only if BOTH solvers produced a result, so
a run that is missing for one of them is never scored as that solver's failure
while the other keeps the full denominator.

Usage:
    python accuracy_profile_fig.py <panoc_tag> <proxgrad_tag> <out.pdf>
"""
import csv
import os
import sys

import numpy as np

SOLVERS = ["pbalm", "alpaqa_alm"]
LABELS = [r"\texttt{FS-P-ALM}", r"\texttt{alpaqa ALM}"]
EPS_FEAS = 1e-5


def returned_gap(tag):
    """gap attained by the returned point, per (instance, solver)."""
    rows = list(csv.DictReader(open(f"alpaqa_alm_bench_fair_{tag}.csv")))
    insts = sorted({r["instance"] for r in rows})
    idx = {nm: i for i, nm in enumerate(insts)}
    fval = np.full((len(insts), 2), np.nan)
    viol = np.full((len(insts), 2), np.nan)
    have = np.zeros((len(insts), 2), dtype=bool)
    for r in rows:
        if r["solver"] not in SOLVERS or not r["grad_evals"]:
            continue
        i, j = idx[r["instance"]], SOLVERS.index(r["solver"])
        fval[i, j] = float(r["objective"])
        viol[i, j] = float(r["violation"])
        have[i, j] = True
    f_best = np.nanmin(np.where(viol <= EPS_FEAS, fval, np.inf), axis=1)
    with np.errstate(invalid="ignore"):
        gap = np.maximum(viol, np.abs(fval - f_best[:, None])
                         / np.maximum(1.0, np.abs(f_best[:, None])))
    gap = np.where(np.isfinite(gap), gap, np.inf)
    return insts, gap, have.all(axis=1)


def main(panoc_tag, proxgrad_tag, out):
    sys.path.insert(0, "../../src")
    from pbalm.utils.plotting import setup_matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = np.logspace(1, -10, 240)
    setup_matplotlib(font_scale=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    for ax, tag, title in ((axes[0], panoc_tag, r"\texttt{PANOC}"),
                           (axes[1], proxgrad_tag, r"\texttt{ProxGrad}")):
        insts, gap, both = returned_gap(tag)
        n = int(both.sum())
        for j, (lab, style) in enumerate(zip(LABELS, ["-", "--"])):
            frac = [(gap[both, j] <= t).sum() / n for t in taus]
            ax.plot(taus, frac, style, linewidth=2.5, label=lab)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlim(10, 1e-5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"accuracy threshold $\tau$")
        ax.set_ylabel(r"fraction with gap $\leq \tau$")
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12, loc="lower left")
        print(f"{tag}: common set {n} of {len(insts)}")
        for t in (1e-1, 1e-3, 1e-5, 1e-6, 1e-8):
            a = int((gap[both, 0] <= t).sum())
            b = int((gap[both, 1] <= t).sum())
            print(f"   tau={t:.0e}  FS-P-ALM {a:2d}/{n}   alpaqa {b:2d}/{n}")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
