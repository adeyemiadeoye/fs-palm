"""Two-panel Dolan and More performance profile on gradient evaluations.

COST IS THE FAIR COST, not evaluations to self-reported termination. The two
methods do not stop on the same test, ours additionally requiring the proximal
residual to fall below the tolerance, so "work until each says it is done" is
not a like-for-like quantity. Instead we read each method's recorded history
and take the evaluations at which it FIRST reaches one common, externally
defined target, namely feasible to EPS and within FTOL of the best objective
either method found at a feasible point.

WHY COST RATHER THAN ACCURACY. Attained accuracy is capped by the tolerance the
solvers are given: a method asked for 1e-5 stops on reaching it, so anything
tighter measures incidental overshoot rather than capability, and the profile
collapses just below the requested tolerance for both methods. Cost has no such
ceiling, since a method spends whatever evaluations it needs.

Usage:
    python cost_profile_fig.py <panoc_tag> <proxgrad_tag> <out.pdf>
"""
import csv
import json
import os
import sys

import numpy as np

SOLVERS = ["pbalm", "alpaqa_alm"]
LABELS = [r"\texttt{FS-P-ALM}", r"\texttt{alpaqa ALM}"]
EPS, FTOL = 1e-5, 1e-1


def fair_cost(tag):
    rows = list(csv.DictReader(open(f"alpaqa_alm_bench_fair_{tag}.csv")))
    insts = sorted({r["instance"] for r in rows})
    idx = {m: i for i, m in enumerate(insts)}
    fv = np.full((len(insts), 2), np.nan)
    vl = np.full((len(insts), 2), np.nan)
    for r in rows:
        if r["solver"] not in SOLVERS or not r["grad_evals"]:
            continue
        i, j = idx[r["instance"]], SOLVERS.index(r["solver"])
        fv[i, j] = float(r["objective"])
        vl[i, j] = float(r["violation"])
    fb = np.nanmin(np.where(vl <= EPS, fv, np.inf), axis=1)
    succ = (vl <= EPS) & (np.abs(fv - fb[:, None])
                          <= FTOL * np.maximum(1, np.abs(fb[:, None])))
    cost = np.full((len(insts), 2), np.inf)
    for i, nm in enumerate(insts):
        for j, s in enumerate(SOLVERS):
            if not succ[i, j]:
                continue
            p = f"alpaqa_alm_bench_history_{tag}/{nm}__{s}.json"
            if not os.path.exists(p):
                continue
            try:
                h = json.load(open(p))
            except Exception:
                continue          # truncated history, treated as missing
            e = np.asarray(h["evals"], float)
            f = np.asarray(h["f"], float)
            v = np.asarray(h["viol"], float)
            hit = (v <= EPS) & (np.abs(f - fb[i]) <= FTOL * max(1, abs(fb[i])))
            if hit.any():
                # floored at 1: a starting point that already meets the target
                # is a real result but breaks a log-ratio profile
                cost[i, j] = max(float(e[np.argmax(hit)]), 1.0)
    return insts, cost, succ


def main(panoc_tag, proxgrad_tag, out):
    sys.path.insert(0, "../../src")
    from pbalm.utils.plotting import setup_matplotlib
    from pbalm.utils import perfprof
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_matplotlib(font_scale=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, tag, title in ((axes[0], panoc_tag, r"\texttt{PANOC}"),
                           (axes[1], proxgrad_tag, r"\texttt{ProxGrad}")):
        insts, cost, succ = fair_cost(tag)
        perfprof.pairwise_profiles(cost, succ, LABELS,
                                   r"$\tau$ ($\times$ best grad evals)", ax=ax)
        ax.set_title(title, fontsize=14)
        both = succ.all(axis=1)
        r = cost[both, 0] / cost[both, 1]
        print(f"{tag}: both solve {both.sum()}/{len(insts)}, "
              f"we cheaper on {int((r<1).sum())}, alpaqa on {int((r>1).sum())}, "
              f"median ratio {np.median(r):.2f}")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
