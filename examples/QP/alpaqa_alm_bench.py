"""Performance profile: fs_palm (FS-P-ALM, alpha=1.01) vs alpaqa's own
ALMSolver -- a genuinely safeguarded augmented Lagrangian method (explicit
multiplier projection onto a bounded set, classical multiplicative penalty
growth; see alpaqa.ALMParams: max_multiplier, penalty_update_factor,
rel_penalty_increase_threshold) -- on the Maros-Meszaros/CUTEst QP problems
decoded under ~/opt/CUTEST/compiled_QP_MM.

Both solvers run in-process (no cross-language bridge: alpaqa's ALM drives
the SAME CUTEstProblem/PANOC machinery fs_palm's own inner solver uses), so
this avoids the language-bridge confound a cross-language comparison would
have had. Confirmed correct on GENHS28 against the already-known optimum
before this script was written.

ROBUSTNESS. Each solver call runs in its own spawned subprocess
(fs_palm.utils.subprocess_timeout.run_with_timeout) with a hard wall-clock cap
(SOLVER_TIMEOUT): if either solver hangs, the subprocess is killed and the
attempt is recorded as an ordinary failure, and the run moves on to the next
instance -- it does not stall the whole batch (this is exactly the failure
mode that cost hours on an ill-conditioned CUTEst/DUALC1 instance earlier). Results are
appended to CSV one row at a time, flushed immediately, and already-recorded
(instance, solver) pairs are skipped on a rerun -- if the process or the
machine dies mid-run, only the one in-flight instance is lost; rerunning the
same command resumes from the CSV rather than starting over.

Usage:
    python alpaqa_alm_bench.py                 # run all problems, then plot
    python alpaqa_alm_bench.py run              # run only (resumable)
    python alpaqa_alm_bench.py plot              # re-plot from the existing CSV
"""
import csv
import json
import os
import sys
import time

import numpy as np
# Prefer this repo's solver over any installed fs_palm.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "src")
if _os.path.isdir(_os.path.join(_SRC, "fs_palm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import fs_palm
import alpaqa as pa
from qp_problem import load_cutest, fs_palm_problem
from fs_palm.utils.subprocess_timeout import run_with_timeout

CUTEST_DIR = os.environ.get(
    "CUTEST_QP_DIR", os.path.expanduser("~/opt/CUTEST/compiled_QP_MM"))

# 53 problems decoded (44 convex Maros-Meszaros QPs + all 9 NCVXQP instances,
# kept deliberately: this is a nonconvex-focused paper, and showing the method
# holds on genuinely nonconvex QPs too is a point in its favour, not a
# complication).
PROBLEMS = """GENHS28 DUALC1 DUALC2 DUALC5 DUALC8 LOTSCHD HS118 HS21 HS21MOD HS268
HS35 HS35I HS35MOD HS51 HS52 HS53 HS76 HS76I KSIP S268 TAME ZECEVIC2 DUAL1
DUAL2 DUAL3 DUAL4 GOULDQP1 PRIMALC1 PRIMALC2 PRIMALC5 PRIMALC8 QPCBLEND
QPCBOEI1 QPCBOEI2 QPCSTAIR PRIMAL1 PRIMAL2 PRIMAL3 CVXQP1_N100 CVXQP2_N100
CVXQP3_N100 NCVXQP1_N100 NCVXQP2_N100 NCVXQP3_N100 NCVXQP4_N100 NCVXQP5_N100
NCVXQP6_N100 NCVXQP7_N100 NCVXQP8_N100 NCVXQP9_N100 DTOC3_N50 UBH1_N10
STCQP1_P8""".split()

# Restrict the run to a subset, space-separated, for targeted re-runs such as
# repeating only the instances a previous pass timed out on.
if os.environ.get("FS_PALM_PROBLEMS"):
    PROBLEMS = os.environ["FS_PALM_PROBLEMS"].split()

# alpha/alpha_gamma-tagged filenames: this repo keeps a full run per
# (alpha, alpha_gamma) combination under test, driven by the FS_PALM_ALPHA /
# FS_PALM_ALPHA_GAMMA env vars (default 4.0 / 1.01, matching fs_palm.solve's own
# defaults) so an overnight grid can invoke this script repeatedly without
# editing it -- and so nothing here is ever the untagged/bare name, which
# would let one run silently overwrite another's results.
_ALPHA = float(os.environ.get("FS_PALM_ALPHA", "4.0"))
_ALPHA_GAMMA = float(os.environ.get("FS_PALM_ALPHA_GAMMA", "1.01"))
# PANOC stays untagged-by-solver-name (matches every filename already on disk
# from tonight's PANOC grid); ProxGrad gets an explicit prefix since it's a
# new track -- alpaqa's side uses FISTASolver(disable_acceleration=True) in
# place of PANOCSolver+LBFGSDirection, see _run_alpaqa_alm_impl.
_INNER = os.environ.get("FS_PALM_INNER_SOLVER", "PANOC")
_TAG = (f"alpha{_ALPHA:g}_ag{_ALPHA_GAMMA:g}" if _INNER == "PANOC" else
        f"{_INNER.lower()}_alpha{_ALPHA:g}_ag{_ALPHA_GAMMA:g}")
CSV = f"alpaqa_alm_bench_fair_{_TAG}.csv"
FIELDS = ["instance", "n", "m", "solver", "grad_evals", "objective",
          "violation", "status", "wall_time"]
TOL = 1e-5
MAX_ITER = 500
EPS_FEAS = 1e-5
FTOL = 1e-1

# Per-iteration history (grad_evals, objective, violation), one JSON file per
# (instance, solver). Exists so stage_plot can compute a FAIR cost metric:
# "evals to first reach the common objective+feasibility target" for BOTH
# solvers, rather than "evals until each solver's own internal stopping test
# fires" -- fs_palm's own test is strictly harder (it additionally requires the
# proximal residual (1/gamma_k)||x^{k+1}-x^k|| below tol, which alpaqa's ALM
# has no analogue of), so comparing self-reported-done times was not
# apples-to-apples. Kept in a separate CSV/history dir from the earlier,
# final-result-only run so the two are never silently mixed.
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"alpaqa_alm_bench_history_{_TAG}")


def _history_path(name, solver):
    return os.path.join(HISTORY_DIR, f"{name}__{solver}.json")


def save_history(name, solver, evals, f, viol):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    payload = {"evals": [int(e) for e in evals],
              "f": [float(v) for v in f],
              "viol": [float(v) for v in viol]}
    path = _history_path(name, solver)
    tmp = path + ".tmp"
    # fsync BEFORE the rename. os.replace is atomic with respect to the
    # rename itself, but not with respect to the data: without the fsync the
    # rename can commit while the tail of the file is still in the page
    # cache, so a power loss leaves a correctly-named file truncated at a
    # writeback boundary. Observed exactly that on a laptop power-off during
    # the ProxGrad run, which left PRIMALC1__alpaqa_alm.json truncated at
    # 32 MiB and unparseable. These files reach ~130 MB under ProxGrad, so
    # the window between write and writeback is wide.
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic on POSIX: no half-written file on a crash


def load_history(name, solver):
    path = _history_path(name, solver)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
# Wall-clock cap per solver call, enforced by killing a spawned subprocess.
# alpaqa's own ALMParams.max_time is set a bit below this so a genuinely
# slow (not hung) run reports a clean "MaxTime" status from inside alpaqa
# itself before the outer hard-kill would ever fire.
#
# A WALL-CLOCK CAP IS NOT A NEUTRAL BUDGET when the two solvers cost
# different amounts per gradient evaluation. Measured on the ProxGrad run,
# fs_palm costs 2.39e-3 s/eval against alpaqa's 6.09e-5, a factor of about 39,
# since fs_palm is Python and JAX while alpaqa is compiled C++. That is exactly
# the implementation gap the gradient-evaluation metric exists to remove, and
# a shared wall clock quietly puts it back. Under PANOC it never bit, because
# eval counts were in the hundreds. Under ProxGrad they reach millions, so at
# 600s fs_palm gets ~250k evals while alpaqa runs to its own 500x5000 = 2.5e6
# iteration limit, a ten-fold asymmetry that scores fs_palm as failing runs it
# was merely cut off during. Set FS_PALM_SOLVER_TIMEOUT to give fs_palm the time
# it needs to reach a comparable EVALUATION budget (2.5e6 evals is ~6000s).
SOLVER_TIMEOUT = float(os.environ.get("FS_PALM_SOLVER_TIMEOUT", "600"))

FS_PALM_KW = dict(use_proximal=True, phi_strategy="pow", xi1=1.0, xi2=1.0,
                alpha=_ALPHA, alpha_gamma=_ALPHA_GAMMA, beta=0.5,
                rho0="rule3", nu0="rule3", gamma0="auto", delta="auto",
                gamma_kappa=1e3, adaptive_fp_tol=True, inner_solver=_INNER,
                start_feas=True)


# ------------------------------------------------------------------- runs ---
def _run_fs_palm_impl(name, tol=TOL):
    inst = load_cutest(name)
    problem = fs_palm_problem(inst, fs_palm, jittable=True)
    t0 = time.perf_counter()
    sol = fs_palm.solve(problem, inst.x0(), tol=tol, max_iter=MAX_ITER,
                      verbosity=0, **FS_PALM_KW)
    wall = time.perf_counter() - t0
    x = np.asarray(sol.x, dtype=float)
    # sol.grad_evals/f_hist/total_infeas are per-outer-iteration arrays of
    # matching length (verified directly beforehand); this excludes Phase I,
    # which runs on its own separate fs_palm.Problem with its own counters.
    save_history(name, "fs_palm", sol.grad_evals, sol.f_hist, sol.total_infeas)
    return dict(n=inst.n, m=inst.m, x=x, grad_evals=int(sol.grad_evals[-1]),
                objective=inst.objective(x), violation=inst.violation(x),
                wall=wall, status=str(sol.solve_status))


def _cutest_violation(cutest_prob, x, m):
    viol = 0.0
    if m > 0:
        g = np.asarray(cutest_prob.eval_constraints(x))
        gb = cutest_prob.general_bounds
        viol = float(np.max(np.maximum(
            np.maximum(g - gb.upper, gb.lower - g), 0.0)))
    vb = cutest_prob.variable_bounds
    box_viol = float(np.max(np.maximum(
        np.maximum(x - vb.upper, vb.lower - x), 0.0)))
    return max(viol, box_viol)


def _run_alpaqa_alm_impl(name, tol=TOL):
    so = os.path.join(CUTEST_DIR, name, f"{name}.so")
    outsdif = os.path.join(CUTEST_DIR, name, "OUTSDIF.d")
    cutest_prob = pa.CUTEstProblem(so, outsdif)
    prob = pa.Problem(cutest_prob)
    alm_params = {
        "tolerance": tol,
        "dual_tolerance": tol,
        "max_iter": MAX_ITER,
        "max_time": float(SOLVER_TIMEOUT - 100),
        # alpaqa's own inner-tolerance schedule is
        # eps_{k+1} = max(tolerance_update_factor * eps_k, tolerance),
        # eps_0 = initial_tolerance -- same shape as fs_palm's
        # tau_k = max(tau0 * kappa_tol**k, tol), but with unmatched
        # defaults (initial_tolerance=1.0, tolerance_update_factor=0.1,
        # reaching a 1e-6 floor in ~6 outer iterations) vs fs_palm's
        # (tau0=0.1, kappa_tol=0.5, reaching the same floor in ~17).
        # Match fs_palm's schedule so neither side's inner solves are
        # asked to tighten faster or slower than the other's.
        "initial_tolerance": 0.1,
        "tolerance_update_factor": 0.5,
    }
    n, m = cutest_prob.num_variables, cutest_prob.num_constraints
    cnt_holder = {}
    hist = {"evals": [], "f": [], "viol": []}

    def _progress(info):
        # info.x_hat is the PANOC candidate at this inner step; cnt_holder
        # is filled in below before the solve starts, so the closure sees
        # the live evaluation counter, not a snapshot.
        x_hat = np.asarray(info.x_hat)
        cnt = cnt_holder.get("cnt")
        evals_now = int(cnt.evaluations.lagrangian_gradient) if cnt is not None else int(info.k)
        hist["evals"].append(evals_now)
        hist["f"].append(float(cutest_prob.eval_objective(x_hat)))
        hist["viol"].append(_cutest_violation(cutest_prob, x_hat, m))

    if _INNER == "PANOC":
        # alpaqa's own PANOCSolver default is max_iter=100 per inner call,
        # far below fs_palm's 1000-2000 -- match it explicitly so neither side
        # gets an unmatched inner-iteration budget.
        # PANOCSolver(dict) alone resolves to the "structured L-BFGS
        # direction" overload (memory=10 default), not the plain
        # LBFGSDirection fs_palm's own inner_solvers.py builds explicitly
        # (memory=20, see PaProblem.pa_direction there) -- pass the direction
        # explicitly so both sides run the same direction/memory, and match
        # fs_palm's stop_crit (ProjGradUnitNorm) instead of PANOCParams' own
        # default (ApproxKKT).
        inner = pa.PANOCSolver(
            {"max_iter": 1000, "stop_crit": pa.ProjGradUnitNorm},
            pa.LBFGSDirection({"memory": 20}),
        )
    else:
        # "basic ProxGrad": FISTA is Nesterov-accelerated proximal gradient;
        # disable_acceleration=True reduces it to plain forward-backward
        # splitting, matching fs_palm's own ProxGrad inner solver in spirit
        # (neither is accelerated/quasi-Newton). max_iter=5000 matches
        # fs_palm's own ProxGrad budget (see fs_palm.py's max_iter_inner default,
        # 5000 for any inner_solver other than PANOC), not PANOC's 1000 --
        # ProxGrad needs far more steps per outer iteration with no
        # acceleration to fall back on.
        inner = pa.FISTASolver(
            {"max_iter": 5000, "stop_crit": pa.ProjGradUnitNorm,
             "disable_acceleration": True})
    inner.set_progress_callback(_progress)
    solver = pa.ALMSolver(alm_params, inner)
    x0 = np.zeros(n)
    y0 = np.zeros(m)
    cnt = pa.problem_with_counters(prob)
    cnt_holder["cnt"] = cnt
    t0 = time.perf_counter()
    x_sol, y_sol, stats = solver(cnt.problem, x0, y0)
    wall = time.perf_counter() - t0
    x_sol = np.asarray(x_sol, dtype=float)

    f = float(cutest_prob.eval_objective(x_sol))
    viol = _cutest_violation(cutest_prob, x_sol, m)
    save_history(name, "alpaqa_alm", hist["evals"], hist["f"], hist["viol"])

    status = stats.get("status") if hasattr(stats, "get") else stats
    # lagrangian_gradient, not objective_gradient: ALMSolver on a
    # general-constraint problem evaluates the Lagrangian gradient
    # (grad f + J^T y) as one fused call per PANOC step, which is what's
    # actually comparable to fs_palm's own grad_evals count -- verified
    # against fs_palm's known GENHS28 result (269 vs 220, same order of
    # magnitude) before trusting this for the real run.
    return dict(n=n, m=m, x=x_sol, grad_evals=int(cnt.evaluations.lagrangian_gradient),
                objective=f, violation=viol, wall=wall, status=str(status))


def run_fs_palm(name, tol=TOL):
    return run_with_timeout(_run_fs_palm_impl, args=(name,), kwargs={"tol": tol},
                            timeout=SOLVER_TIMEOUT)


def run_alpaqa_alm(name, tol=TOL):
    return run_with_timeout(_run_alpaqa_alm_impl, args=(name,), kwargs={"tol": tol},
                            timeout=SOLVER_TIMEOUT)


# -------------------------------------------------------------------- csv ---
def load_rows(path=CSV):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def append_row(row, path=CSV):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()


def record(name, solver, out=None, err=None):
    if err is not None:
        row = dict(instance=name, n="", m="", solver=solver, grad_evals="",
                   objective="", violation="", status=f"ERROR:{err}"[:150],
                   wall_time="")
    else:
        row = dict(instance=name, n=out["n"], m=out["m"], solver=solver,
                   grad_evals=out["grad_evals"], objective=out["objective"],
                   violation=out["violation"], status=out["status"],
                   wall_time=out["wall"])
    append_row(row)
    return row


# -------------------------------------------------------------------- run ---
def stage_run(names):
    done = {(r["instance"], r["solver"]) for r in load_rows()}
    print(f"{len(names)} instances; {len(done)} runs already recorded\n")

    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}", flush=True)
        for solver, fn in (("fs_palm", run_fs_palm), ("alpaqa_alm", run_alpaqa_alm)):
            if (name, solver) in done:
                print(f"    {solver:12s} already recorded, skipping", flush=True)
                continue
            try:
                out = fn(name)
                row = record(name, solver, out=out)
                print(f"    {solver:12s} {int(float(row['grad_evals'])):>8} evals "
                      f"viol={float(row['violation']):.2e} "
                      f"f={float(row['objective']):.8f} status={row['status']}",
                      flush=True)
            except Exception as e:
                record(name, solver, err=repr(e))
                print(f"    {solver:12s} FAILED {type(e).__name__}: {e}"[:150],
                      flush=True)
    print(f"\ndone -> {CSV}")


# ------------------------------------------------------------------- plot ---
def stage_plot():
    from fs_palm.utils.plotting import setup_matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fs_palm.utils import perfprof

    rows = load_rows()
    if not rows:
        sys.exit(f"no results in {CSV}; run the 'run' stage first")
    setup_matplotlib(font_scale=1.8)

    solvers = ["fs_palm", "alpaqa_alm"]
    names = [r"\texttt{FS-P-ALM}", r"\texttt{alpaqa ALM}"]
    insts = sorted({r["instance"] for r in rows})
    idx = {nm: i for i, nm in enumerate(insts)}
    cost = np.full((len(insts), 2), np.nan)
    fval = np.full((len(insts), 2), np.nan)
    viol = np.full((len(insts), 2), np.nan)
    for r in rows:
        if r["solver"] not in solvers or not r["grad_evals"]:
            continue
        i, j = idx[r["instance"]], solvers.index(r["solver"])
        cost[i, j] = float(r["grad_evals"])
        fval[i, j] = float(r["objective"])
        viol[i, j] = float(r["violation"])

    ok = (viol <= EPS_FEAS)
    f_best = np.nanmin(np.where(ok, fval, np.inf), axis=1)
    close_enough = np.abs(fval - f_best[:, None]) <= FTOL * np.maximum(1.0, np.abs(f_best[:, None]))
    success = ok & close_enough

    disagree = np.sum(ok.all(axis=1) & ~close_enough.all(axis=1))
    print(f"different local minima (rel objective gap > {FTOL:.0e}): "
          f"{disagree}/{len(insts)} instances")

    # FAIR cost: evals to the FIRST point in each solver's own history that
    # already meets the (feasible, close to f_best) target -- not evals until
    # that solver's own internal stopping test fires. fs_palm's own test is
    # strictly harder (see module docstring), so the two are not comparable
    # as-is; this recomputes both against one common, externally-defined
    # target. Falls back to the final-result cost if no history file exists
    # (e.g. a run recorded before this fix was added) with a warning, so a
    # partially-mixed CSV degrades visibly rather than silently.
    fair_cost = np.where(success, cost, np.inf)
    missing_history = 0
    for i, name in enumerate(insts):
        for j, solver in enumerate(solvers):
            if not success[i, j]:
                continue
            hist = load_history(name, solver)
            if hist is None:
                missing_history += 1
                continue
            evals_h = np.asarray(hist["evals"], dtype=float)
            f_h = np.asarray(hist["f"], dtype=float)
            viol_h = np.asarray(hist["viol"], dtype=float)
            hit = (viol_h <= EPS_FEAS) & (
                np.abs(f_h - f_best[i]) <= FTOL * max(1.0, abs(f_best[i])))
            if np.any(hit):
                # floor at 1: a genuinely-0-eval hit (starting point already
                # meets the target) is a real result, but a literal 0 cost
                # breaks the log-ratio performance profile -- same floor
                # convention as gm() below.
                fair_cost[i, j] = max(float(evals_h[np.argmax(hit)]), 1.0)
    if missing_history:
        print(f"  [warning] {missing_history} successful (instance, solver) "
              f"pairs had no history file; used final-result cost for those")

    _outfile = f"alpaqa_alm_profile_fs_palm_alpaqa_{_TAG}.pdf"
    perfprof.pairwise_profiles(
        fair_cost, success, names, r"$\textbf{grad evals}$",
        outfile=_outfile)
    print(f"wrote {_outfile}")

    # Over each solver's OWN successes only -- fair_cost is inf on a failure
    # by construction (for the Dolan-More profile, where that is correct),
    # so averaging it in unmasked would make one failure among 50 instances
    # blow the whole summary number up to inf and hide everything else.
    def gm(v, mask):
        v = np.asarray(v)[mask]
        return float(np.exp(np.mean(np.log(np.maximum(v, 1.0))))) if v.size else float("nan")

    print(f"shifted geometric mean grad_evals, among each solver's own "
          f"successes (fair, final-cost in brackets): "
          f"fs_palm={gm(fair_cost[:,0], success[:,0]):.1f} "
          f"({gm(cost[:,0], success[:,0]):.1f})  "
          f"alpaqa_alm={gm(fair_cost[:,1], success[:,1]):.1f} "
          f"({gm(cost[:,1], success[:,1]):.1f})")
    print(f"success (feasible & within {FTOL:.0e} of best): "
          f"fs_palm={int(np.nansum(success[:,0]))}/{len(insts)}  "
          f"alpaqa_alm={int(np.nansum(success[:,1]))}/{len(insts)}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("run", "all"):
        stage_run(PROBLEMS)
    if stage in ("plot", "all"):
        stage_plot()


if __name__ == "__main__":
    main()
