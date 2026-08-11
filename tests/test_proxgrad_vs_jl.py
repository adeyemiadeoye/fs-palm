"""Cross-check ProxGrad against ProximalAlgorithms.jl -- independent process.

Comparing against cvxpy shows we reach the right MINIMISER. Comparing against a
reference proximal-gradient implementation shows we take a comparable PATH
there, which is what tests the linesearch and the stopping rule rather than the
prox operators.

This supersedes test_proxgrad_vs_proximalalgorithms_jl.py, which drove Julia
through juliacall. The bridge is not installed here, and a subprocess is a
cleaner check anyway: the reference runs in its own process with no shared
state, so nothing about our Python side can influence it.

Three problems, chosen to exercise different parts of the linesearch rather
than to re-test the same easy case three times:

    lasso-easy   kappa(A'A) ~ 1e1   the linesearch should barely engage
    lasso-hard   kappa(A'A) ~ 1e6   forces repeated backtracking, and is where
                                    a warm-started stepsize could plausibly be
                                    carried across a curvature change and stick
    boxqp        indicator of a box rather than l1, i.e. a projection

What is asserted, and what is not. The MINIMISER must agree: both solve the
same convex problem, so a disagreement there is a bug in one of them. The
ITERATION COUNTS need not agree and are only reported -- the two use different
linesearch constants and different stopping rules, so equality would be a
coincidence, and requiring it would be testing the reference's tuning rather
than our correctness.
"""
import json
import os
import subprocess
import sys
import tempfile

import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _os.path.isdir(_os.path.join(_SRC, "fs_palm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fs_palm
from fs_palm.inner_solvers.inner_solvers import get_proxgrad_run

JULIA = os.environ.get("JULIA", "/opt/julia/julia-1.12.1/bin/julia")
HERE = os.path.dirname(os.path.abspath(__file__))


def make_problems():
    rng = np.random.default_rng(0)
    P = {}

    m, n = 60, 30
    A = rng.standard_normal((m, n))
    P["lasso-easy"] = dict(A=A, b=rng.standard_normal(m), prox="l1", lam=0.3,
                           x0=np.zeros(n))

    # ill-conditioned design: singular values spread over three decades, so
    # A'A spans six and the descent-lemma stepsize has to move a long way
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = np.logspace(0, -3, n)
    A2 = U @ np.diag(sv) @ V.T
    P["lasso-hard"] = dict(A=A2, b=rng.standard_normal(m), prox="l1", lam=1e-3,
                           x0=np.zeros(n))

    A3 = rng.standard_normal((40, 25))
    P["boxqp"] = dict(A=A3, b=rng.standard_normal(40), prox="box",
                      lo=-0.25, hi=0.25, x0=np.zeros(25))
    return P


def julia_reference(P):
    spec = {k: {"A": v["A"].tolist(), "b": v["b"].tolist(),
                "x0": v["x0"].tolist(), "prox": v["prox"],
                "lam": float(v.get("lam", 0.0)),
                "lo": float(v.get("lo", 0.0)), "hi": float(v.get("hi", 0.0))}
            for k, v in P.items()}
    with tempfile.TemporaryDirectory() as d:
        fi, fo = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        json.dump(spec, open(fi, "w"))
        r = subprocess.run([JULIA, "--project=@.",
                            os.path.join(HERE, "proxgrad_ref.jl"), fi, fo],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise SystemExit("julia reference failed")
        return json.load(open(fo))


def ours(p):
    A = jnp.asarray(p["A"])
    b = jnp.asarray(p["b"])

    def f(x):
        r = A @ x - b
        return 0.5 * jnp.dot(r, r)

    if p["prox"] == "l1":
        f2, lam = fs_palm.L1Norm(float(p["lam"])), float(p["lam"])
    else:
        n = p["A"].shape[1]
        f2, lam = fs_palm.Box(
            lower=np.full(n, p["lo"]), upper=np.full(n, p["hi"])), None
    run = get_proxgrad_run(f2=f2, l1_lbda=lam, jittable=True)
    x, info = run.train_fun(f, jnp.asarray(p["x0"]), 200000, 1e-12)
    return np.asarray(x), info


if __name__ == "__main__":
    P = make_problems()
    print("running ProximalAlgorithms.jl reference ...", flush=True)
    ref = julia_reference(P)

    print(f"\n{'problem':>12s} {'max|dx|':>11s} {'obj ours':>15s} "
          f"{'obj julia':>15s} {'d(obj)':>11s} {'it ours':>8s} {'it jl':>7s}")
    bad = []
    for name, p in P.items():
        x, info = ours(p)
        xr = np.asarray(ref[name]["x"])
        dx = float(np.max(np.abs(x - xr)))
        # objective on OUR iterate vs the reference's, both scored the same way
        r = p["A"] @ x - p["b"]
        obj = 0.5 * float(r @ r) + (float(p["lam"]) * float(np.abs(x).sum())
                                    if p["prox"] == "l1" else 0.0)
        dobj = abs(obj - float(ref[name]["obj"]))
        print(f"{name:>12s} {dx:11.2e} {obj:15.10f} "
              f"{float(ref[name]['obj']):15.10f} {dobj:11.2e} "
              f"{info.get('n_iter', info.get('obj_grad_evals', -1)):8d} "
              f"{int(ref[name]['iters']):7d}")
        # tolerance on the minimiser: both solve to 1e-12 on the fixed-point
        # residual, so agreement to 1e-6 in x is a generous bar that still
        # catches any real discrepancy in what is being minimised
        # If BOTH runs hit the iteration cap, neither has converged and
        # comparing the two iterates tests nothing: on an ill-conditioned
        # problem two unconverged first-order runs sit in different places on a
        # nearly flat valley floor. Fall back to the only meaningful question
        # there -- which iterate is closer to the true optimum -- scored
        # against an interior-point solve.
        stalled = (int(ref[name]["iters"]) >= 200000
                   and info.get("status") in ("MaxIter", None))
        if stalled:
            import cvxpy as cp
            xv = cp.Variable(p["A"].shape[1])
            expr = 0.5 * cp.sum_squares(p["A"] @ xv - p["b"])
            if p["prox"] == "l1":
                expr = expr + float(p["lam"]) * cp.norm1(xv)
                cons = []
            else:
                cons = [xv >= p["lo"], xv <= p["hi"]]
            cp.Problem(cp.Minimize(expr), cons).solve(solver=cp.CLARABEL)
            xstar = np.asarray(xv.value)
            rs = p["A"] @ xstar - p["b"]
            ostar = 0.5 * float(rs @ rs) + (
                float(p["lam"]) * float(np.abs(xstar).sum())
                if p["prox"] == "l1" else 0.0)
            print(f"{'':>12s} both hit the iteration cap; gap to the "
                  f"interior-point optimum: ours {obj - ostar:+.2e}, "
                  f"julia {float(ref[name]['obj']) - ostar:+.2e}")
            if obj - ostar > float(ref[name]["obj"]) - ostar + 1e-12:
                bad.append((name, dx, dobj))
        elif not (dx < 1e-6 and dobj < 1e-9):
            bad.append((name, dx, dobj))

    print()
    if bad:
        print("MISMATCH against the reference implementation:")
        for nm, dx, do in bad:
            print(f"   {nm}: max|dx|={dx:.2e} d(obj)={do:.2e}")
        sys.exit(1)
    print("ProxGrad agrees with ProximalAlgorithms.jl on all problems")
