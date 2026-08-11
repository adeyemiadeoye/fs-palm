"""Independent correctness check of the ProxGrad inner solver.

The claim "ALM diverges under a first-order inner solver" is only as good as
this code, so verify it against problems with independently known answers
rather than against PANOC (which would only show the two agree, not that
either is right).
"""
# Prefer this repo's solver over any installed pbalm. A pip-installed pbalm in
# the same environment would otherwise shadow it and the test would exercise the
# published package instead of the working copy. This must run BEFORE any
# import of pbalm, including one buried in a comma-separated import line.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _os.path.isdir(_os.path.join(_SRC, "pbalm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import numpy as np, jax, jax.numpy as jnp, pbalm, sys
jax.config.update("jax_platform_name", "cpu"); jax.config.update("jax_enable_x64", True)
from pbalm.inner_solvers.inner_solvers import _make_prox, get_proxgrad_run, get_solver_run

fail = 0
def check(name, ok, detail=""):
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok: fail += 1

print("=== 1. prox operators against closed forms ===")
v = jnp.array([-3.0, -0.4, 0.0, 0.7, 5.0])
prox, reg = _make_prox(f2=pbalm.L1Norm(0.5))
got = np.asarray(prox(v, 1.0))
want = np.sign(np.asarray(v)) * np.maximum(np.abs(np.asarray(v)) - 0.5, 0.0)
check("L1 soft-threshold (gamma=1)", np.allclose(got, want), f"max|d|={np.abs(got-want).max():.2e}")
got = np.asarray(prox(v, 2.0)); want = np.sign(np.asarray(v))*np.maximum(np.abs(np.asarray(v))-1.0, 0.0)
check("L1 soft-threshold (gamma=2)", np.allclose(got, want), f"max|d|={np.abs(got-want).max():.2e}")
check("L1 reg value", np.isclose(float(reg(v)), 0.5*np.abs(np.asarray(v)).sum()))
proxb, regb = _make_prox(f2=pbalm.Box(lower=np.full(5,-1.0), upper=np.full(5,1.0)))
check("Box projection", np.allclose(np.asarray(proxb(v,1.0)), np.clip(np.asarray(v),-1,1)))

print("\n=== 2. LASSO: ProxGrad vs cvxpy (convex, unique optimum) ===")
rng = np.random.default_rng(0); m, n, lam = 60, 30, 0.3
A = rng.standard_normal((m, n)); b = rng.standard_normal(m)
Aj, bj = jnp.array(A), jnp.array(b)
def f_smooth(x): return 0.5*jnp.sum((Aj@x - bj)**2)
import cvxpy as cp
xv = cp.Variable(n)
cp.Problem(cp.Minimize(0.5*cp.sum_squares(A@xv-b) + lam*cp.norm1(xv))).solve(solver="CLARABEL")
x_ref = xv.value
runner = get_proxgrad_run(f2=pbalm.L1Norm(lam), jittable=True)
x_pg, st = runner.train_fun(f_smooth, jnp.zeros(n), 20000, 1e-10)
x_pg = np.asarray(x_pg)
F = lambda x: 0.5*np.sum((A@x-b)**2) + lam*np.abs(x).sum()
check("LASSO solution matches cvxpy", np.max(np.abs(x_pg-x_ref)) < 1e-6,
      f"max|dx|={np.max(np.abs(x_pg-x_ref)):.2e}")
# relative, and one-sided: finding a LOWER objective than the reference is
# success, not failure (it happens: our point beat Clarabel's by 2e-9 here).
check("LASSO objective <= cvxpy + tol",
      F(x_pg) <= F(x_ref) + 1e-8*max(1.0, abs(F(x_ref))), f"dF={F(x_pg)-F(x_ref):+.2e}")
check("reports Converged", st["status"] == "Converged", f"status={st['status']} fp={st['fp_res']:.1e}")
# count at a cutoff well clear of the coefficients that sit near zero;
# at 1e-8 two coefficients straddle the threshold and the count is not a
# property of the solution
check("sparsity recovered", int((np.abs(x_pg)>1e-5).sum()) == int((np.abs(x_ref)>1e-5).sum()),
      f"nnz {int((np.abs(x_pg)>1e-5).sum())} vs {int((np.abs(x_ref)>1e-5).sum())} (cutoff 1e-5)")

print("\n=== 3. box-constrained QP: ProxGrad vs cvxpy ===")
Q = A.T@A + 0.1*np.eye(n); q = rng.standard_normal(n)
Qj, qj = jnp.array(Q), jnp.array(q)
def f_qp(x): return 0.5*jnp.dot(x, Qj@x) + jnp.dot(qj, x)
xv = cp.Variable(n)
cp.Problem(cp.Minimize(0.5*cp.quad_form(xv, cp.psd_wrap(Q)) + q@xv),
           [xv >= -0.5, xv <= 0.5]).solve(solver="CLARABEL")
x_ref = xv.value
runner = get_proxgrad_run(f2=pbalm.Box(lower=np.full(n,-0.5), upper=np.full(n,0.5)), jittable=True)
x_pg, st = runner.train_fun(f_qp, jnp.zeros(n), 50000, 1e-10)
x_pg = np.asarray(x_pg)
check("box-QP solution matches cvxpy", np.max(np.abs(x_pg-x_ref)) < 1e-6,
      f"max|dx|={np.max(np.abs(x_pg-x_ref)):.2e}")
check("stays inside the box", x_pg.min() >= -0.5-1e-12 and x_pg.max() <= 0.5+1e-12)

print("\n=== 4. ill-conditioned smooth QP: does cost track the condition number? ===")
print("     (this is the property the whole ProxGrad argument relies on)")
for kappa in (1e1, 1e3, 1e5):
    V,_ = np.linalg.qr(rng.standard_normal((n,n)))
    w = np.logspace(0, np.log10(kappa), n)
    Qk = jnp.array((V*w)@V.T); qk = jnp.array(rng.standard_normal(n))
    def fk(x): return 0.5*jnp.dot(x, Qk@x) + jnp.dot(qk, x)
    for nm, run in (("ProxGrad", get_proxgrad_run(jittable=True)),
                    ("PANOC   ", get_solver_run(jittable=True))):
        _, s = run.train_fun(fk, jnp.zeros(n), 200000, 1e-8)
        print(f"     kappa={kappa:7.0e}  {nm}  grad_evals={s['obj_grad_evals']:>8d}  status={s['status']}")

print(f"\n{'ALL CHECKS PASSED' if fail==0 else str(fail)+' CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
