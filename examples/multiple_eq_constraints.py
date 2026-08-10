import jax.numpy as jnp
# Prefer this repo's solver over any installed pbalm. A pip-installed pbalm
# in the same environment would otherwise shadow it and the script would
# exercise the published package instead of the working copy.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")
if _os.path.isdir(_os.path.join(_SRC, "pbalm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
import pbalm

def f1(x):
    return jnp.sum(x**2)

def h1(x):
    return x[0] * x[1] - 1.0

def h2(x):
    return x[1] * x[2] - 2.0

# Create problem with multiple equality constraints
problem = pbalm.Problem(
    f1=f1,
    h=[h1, h2],
    jittable=True
)

x0 = jnp.array([1.0, 1.0, 2.0])

result = pbalm.solve(
    problem,
    x0,
    tol=1e-9
)

print(f"Solution: {result.x}")
print(f"h1(x) = x1*x2 - 1 = {result.x[0] * result.x[1] - 1:.2e}")
print(f"h2(x) = x2*x3 - 2 = {result.x[1] * result.x[2] - 2:.2e}")
print(f"Objective: {f1(result.x):.6f}")