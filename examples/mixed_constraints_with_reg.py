import jax.numpy as jnp
# Prefer this repo's solver over any installed pbalm. A pip-installed pbalm
# in the same environment would otherwise shadow it and the script would
# exercise the published package instead of the working copy.
import os as _os, sys as _sys
_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")
if _os.path.isdir(_os.path.join(_SRC, "pbalm")) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
import pbalm
import numpy as np

# Configure JAX
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Problem data
m, n = 50, 100
rng = np.random.default_rng(789)

A = jnp.array(rng.standard_normal((m, n)))
b = jnp.array(rng.standard_normal(m))

# Smooth part of objective
def f1(x):
    residual = A @ x - b
    return 0.5 * jnp.sum(residual**2)

# Equality: sum(x) = 1
def h(x):
    return jnp.sum(x) - 1.0

# Inequality: x >= 0
def g(x):
    return -x

# L1 regularization
f2_lbda = 0.1
f2 = pbalm.L1Norm(f2_lbda)

# Create problem
problem = pbalm.Problem(
    f1=f1,
    h=[h],
    g=[g],
    f2=f2,
    jittable=True
)

x0 = jnp.ones(n) / n

result = pbalm.solve(
    problem,
    x0,
    tol=1e-5,
    max_iter=500
)

print(f"Sum of x: {jnp.sum(result.x):.6f}")
print(f"Min of x: {jnp.min(result.x):.6f}")
print(f"Number of zeros: {jnp.sum(jnp.abs(result.x) < 1e-4)}")
print(f"Objective: {f1(result.x) + f2_lbda * jnp.sum(jnp.abs(result.x)):.6f}")