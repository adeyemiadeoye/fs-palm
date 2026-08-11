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
n = 10
rng = np.random.default_rng(42)

# Positive definite Q matrix
M = rng.standard_normal((n, n))
Q = jnp.array(M.T @ M + 0.1 * np.eye(n))
c = jnp.array(rng.standard_normal(n))

# Equality constraint: sum(x) = 1
A = jnp.ones((1, n))
b_eq = jnp.array([1.0])

# Inequality constraint: x >= 0 (i.e., -x <= 0)
G = -jnp.eye(n)
h_ineq = jnp.zeros(n)

# Define functions
def f1(x):
    return 0.5 * x @ Q @ x + c @ x

def h(x):
    return A @ x - b_eq

def g(x):
    return G @ x - h_ineq

# Create and solve problem
problem = pbalm.Problem(f1=f1, h=[h], g=[g], jittable=True)
x0 = jnp.ones(n) / n  # Start on simplex

result = pbalm.solve(problem, x0, tol=1e-6)

print(f"Optimal x: {result.x}")
eq_con = h(result.x)
ineq_con = g(result.x)
print(f"Equality constraint: {eq_con}")
print(f"Inequality constraint: {ineq_con}")