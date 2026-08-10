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
n = 5
rng = np.random.default_rng(456)
c = jnp.array(rng.standard_normal(n))

# Objective
def f1(x):
    return c @ x

# Sphere constraint: ||x||^2 = 1
def h(x):
    return jnp.sum(x**2) - 1.0

# Create problem
problem = pbalm.Problem(f1=f1, h=[h], jittable=True)

# Initial point (will be projected to feasible)
x0 = jnp.ones(n) / jnp.sqrt(n)

# Solve
result = pbalm.solve(
    problem,
    x0,
    tol=1e-8,
    start_feas=False  # Start from normalized point
)

# Analytical solution: x* = -c / ||c||
x_analytical = -c / jnp.linalg.norm(c)

print(f"PBALM solution: {result.x}")
print(f"Analytical solution: {x_analytical}")
print(f"Solution norm: {jnp.linalg.norm(result.x)}")
print(f"Error: {jnp.linalg.norm(result.x - x_analytical):.2e}")