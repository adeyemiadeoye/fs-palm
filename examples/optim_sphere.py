import jax.numpy as jnp
import fs_palm
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
problem = fs_palm.Problem(f1=f1, h=[h], jittable=True)

# Initial point (will be projected to feasible)
x0 = jnp.ones(n) / jnp.sqrt(n)

# Solve
result = fs_palm.solve(
    problem,
    x0,
    tol=1e-8,
    start_feas=False  # Start from normalized point
)

# Analytical solution: x* = -c / ||c||
x_analytical = -c / jnp.linalg.norm(c)

print(f"PFS-ALM solution: {result.x}")
print(f"Analytical solution: {x_analytical}")
print(f"Solution norm: {jnp.linalg.norm(result.x)}")
print(f"Error: {jnp.linalg.norm(result.x - x_analytical):.2e}")