import jax.numpy as jnp
import fs_palm
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
problem = fs_palm.Problem(f1=f1, h=[h], g=[g], jittable=True)
x0 = jnp.ones(n) / n  # Start on simplex

result = fs_palm.solve(problem, x0, tol=1e-6)

print(f"Optimal x: {result.x}")
eq_con = h(result.x)
ineq_con = g(result.x)
print(f"Equality constraint: {eq_con}")
print(f"Inequality constraint: {ineq_con}")