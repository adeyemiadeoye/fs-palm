import jax.numpy as jnp
import fs_palm
import numpy as np

#    \min_{x} \quad & \frac{1}{2}\|Ax - b\|^2 + \lambda \|x\|_1 \\
#    \text{s.t.} \quad & \mathbf{1}^T x = 1 \\
#                      & x \geq 0

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
f2 = fs_palm.L1Norm(f2_lbda)

# Create problem
problem = fs_palm.Problem(
    f1=f1,
    h=[h],
    g=[g],
    f2=f2,
    jittable=True
)

x0 = jnp.ones(n) / n

result = fs_palm.solve(
    problem,
    x0,
    tol=1e-5,
    max_iter=500
)

print(f"Sum of x: {jnp.sum(result.x):.6f}")
print(f"Min of x: {jnp.min(result.x):.6f}")
print(f"Number of zeros: {jnp.sum(jnp.abs(result.x) < 1e-4)}")
print(f"Objective: {f1(result.x) + f2_lbda * jnp.sum(jnp.abs(result.x)):.6f}")