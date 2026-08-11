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

# Generate synthetic data
rng = np.random.default_rng(123)
n_samples = 100
x_data = jnp.linspace(0, 1, n_samples)

# True model: y = theta1 * exp(-theta2 * x) + noise
theta_true = jnp.array([0.5, 2.0])
y_data = theta_true[0] * jnp.exp(-theta_true[1] * x_data) + 0.05 * rng.standard_normal(n_samples)

# Model prediction
def model(x, theta):
    return theta[0] * jnp.exp(-theta[1] * x)

# Objective: sum of squared residuals
def f1(theta):
    predictions = model(x_data, theta)
    residuals = y_data - predictions
    return jnp.sum(residuals**2)

# Constraints
def g1(theta):
    return theta[0] + theta[1] - 5.0  # theta1 + theta2 <= 5

def g2(theta):
    return -theta  # theta >= 0 (element-wise)

# Create problem
problem = pbalm.Problem(
    f1=f1,
    g=[g1, g2],
    jittable=True
)

# Initial guess
theta0 = jnp.array([0.3, 1.0])

# Solve
result = pbalm.solve(
    problem,
    theta0,
    tol=1e-6,
    max_iter=200
)

print(f"True parameters: {theta_true}")
print(f"Estimated parameters: {result.x}")
print(f"Final objective: {f1(result.x):.6f}")