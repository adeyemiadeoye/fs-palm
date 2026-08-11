import jax
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

#    \min_{\beta} \quad & \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \\
#    \text{s.t.} \quad & \mathbf{1}^T \beta = 1 \quad \text{(coefficients sum to 1)} \\
#                      & \beta \geq 0 \quad \text{(non-negativity)}

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Generate synthetic regression data
def generate_data(n_samples, n_features, n_nonzero, noise_std=0.1, seed=42):
    """Generate sparse regression data."""
    rng = np.random.default_rng(seed)

    # Design matrix
    X = rng.standard_normal((n_samples, n_features))
    X = jnp.array(X)

    # True sparse coefficients (non-negative, sum to 1)
    beta_true = jnp.zeros(n_features)
    support = rng.choice(n_features, size=n_nonzero, replace=False)
    values = rng.uniform(0.1, 1.0, size=n_nonzero)
    values = values / values.sum()  # Normalize to sum to 1
    beta_true = beta_true.at[support].set(jnp.array(values))

    # Response with noise
    y = X @ beta_true + noise_std * jnp.array(rng.standard_normal(n_samples))

    return X, y, beta_true

# Problem dimensions
n_samples = 200
n_features = 100
n_nonzero = 10
lmbda = 0.01  # Regularization parameter

# Generate data
X, y, beta_true = generate_data(n_samples, n_features, n_nonzero)

print(f"True number of nonzeros: {jnp.sum(beta_true > 0)}")
print(f"True coefficients sum: {jnp.sum(beta_true):.4f}")

# Define objective (smooth part)
def f1(beta):
    residual = y - X @ beta
    return 0.5 / n_samples * jnp.sum(residual**2)

# Equality constraint: sum(beta) = 1
def h(beta):
    return jnp.sum(beta) - 1.0

# Inequality constraint: beta >= 0
def g(beta):
    return -beta

# L1 regularization
f2 = pbalm.L1Norm(lmbda)

# Create problem
problem = pbalm.Problem(
    f1=f1,
    h=[h],
    g=[g],
    f2=f2,
    jittable=True
)

# Initial point (uniform)
beta0 = jnp.ones(n_features) / n_features

# Solve
result = pbalm.solve(
    problem,
    beta0,
    tol=1e-6,
    max_iter=500,
    alpha=5,
    verbosity=1
)

# Analyze solution
beta_hat = result.x
threshold = 1e-5

print("\n" + "="*50)
print("Results")
print("="*50)
print(f"Solver status: {result.solve_status}")
print(f"Coefficients sum: {jnp.sum(beta_hat):.6f}")
print(f"Min coefficient: {jnp.min(beta_hat):.6e}")
print(f"Number of nonzeros: {jnp.sum(jnp.abs(beta_hat) > threshold)}")
print(f"True nonzeros: {jnp.sum(beta_true > 0)}")

# Prediction error
y_pred = X @ beta_hat
mse = jnp.mean((y - y_pred)**2)
print(f"MSE: {mse:.6f}")

# Support recovery
support_true = set(jnp.where(beta_true > 0)[0].tolist())
support_hat = set(jnp.where(jnp.abs(beta_hat) > threshold)[0].tolist())

precision = len(support_true & support_hat) / max(len(support_hat), 1)
recall = len(support_true & support_hat) / len(support_true)

print(f"Support precision: {precision:.4f}")
print(f"Support recall: {recall:.4f}")