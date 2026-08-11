# fs-palm

Python implementation of [FS-P-ALM](#references).

**fs-palm** solves optimization problems of the form:

$$
\begin{aligned}
\min_{x} \quad & f_1(x) + f_2(x) \\
\text{s.t.} \quad & g(x) \leq 0 \\
& h(x) = 0
\end{aligned}
$$

where $f_1$ is smooth (possibly nonconvex), $f_2$ is possibly nonsmooth but prox-friendly, and $g$, $h$ define smooth inequality and equality constraints, respectively.

Quick example:

```bash
python3 -m pip install fs-palm
```

```python
import jax.numpy as jnp
import fs_palm


def f1(x):
    return jnp.sum(x ** 2)


# L1 regularization (nonsmooth)
lbda = 0.1
f2 = fs_palm.L1Norm(lbda)


# inequality constraint g_j(x) <= 0; j=1
def g_1(x):
    return x[0] - 0.8


# equality constraints h_i(x) = 0; i=1,2
def h_1(x):
    return x[0] + x[1] - 1.0


def h_2(x):
    return x[1] * x[2] - 2.0


x0 = jnp.array([0.0, 0.0, 0.0])

# create problem and solve
problem = fs_palm.Problem(f1=f1, f2=f2, g=[g_1], h=[h_1, h_2])
result = fs_palm.solve(problem, x0=x0, tol=1e-3)

print(f"Solution: {result.x}")
```

More examples in `/examples/` directory.

## References

```bibtex
@article{ncvxproxALM2025,
  title={A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints},
  author={Adeoye, Adeyemi D. and Latafat, Puya and Bemporad, Alberto},
  journal={arXiv preprint arXiv:2509.02894},
  year={2025}
}
```

## Acknowledgments

This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
    <img src="https://github.com/adeyemiadeoye/fs-palm/blob/main/src/other_media/erc-logo.png" alt="ERC logo" width="400"/>
</p>

Default inner solver: [PANOC](https://ieeexplore.ieee.org/abstract/document/8263933); also calls prox functions from: [alpaqa](https://github.com/kul-optec/alpaqa).