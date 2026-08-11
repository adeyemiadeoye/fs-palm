import matplotlib
import seaborn as sns

from matplotlib import style
import scienceplots

def setup_matplotlib(font_scale=2.5, style_name="science", grid=True):
    if grid:
        sns.set_style(style.library[style_name])
        matplotlib.rcParams.update({'axes.axisbelow': True, 'axes.grid': True, 'grid.alpha': 0.5, 'grid.color': 'k', 'grid.linestyle': '--', 'grid.linewidth': 0.5, 'legend.fancybox': True, 'legend.framealpha': 1.0, 'legend.frameon': True, 'legend.numpoints': 1})
    else:
        sns.set_style(style.library[style_name])
        matplotlib.rcParams.update({'axes.grid': False, 'legend.fancybox': True, 'legend.frameon': True, 'legend.numpoints': 1})

    sns.set_context("paper", font_scale=font_scale,
                  rc={"lines.linewidth": 3,
                      "mathtext.fontset": "stix",
                  })
    
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = (
        r'\usepackage{amsmath,amssymb}'
        r'\boldmath'
        r'\bfseries'
    )


# Inner solvers the example scripts can produce figures for. Kept here rather
# than duplicated in each example because it is shared plotting plumbing, and
# the examples directory is for experiments, not utilities.
INNER_SOLVERS = ("PANOC", "ProxGrad")


def inner_solvers(spec):
    """Resolve --inner-solver into a list.

    "both" (the default in the example scripts) produces a SEPARATE set of
    figures per inner solver. The two are not interchangeable views of one
    experiment: PANOC is quasi-Newton, so its L-BFGS directions absorb the
    subproblem ill-conditioning that a growing penalty induces, whereas a
    proximal gradient method's cost reflects that conditioning directly.
    Reporting only one of them either hides the effect (PANOC) or overstates
    how a practitioner using a strong inner solver would experience it
    (ProxGrad), so the examples report both.
    """
    if spec is None or str(spec).lower() == "both":
        return list(INNER_SOLVERS)
    for known in INNER_SOLVERS:
        if str(spec).lower() == known.lower():
            return [known]
    raise ValueError(f"unknown inner solver {spec!r}; "
                     f"expected one of {INNER_SOLVERS} or 'both'")
