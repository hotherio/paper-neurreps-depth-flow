"""Figure 3: the transition with the largest in-plane rotation, its quadratic drift fit in the leading principal
plane split into the gradient of the best cubic potential and the remainder."""

import numpy as np
from style import MUT, field, mean_steps, plane, plt, save

from depthflow import data
from depthflow.instruments import load_ladder

S = load_ladder(data("ladder_160m_wikitext.npz"))
s = np.random.default_rng(0).choice(S.shape[1], 6000, replace=False)
quad = lambda P: np.column_stack([np.ones(len(P)), P, P[:, :1] ** 2, P[:, :1] * P[:, 1:], P[:, 1:] ** 2])
potential = lambda P: np.column_stack(
    [
        P,
        P[:, :1] ** 2,
        P[:, :1] * P[:, 1:],
        P[:, 1:] ** 2,
        P[:, :1] ** 3,
        P[:, :1] ** 2 * P[:, 1:],
        P[:, :1] * P[:, 1:] ** 2,
        P[:, 1:] ** 3,
    ]
)


def gradient(P):
    x, y = P[:, 0], P[:, 1]
    o, z = np.ones_like(x), np.zeros_like(x)
    return np.stack(
        [
            np.stack([o, z, 2 * x, y, z, 3 * x**2, 2 * x * y, y**2, z], 1),
            np.stack([z, o, z, x, 2 * y, z, x**2, 2 * x * y, 3 * y**2], 1),
        ],
        1,
    )


def split(z, dz, fit, score):
    """Quadratic fit and best gradient part on the fit rows, their predictions on the score rows, gradient share."""
    W = np.linalg.lstsq(quad(z[fit]), dz[fit], rcond=None)[0]
    c = np.linalg.lstsq(gradient(z[fit]).reshape(-1, 9), dz[fit].reshape(-1), rcond=None)[0]
    full, grad = quad(z[score]) @ W, gradient(z[score]) @ c
    return W, c, 1 - ((full - grad) ** 2).sum() / max((full**2).sum(), 1e-9)


planes = {l: plane(S[l][s], S[l + 1][s]) for l in range(12)}
shares = {l: split(planes[l][0], planes[l][1], slice(None), slice(None))[2] for l in planes}
l = min(shares, key=shares.get)
z, dz, gx, gy, grid = planes[l]
rs = np.random.default_rng(1)
tr = rs.choice(len(z), int(0.8 * len(z)), replace=False)
te = np.setdiff1d(np.arange(len(z)), tr)
W, c, share = split(z, dz, tr, te)
full = (quad(grid) @ W).T.reshape(2, len(gx), len(gy))
grad = (gradient(grid) @ c).T.reshape(2, len(gx), len(gy))
U = (potential(grid) @ c).reshape(len(gx), len(gy))
smax = max(np.sqrt(f[0] ** 2 + f[1] ** 2).max() for f in (full, grad, full - grad))
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), sharex=True, sharey=True)
field(axes[0], gx, gy, *full, smax, f"full drift fit, L{l}$\\to$L{l + 1}", mean_steps(z[te], dz[te]))
axes[1].contour(*np.meshgrid(gx, gy), U, levels=10, colors=MUT, linewidths=0.5, alpha=0.5)
field(axes[1], gx, gy, *grad, smax, f"best gradient part ({100 * share:.0f}% of the fit)")
field(axes[2], gx, gy, *(full - grad), smax, f"remainder, the circulation ({100 * (1 - share):.0f}%)")
fig.tight_layout()
save(fig, "fig_helm")
