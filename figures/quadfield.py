"""Figure 2: one principal plane of layer 2, the same held-out data in three panels, the drift fitted as an affine
map, a quadratic map and the local Kramers-Moyal estimate, with the mean actual step per bin in black."""

import numpy as np
from style import INK, field, mean_steps, plane, plt, save

from depthflow import data
from depthflow.instruments import km_drift, load_ladder

S = load_ladder(data("ladder_160m_wikitext.npz"))
s = np.random.default_rng(0).choice(S.shape[1], 6000, replace=False)
z, dz, gx, gy, grid = plane(S[2][s], S[3][s])
tr, te = slice(None, 4800), slice(4800, None)
affine = lambda p: np.column_stack([np.ones(len(p)), p])
quad = lambda p: np.column_stack([affine(p), p[:, :1] ** 2, p[:, :1] * p[:, 1:], p[:, 1:] ** 2])
maps = {name: np.linalg.lstsq(f(z[tr]), dz[tr], rcond=None)[0] for name, f in (("affine", affine), ("quadratic", quad))}
predict = {name: (lambda p, f=f, W=maps[name]: f(p) @ W) for name, f in (("affine", affine), ("quadratic", quad))}
predict["local (KM)"] = lambda p: km_drift(z[tr], dz[tr], p)

steps = mean_steps(z[te], dz[te])
fields = {n: fn(grid).T.reshape(2, len(gx), len(gy)) for n, fn in predict.items()}
smax = max(np.sqrt(U**2 + V**2).max() for U, V in fields.values())
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), sharex=True, sharey=True)
for ax, name in zip(axes, predict):
    P, A = predict[name](np.array([b[0] for b in steps])), np.array([b[1] for b in steps])
    cos = ((P * A).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(A, axis=1))).mean()
    field(
        ax,
        gx,
        gy,
        *fields[name],
        smax,
        f"{name} drift fit, L2$\\to$L3 (mean cosine to mean steps: {cos:.2f})",
        steps,
        density=1.5,
        dot=22,
        lw=1.6,
    )
axes[0].scatter([], [], s=22, color=INK, label="held-out tokens, mean actual step per bin")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
save(fig, "fig_quadfield")
