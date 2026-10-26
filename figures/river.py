"""Figure 1: every kept token's trajectory along the leading comoving coordinate, coloured by its centrality rank at
layer 0, with decile medians and the colour order at both ends."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from style import CRIM, INK, MUT, plt, save

from depthflow import data
from depthflow.instruments import load_ladder

S = load_ladder(data("ladder_160m_wikitext.npz"))
N = S.shape[1]
mdir = S.mean(1)
mdir /= np.linalg.norm(mdir, axis=1, keepdims=True)
crank = rankdata(np.einsum("lnd,ld->ln", S, mdir)[0] / np.linalg.norm(S[0], axis=1)) / N

Sz = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-6)
rng = np.random.default_rng(0)
X = Sz[:, rng.choice(N, 8000, replace=False)].reshape(-1, S.shape[2])
Vt = np.linalg.svd(X[rng.choice(len(X), 20000, replace=False)], full_matrices=False)[2]
Y = np.einsum("lnd,d->ln", Sz, Vt[0])
if np.median(Y[0][crank > 0.9]) < np.median(Y[0][crank < 0.1]):
    Y = -Y

cmap = LinearSegmentedColormap.from_list("river", ["#efe3e8", "#d09aac", CRIM, "#7d2740", INK])
draw = rng.choice(N, 2600, replace=False)
draw = draw[np.argsort(crank[draw])]
L = np.arange(13)
fig, ax = plt.subplots(figsize=(8.0, 3.9))
ax.axvspan(11, 12, color="#f5eef1", zorder=0)
for i in draw:
    ax.plot(
        L,
        Y[:, i],
        lw=0.5,
        alpha=0.05 + 0.10 * crank[i],
        color=cmap(crank[i]),
        solid_capstyle="round",
        zorder=2,
        rasterized=True,
    )
edges = np.linspace(0, 1, 11)
for a, b in zip(edges[:-1], edges[1:]):
    ax.plot(
        L,
        np.median(Y[:, (crank >= a) & (crank < b)], axis=1),
        lw=2.4,
        alpha=0.95,
        color=cmap((a + b) / 2),
        solid_capstyle="round",
        zorder=4,
    )
med = np.median(Y[:, (crank >= 0.2) & (crank < 0.3)], axis=1)
ax.annotate(
    "decile medians",
    xy=(6, med[6]),
    xytext=(5.2, med[6] - 5.5),
    fontsize=7.5,
    color=MUT,
    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.7),
)
lo, hi = np.quantile(Y, 0.01), np.quantile(Y, 0.999)
for x, l in ((-0.55, 0), (12.55, 12)):
    cbin = np.array([np.median(crank[c]) for c in np.array_split(np.argsort(Y[l]), 120)])
    ax.scatter(np.full(120, x), lo + (np.arange(120) + 0.5) / 120 * (hi - lo), s=6, marker="s", c=cmap(cbin), zorder=3)
ax.annotate("block 12 +\nnormalization", xy=(11.5, lo + 0.30 * (hi - lo)), fontsize=6.8, color=MUT, ha="center")
ax.set(xlim=(-0.85, 12.85), ylim=(lo, hi), xticks=L, xlabel="layer", ylabel="leading comoving coordinate")
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)), ax=ax, pad=0.015, fraction=0.035)
cb.set_label("centrality percentile at layer 0", fontsize=8)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=7.5)
fig.tight_layout()
save(fig, "fig_river")
