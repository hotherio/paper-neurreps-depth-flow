"""The three figures of the paper, written to paper/figures/. Figure 1 is every kept token's trajectory along the
leading comoving coordinate, coloured by its centrality rank at layer 0. Figure 2 is the leading principal plane of
layer 2 with the drift fitted as an affine map, as a quadratic map and by the local Kramers-Moyal estimate. Figure 3
is the quadratic drift fit of the transition with the largest in-plane rotation, split into the gradient of the best
cubic potential and the remainder."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

from depthflow import ROOT, data
from depthflow.instruments import km_drift, load_ladder

CRIM, INK, MUT = "#b13a5b", "#1a1626", "#57506b"
CMAP = LinearSegmentedColormap.from_list("crim", ["#e9d8de", "#c9829a", CRIM, "#6f2439"])
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    fig.savefig(os.path.join(ROOT, "paper", "figures", name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def plane(X, X1, n_grid=26):
    """The leading principal plane of the states X, the in-plane steps to X1, and a grid over the plane."""
    m = X.mean(0)
    P2 = np.linalg.svd(X - m, full_matrices=False)[2][:2]
    z, dz = (X - m) @ P2.T, (X1 - X) @ P2.T
    gx = np.linspace(np.quantile(z[:, 0], 0.01), np.quantile(z[:, 0], 0.99), n_grid)
    gy = np.linspace(np.quantile(z[:, 1], 0.01), np.quantile(z[:, 1], 0.99), n_grid)
    GX, GY = np.meshgrid(gx, gy)
    return z, dz, gx, gy, np.column_stack([GX.ravel(), GY.ravel()])


def mean_steps(z, dz, n_bins=6, least=25):
    """The mean held-out step per spatial bin of the plane, the object a drift field predicts."""
    bx, by = (np.quantile(z[:, i], np.linspace(0, 1, n_bins + 1)) for i in (0, 1))
    out = []
    for i in range(n_bins):
        for j in range(n_bins):
            m = (z[:, 0] >= bx[i]) & (z[:, 0] < bx[i + 1]) & (z[:, 1] >= by[j]) & (z[:, 1] < by[j + 1])
            if m.sum() >= least:
                out.append((z[m].mean(0), dz[m].mean(0)))
    return out


def field(ax, gx, gy, U, V, smax, title, steps=(), density=1.4, dot=18, lw=1.5):
    ax.streamplot(
        gx,
        gy,
        U,
        V,
        color=np.sqrt(U**2 + V**2),
        cmap=CMAP,
        density=density,
        linewidth=1.1,
        arrowsize=1.1,
        norm=plt.Normalize(0, smax),
    )
    for pos, step in steps:
        ax.annotate(
            "", xy=pos + step, xytext=pos, arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw, shrinkA=0, shrinkB=0)
        )
        ax.scatter(*pos, s=dot, color=INK, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


S = load_ladder(data("ladder_160m_wikitext.npz"))
N = S.shape[1]


def river():
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
        ax.scatter(
            np.full(120, x), lo + (np.arange(120) + 0.5) / 120 * (hi - lo), s=6, marker="s", c=cmap(cbin), zorder=3
        )
    ax.annotate("block 12 +\nnormalization", xy=(11.5, lo + 0.30 * (hi - lo)), fontsize=6.8, color=MUT, ha="center")
    ax.set(xlim=(-0.85, 12.85), ylim=(lo, hi), xticks=L, xlabel="layer", ylabel="leading comoving coordinate")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)), ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("centrality percentile at layer 0", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5)
    fig.tight_layout()
    save(fig, "fig_river")


def quadfield():
    s = np.random.default_rng(0).choice(N, 6000, replace=False)
    z, dz, gx, gy, grid = plane(S[2][s], S[3][s])
    tr, te = slice(None, 4800), slice(4800, None)
    affine = lambda p: np.column_stack([np.ones(len(p)), p])
    quad = lambda p: np.column_stack([affine(p), p[:, :1] ** 2, p[:, :1] * p[:, 1:], p[:, 1:] ** 2])
    maps = {
        name: np.linalg.lstsq(f(z[tr]), dz[tr], rcond=None)[0] for name, f in (("affine", affine), ("quadratic", quad))
    }
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


def helm():
    s = np.random.default_rng(0).choice(N, 6000, replace=False)
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
        """The quadratic fit and its best gradient part on the fit rows, and the gradient share of the fit on the
        score rows."""
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


river()
quadfield()
helm()
