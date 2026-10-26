"""Shared style of the three figures, and the two-dimensional drift pictures they draw."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from depthflow import ROOT

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
