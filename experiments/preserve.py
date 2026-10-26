"""What the flow preserves (Section 6): the intraclass correlation of the per-layer z-scores and the Spearman
correlation between entry and exit, for four token observables on the raw states, on three sets of mean
trajectories. A planted persistent rank and its shuffle calibrate the statistic."""

import numpy as np
from scipy.stats import spearmanr

from depthflow import data, save

SETS = {
    "wikitext_160m": "ladder_160m_wikitext.npz",
    "pile_160m": "ladder_160m_pile.npz",
    "wikitext_410m": "ladder_410m_wikitext.npz",
}


def kth_distance(X, k=10, chunk=2000):
    sq = (X**2).sum(1)
    out = np.empty(len(X), np.float32)
    for a in range(0, len(X), chunk):
        d2 = sq[a : a + chunk, None] - 2 * X[a : a + chunk] @ X.T + sq[None]
        out[a : a + chunk] = np.sqrt(np.maximum(np.partition(d2, k, axis=1)[:, k], 0))
    return out


def observables(S):
    norms = np.linalg.norm(S, axis=2)
    mdir = S.mean(1)
    mdir /= np.linalg.norm(mdir, axis=1, keepdims=True)
    return {
        "cos_mean": np.einsum("lnd,ld->ln", S, mdir) / np.maximum(norms, 1e-9),
        "norm": norms,
        "participation": (S**2).sum(2) ** 2 / np.maximum((S**4).sum(2), 1e-9),
        "density": -np.stack([kth_distance(S[l]) for l in range(S.shape[0])]),
    }


def icc(Q):
    Zq = (Q - Q.mean(1, keepdims=True)) / np.maximum(Q.std(1, keepdims=True), 1e-9)
    between, within = Zq.mean(0).var(), Zq.var(0).mean()
    return float(between / (between + within))


def stats(Q):
    return dict(icc=round(icc(Q), 3), spearman_entry_exit=round(float(spearmanr(Q[0], Q[-1]).statistic), 3))


rng = np.random.default_rng(0)
planted = np.tile(rng.standard_normal(5000), (13, 1)) + 0.1 * rng.standard_normal((13, 5000))
res = {
    "instrument": dict(
        planted=round(icc(planted), 3),
        shuffled=round(icc(np.stack([rng.permutation(planted[0]) for _ in range(13)])), 3),
    )
}
for name, path in SETS.items():
    z = np.load(data(path))
    keep = np.where(z["cnt"] >= 5)[0]
    S = z["S"].astype(np.float32)[:, keep]
    res[name] = dict(n_trajectories=int(len(keep)), **{k: stats(Q) for k, Q in observables(S).items()})
    print(f"[{name}]", {k: res[name][k] for k in ("cos_mean", "norm", "participation", "density")}, flush=True)
save("preserve", res)
