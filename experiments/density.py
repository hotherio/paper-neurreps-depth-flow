"""The density-gradient test (Section 5, Appendix A, Table 3): the cosine gap between increments and the estimated
score direction against a shuffled pairing, at 16 components and wider, with the common translation removed, on
Pythia-410M, and the sign test for any diffusion matrix with its controls. The Ornstein-Uhlenbeck control gives
the ceiling of the instrument."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import (
    affine_features,
    alignment_gap,
    basis,
    fit,
    km_drift,
    knn,
    load_ladder,
    project,
    score_dir,
)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def probe(pts, inc, rng):
    return alignment_gap(score_dir(pts, exclude_self=False), inc, rng)


def readings(v, g, rng):
    """Positive fraction and mean cosine of v against g, each with its value under a shuffled pairing."""
    c = (unit(v) * unit(g)).sum(1)
    cs = (unit(v) * unit(g[rng.permutation(len(g))])).sum(1)
    return dict(
        pos=round(float((c > 0).mean()), 4),
        pos_null=round(float((cs > 0).mean()), 4),
        cos_gap=round(float(c.mean() - cs.mean()), 4),
    )


def local_cov_times(Ztr, dtr, Zte, g, k=64, shrink=0.1):
    """The score g multiplied by the local covariance of the k nearest training steps."""
    nb = dtr[knn(Ztr, Zte, k)]
    E = nb - nb.mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", E, E) / (k - 1)
    d = np.einsum("nii->ni", C).copy()
    C *= 1 - shrink
    i = np.arange(C.shape[1])
    C[:, i, i] = d
    return np.einsum("nij,nj->ni", C, g)


def sign_test(X, X1, tr, te, rng, local=False):
    """The sign test and the cosine gaps of one transition, for the increment and the local mean step, each with
    and without the common translation. The cosine gap is also read with the score multiplied by the affine
    residual covariance and, at 16 components, by the local Kramers-Moyal covariance."""
    g = score_dir(X)[te]
    inc = X1 - X
    step = inc[tr].mean(0)
    drift = km_drift(X[tr], inc[tr], X[te])
    W = fit(affine_features, X[tr], X1[tr])
    C = np.cov((X1[tr] - affine_features(X[tr]) @ W).T)
    out = {}
    for name, v in (
        ("increment", inc[te]),
        ("increment_centred", inc[te] - step),
        ("drift", drift),
        ("drift_centred", drift - step),
    ):
        out[name] = readings(v, g, rng)
        out[name]["cos_gap_D"] = readings(v, g @ C, rng)["cos_gap"]
        if local:
            out[name]["cos_gap_Dlocal"] = readings(v, local_cov_times(X[tr], inc[tr], X[te], g), rng)["cos_gap"]
    return out


def control(D, N, tr, te, kind, ratio, seed=3):
    """One Euler step of dx = (D grad log p + div D) dt + sqrt(2 D) dW from a stationary Gaussian cloud."""
    rng = np.random.default_rng(seed)
    sig = np.exp(np.linspace(np.log(1.0), np.log(4.0), D))
    X = rng.standard_normal((N, D)) * sig
    s = -X / sig**2
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    lam = np.exp(np.linspace(0.0, np.log(16.0), D))
    Db = np.eye(D) if kind == "isotropic" else (Q * lam) @ Q.T * (D / lam.sum())
    c = (X**2).sum(1).mean()
    phi = 1 + (X**2).sum(1, keepdims=True) / c if kind == "state_dependent" else np.ones((N, 1))
    a = phi * (s @ Db) + (2 / c * (X @ Db) if kind == "state_dependent" else 0)
    noise = np.sqrt(phi) * (rng.standard_normal((N, D)) @ np.linalg.cholesky(Db).T)
    dt = (ratio * np.median(np.linalg.norm(noise, axis=1)) * np.sqrt(2) / np.median(np.linalg.norm(a, axis=1))) ** 2
    X1 = X + a * dt + np.sqrt(2 * dt) * noise
    out = sign_test(X.astype(np.float32), X1.astype(np.float32), tr, te, np.random.default_rng(seed + 1), D == 16)
    out["true_drift"] = readings(a[te], score_dir(X.astype(np.float32))[te], np.random.default_rng(seed + 2))
    return out


S = load_ladder(data("ladder_160m_wikitext.npz"))
mu, Vt, tr, te, rng = basis(S)
N = S.shape[1]
Z = project(S, mu, Vt, 16)
res = {"gap_16": [round(alignment_gap(score_dir(Z[l]), Z[l + 1] - Z[l], rng), 4) for l in range(12)], "gap_wide": {}}
for D in (32, 64, 128):
    ZD, r = project(S, mu, Vt, D), np.random.default_rng(0)
    res["gap_wide"][D] = [round(alignment_gap(score_dir(ZD[l]), ZD[l + 1] - ZD[l], r), 4) for l in range(12)]
print("gap at 16:", res["gap_16"], flush=True)

res["translation"] = []
for l in range(12):
    inc = Z[l + 1] - Z[l]
    m = inc.mean(0)
    res["translation"].append(
        dict(
            raw_gap=round(probe(Z[l], inc, np.random.default_rng(0)), 4),
            centred_gap=round(probe(Z[l], inc - m, np.random.default_rng(0)), 4),
            share=round(float((m**2).sum() / (inc**2).sum(1).mean()), 3),
        )
    )
g = np.random.default_rng(3)
sig = np.exp(np.linspace(np.log(1.0), np.log(4.0), 16))
X = g.standard_normal((N, 16)) * sig
a = -X / sig**2
dt = (5.0 * np.sqrt(32) / np.median(np.linalg.norm(a, axis=1))) ** 2
inc = a * dt + np.sqrt(2 * dt) * g.standard_normal((N, 16))
tdir = g.standard_normal(16)
share = float(np.median([t["share"] for t in res["translation"]]))
inc_t = inc + np.sqrt(share * (inc**2).sum(1).mean() / (1 - share)) * tdir / np.linalg.norm(tdir)
res["ou_control"] = dict(
    gap=round(probe(X, inc, np.random.default_rng(4)), 4),
    translated_gap=round(probe(X, inc_t, np.random.default_rng(4)), 4),
    translated_centred_gap=round(probe(X, inc_t - inc_t.mean(0), np.random.default_rng(4)), 4),
)
print("OU control:", res["ou_control"], flush=True)

S4 = load_ladder(data("ladder_410m_wikitext.npz"))
mu4, Vt4, *_ = basis(S4)
Z4, r = project(S4, mu4, Vt4, 16), np.random.default_rng(0)
res["gap_410m"] = [round(alignment_gap(score_dir(Z4[l]), Z4[l + 1] - Z4[l], r), 4) for l in range(len(Z4) - 1)]

res["sign_test"], res["sign_controls"] = {}, {}
for D in (16, 32, 64, 128):
    ZD = project(S, mu, Vt, D)
    res["sign_test"][D] = [
        sign_test(ZD[l], ZD[l + 1], tr, te, np.random.default_rng(100 + l), D == 16) for l in range(12)
    ]
    res["sign_controls"][D] = {
        f"{kind}_ratio{ratio:g}": control(D, N, tr, te, kind, ratio)
        for kind in ("isotropic", "anisotropic", "state_dependent")
        for ratio in (2.0, 5.0)
    }
    print(
        f"[sign test {D}] drift positive {np.mean([t['drift']['pos'] for t in res['sign_test'][D]]):.3f} | controls "
        f"{min(c['drift']['pos'] for c in res['sign_controls'][D].values()):.2f}",
        flush=True,
    )
save("density", res)
