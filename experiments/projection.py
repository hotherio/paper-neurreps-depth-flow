"""The projection control (Section 4, Table 4): flows that are affine at full width by construction, read through
the same camera as the real flow. The affine arm keeps the fitted 768-d map and resamples its residuals. The
symmetric arm keeps only the symmetric part of the map, which is a gradient flow at full width. The planted arm
adds a quadratic term of known size. The cross-fitted arm predicts every training token with a map fitted on the
other half. The Gaussian arm replaces the current states by Gaussian draws with the real mean and covariance."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import (
    CubicPotential,
    affine_features,
    alignment_gap,
    camera,
    circulation,
    contest,
    km_drift,
    load_ladder,
    project,
    r2,
    score_dir,
)

WIDTHS = (8, 16, 32, 64)


def battery(Z, Z1, tr, te, pot, full=True):
    """Quadratic against affine at every width, and at 16 the local estimator, the score gap and the rotation."""
    out = {}
    for D in WIDTHS:
        ra, rq = contest(Z[..., :D], Z1[..., :D], tr, te)
        out[D] = dict(
            quad_wins=int((rq > ra).sum()), margin=np.round(rq - ra, 4).tolist(), r2_affine=np.round(ra, 4).tolist()
        )
    if full:
        Z, Z1, rng = Z[..., :16], Z1[..., :16], np.random.default_rng(0)
        out["km_r2"] = [
            round(
                r2(
                    Z[l][te] + km_drift(Z[l][tr], Z1[l][tr] - Z[l][tr], Z[l][te]),
                    Z[l][te],
                    Z1[l][te],
                    (Z1[l][tr] - Z[l][tr]).mean(0),
                ),
                4,
            )
            for l in range(12)
        ]
        out["score_gap"] = [round(alignment_gap(score_dir(Z[l]), Z1[l] - Z[l], rng), 4) for l in range(12)]
        out["rotational"] = [round(circulation(Z[l], Z1[l], tr, te, pot)[2], 4) for l in range(12)]
    return out


S = load_ladder(data("ladder_160m_wikitext.npz"))
mu, Vt, tr, te, _ = camera(S)
N = S.shape[1]
P = Vt[:64].astype(np.float64).T
pot = CubicPotential(16)
Z = project(S, mu, Vt, 64)
rng = np.random.default_rng(11)
arms = {k: np.empty((12, N, 64), np.float32) for k in ("affine", "symmetric", "planted")}
res = {"fullwidth_affine_r2": []}
for l in range(12):
    X, X1 = S[l].astype(np.float64), S[l + 1].astype(np.float64)
    m0 = X[tr].mean(0)
    W = np.linalg.lstsq(affine_features(X[tr] - m0), X1[tr], rcond=None)[0]
    A, b = W[:-1], W[-1]
    pred = (X - m0) @ A + b
    resid = X1 - pred
    donor = np.empty(N, np.int64)
    donor[tr], donor[te] = rng.permutation(tr), rng.permutation(te)
    noise = resid[donor]
    step_sym = (X - m0) @ ((A - np.eye(768)) + (A - np.eye(768)).T) / 2 + (b - m0)
    q = ((X - m0) @ Vt[1].astype(np.float64)) ** 2
    q -= q[tr].mean()
    centred = X1 - X - (X1 - X)[tr].mean(0)
    gamma = np.sqrt(0.10 * (centred[tr] ** 2).sum(1).mean() / (q[tr] ** 2).mean())
    arms["affine"][l] = (pred + noise - mu) @ P
    arms["symmetric"][l] = (X + step_sym + noise - mu) @ P
    arms["planted"][l] = (pred + noise + gamma * q[:, None] * Vt[2].astype(np.float64) - mu) @ P
    res["fullwidth_affine_r2"].append(round(float(1 - (resid[te] ** 2).sum() / (centred[te] ** 2).sum()), 4))
res["real"] = battery(Z[:-1], Z[1:], tr, te, pot)
for name, Z1 in arms.items():
    res[name] = battery(Z[:-1], Z1, tr, te, pot)
    print(
        f"[{name}] quad wins "
        + " ".join(f"{D}: {res[name][D]['quad_wins']}/12" for D in WIDTHS)
        + f" | rotational {min(res[name]['rotational']):.2f}-{max(res[name]['rotational']):.2f}",
        flush=True,
    )

rng = np.random.default_rng(11)
half = rng.permutation(tr)
h1, h2 = half[: len(tr) // 2], half[len(tr) // 2 :]
Zin = {k: np.empty((12, N, 64), np.float32) for k in ("crossfit", "gaussian")}
Zout = {k: np.empty((12, N, 64), np.float32) for k in ("crossfit", "gaussian")}
for l in range(12):
    X, X1 = S[l].astype(np.float64), S[l + 1].astype(np.float64)
    m0 = X[tr].mean(0)
    F = affine_features(X - m0)
    fit_on = lambda idx, F=F, X1=X1: np.linalg.lstsq(F[idx], X1[idx], rcond=None)[0]
    W = fit_on(tr)
    pred = F @ W
    pred[h1], pred[h2] = F[h1] @ fit_on(h2), F[h2] @ fit_on(h1)
    donor = np.empty(N, np.int64)
    donor[tr], donor[te] = rng.permutation(tr), rng.permutation(te)
    Zin["crossfit"][l], Zout["crossfit"][l] = (X - mu) @ P, (pred + (X1 - pred)[donor] - mu) @ P
    C = np.cov(X.T)
    Xg = X.mean(0) + rng.standard_normal((N, 768)) @ np.linalg.cholesky(C + 1e-6 * np.trace(C) / 768 * np.eye(768)).T
    Zin["gaussian"][l], Zout["gaussian"][l] = (
        (Xg - mu) @ P,
        (affine_features(Xg - m0) @ W + (X1 - F @ W)[donor] - mu) @ P,
    )
for name in Zin:
    res[name] = battery(Zin[name], Zout[name], tr, te, pot, full=False)
    print(f"[{name}] quad wins " + " ".join(f"{D}: {res[name][D]['quad_wins']}/12" for D in WIDTHS), flush=True)
save("projection", res)
