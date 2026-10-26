"""The split of the drift into the gradient of a cubic potential and a remainder (Section 5): the rotational part
per transition with its two controls, its sensitivity to the ridge of the potential fit, and the symmetric share
of the linear map against row-permutation nulls."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import CubicPotential, camera, circulation, load_ladder, project


def row(r2f, r2g, circ):
    return dict(r2_quad=round(r2f, 4), r2_grad=round(r2g, 4), rotational=round(circ, 4))


S = load_ladder(data("ladder_160m_wikitext.npz"))
mu, Vt, tr, te, _ = camera(S)
N = S.shape[1]
Z = project(S, mu, Vt, 16)
pot = CubicPotential(16)
res = {"transitions": [row(*circulation(Z[l], Z[l + 1], tr, te, pot)) for l in range(12)]}
print("rotational part:", [t["rotational"] for t in res["transitions"]], flush=True)

g = np.random.default_rng(3)
sig = np.exp(np.linspace(np.log(1.0), np.log(4.0), 16)).astype(np.float32)
X = g.standard_normal((N, 16)).astype(np.float32) * sig
a = -X / sig**2
dt = (5.0 * np.sqrt(32) / np.median(np.linalg.norm(a, axis=1))) ** 2
inc = (a * dt + np.sqrt(2 * dt) * g.standard_normal((N, 16))).astype(np.float32)
res["ou_control"] = row(*circulation(X, X + inc, tr, te, pot, centre=False))
R = g.standard_normal((16, 16)).astype(np.float32)
a_rot = X @ ((R - R.T) / 2).T
a_rot *= np.median(np.linalg.norm(a, axis=1)) / np.median(np.linalg.norm(a_rot, axis=1))
inc = (a_rot * dt + np.sqrt(2 * dt) * g.standard_normal((N, 16))).astype(np.float32)
res["rotation_control"] = row(*circulation(X, X + inc, tr, te, pot, centre=False))
print("controls:", res["ou_control"], res["rotation_control"], flush=True)

r = np.random.default_rng(0)
keep = r.choice(N, 25000, replace=False) if N > 25000 else np.arange(N)
tr_r = r.choice(keep, int(0.8 * len(keep)), replace=False)
te_r = np.setdiff1d(keep, tr_r)
res["ridge"] = {
    ridge: [row(*circulation(Z[l], Z[l + 1], tr_r, te_r, pot, ridge, relative=True)) for l in range(12)]
    for ridge in (1e-4, 1e-3)
}
print("ridge:", {k: v[11]["rotational"] for k, v in res["ridge"].items()}, flush=True)

r = np.random.default_rng(0)
res["symmetric_share"] = []
for l in range(12):
    V = Z[l + 1] - Z[l]
    PX = np.linalg.pinv(Z[l] - Z[l].mean(0))
    share = lambda A: float(((0.5 * (A + A.T)) ** 2).sum() / max((A**2).sum(), 1e-12))
    nulls = [share(PX @ V[r.permutation(len(V))]) for _ in range(10)]
    res["symmetric_share"].append(
        dict(share=round(share(PX @ V), 4), null_min=round(min(nulls), 4), null_max=round(max(nulls), 4))
    )
print("symmetric share:", [t["share"] for t in res["symmetric_share"]], flush=True)
save("potential", res)
