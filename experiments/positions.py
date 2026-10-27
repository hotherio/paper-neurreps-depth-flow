"""The contests on the un-averaged flow (Sections 4 and 5): every position of the corpus in the paper's 16-component
principal basis, split by document. The local estimator is fitted on one million training positions and scored on
200,000 test positions, the maps on all of them, and the density-gradient test on 60,000 positions."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import alignment_gap, km_ll, knn, quad_field, score_dir

z = np.load(data("reduced_160m_wikitext.npz"))
Z, doc = z["Z"], z["doc"]
P = Z.shape[1]
train_docs = set(np.random.default_rng(0).choice(25000, 20000, replace=False).tolist())
mtr = np.fromiter((d in train_docs for d in doc), bool, len(doc))
itr, ite = np.where(mtr)[0], np.where(~mtr)[0]
print(f"{P} positions, {len(itr)} train / {len(ite)} test", flush=True)
affine = lambda X: np.column_stack([np.ones(len(X)), X])


def lstsq(feat, X, Y):
    m = feat(X[:1]).shape[1]
    AtA, Aty = np.zeros((m, m)), np.zeros((m, Y.shape[1]))
    for a in range(0, len(X), 200_000):
        F = feat(X[a : a + 200_000]).astype(np.float64)
        AtA += F.T @ F
        Aty += F.T @ Y[a : a + 200_000].astype(np.float64)
    return np.linalg.solve(AtA + 1e-8 * np.eye(m), Aty).astype(np.float32)


def predict(feat, W, X):
    return np.concatenate([feat(X[a : a + 200_000]) @ W for a in range(0, len(X), 200_000)])


res = {"n_positions": int(P), "transitions": []}
r = np.random.default_rng(1)
str_, ste = r.choice(len(itr), 1_000_000, replace=False), r.choice(len(ite), 200_000, replace=False)
for l in range(12):
    Xtr, Xte = Z[l][itr].astype(np.float32), Z[l][ite].astype(np.float32)
    dtr, dte = Z[l + 1][itr].astype(np.float32) - Xtr, Z[l + 1][ite].astype(np.float32) - Xte
    ss0 = ((dte - dtr.mean(0)) ** 2).sum()
    pa, pq = predict(affine, lstsq(affine, Xtr, dtr), Xte), predict(quad_field, lstsq(quad_field, Xtr, dtr), Xte)
    nb = dtr[str_][knn(Xtr[str_], Xte[ste], 64)]
    dq = dte[ste]
    res["transitions"].append(
        dict(
            r2_affine=round(float(1 - ((dte - pa) ** 2).sum() / ss0), 3),
            r2_quad=round(float(1 - ((dte - pq) ** 2).sum() / ss0), 3),
            r2_km=round(float(1 - ((dq - nb.mean(1)) ** 2).sum() / ((dq - dtr.mean(0)) ** 2).sum()), 3),
            ll_km=round(float(km_ll(nb, dq).mean()), 2),
        )
    )
    print(f"[L{l}->L{l + 1}]", res["transitions"][-1], flush=True)

res["gradient"] = []
g = np.random.default_rng(0)
sg = g.choice(P, 60_000, replace=False)
for l in range(12):
    pts = Z[l][sg].astype(np.float32)
    inc = Z[l + 1][sg].astype(np.float32) - pts
    score = score_dir(pts, exclude_self=False)
    res["gradient"].append(
        dict(
            raw_gap=round(alignment_gap(score, inc, g), 4),
            centred_gap=round(alignment_gap(score, inc - inc.mean(0), g), 4),
        )
    )
print("centred gaps:", [t["centred_gap"] for t in res["gradient"]], flush=True)
save("positions", res)
