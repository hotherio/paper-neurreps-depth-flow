"""The contests on the mean trajectories (Section 4, Table 1, Table 6): the per-transition likelihoods and R2 at 16
components with bootstrap intervals and two more seeds, the width ladder on Pythia-160M and Pythia-410M with the
matched-budget rows, the neighbourhood sweep of the local estimator, the variance retained by the camera, the
Markov test, and the rollouts with their controls."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import (
    affine_features,
    camera,
    contest,
    fit,
    gauss_diag_ll,
    gauss_full_ll,
    km_ll,
    knn,
    load_ladder,
    project,
    quad_features,
    r2,
)


def likelihood_rows(Z, tr, te, boot):
    """Per transition at one camera: the likelihood contest of KM-full against the affine map with a full residual
    covariance, and at seed 0 the whole likelihood table, the R2 columns and the bootstrap interval of the margin."""
    rows = []
    dall = (Z[1:] - Z[:-1])[:, tr].reshape(-1, Z.shape[2])
    for l in range(len(Z) - 1):
        Ztr, Ztr1, Zte, Zte1 = Z[l][tr], Z[l + 1][tr], Z[l][te], Z[l + 1][te]
        dtr, dte, step = Ztr1 - Ztr, Zte1 - Zte, (Ztr1 - Ztr).mean(0)
        nb = dtr[knn(Ztr, Zte, 64)]
        Wa, Wq = fit(affine_features, Ztr, Ztr1), fit(quad_features, Ztr, Ztr1)
        pa = affine_features(Zte) @ Wa
        Rtr = Ztr1 - affine_features(Ztr) @ Wa
        ll = {
            "KM": km_ll(nb, dte),
            "KM-full": km_ll(nb, dte, full=True),
            "Null A": gauss_diag_ll(dtr, dte),
            "Null B": gauss_diag_ll(dall, dte),
            "Null C": gauss_full_ll(dtr, dte),
            "Affine": gauss_diag_ll(Rtr, Zte1 - pa),
            "Affine-full": gauss_full_ll(Rtr, Zte1 - pa),
        }
        pw = ll["KM-full"] - ll["Affine-full"]
        bidx = boot.integers(0, len(pw), (1000, len(pw)))
        rows.append(
            dict(
                {k: round(float(v.mean()), 2) for k, v in ll.items()},
                margin=round(float(pw.mean()), 2),
                margin_ci=np.round(np.quantile(pw[bidx].mean(1), [0.025, 0.975]), 2).tolist(),
                r2_km=round(r2(Zte + nb.mean(1), Zte, Zte1, step), 3),
                r2_affine=round(r2(pa, Zte, Zte1, step), 3),
                r2_quad=round(r2(quad_features(Zte) @ Wq, Zte, Zte1, step), 3),
            )
        )
    return rows


def widths(S, mu, Vt, tr, te, D):
    Z = project(S, mu, Vt, D)
    ra, rq = contest(Z[:-1], Z[1:], tr, te)
    return Z, dict(
        quad_wins=int((rq > ra).sum()),
        r2_quad=np.round(rq, 4).tolist(),
        r2_affine=np.round(ra, 4).tolist(),
        r2_quad_median=round(float(np.median(rq)), 3),
        r2_affine_median=round(float(np.median(ra)), 3),
    )


def km_rollout(Z, tr, te, rng, affine=False):
    """Held-out tokens pushed from their real first state by the fitted law, fresh noise at each step; with affine,
    the composed affine maps are rolled out in the same loop with the same generator."""
    x = xa = Z[0][te].copy()
    ratios = []
    for l in range(len(Z) - 1):
        d = (Z[l + 1] - Z[l])[tr]
        nb = d[knn(Z[l][tr], x, 64)]
        x = x + nb.mean(1) + np.sqrt(nb.var(1) + 1e-6) * rng.standard_normal(x.shape)
        if affine:
            W = fit(affine_features, Z[l][tr], Z[l + 1][tr])
            rs = np.sqrt((Z[l + 1][tr] - affine_features(Z[l][tr]) @ W).var(0) + 1e-6)
            xa = affine_features(xa) @ W + rs * rng.standard_normal(xa.shape)
        ratios.append(round(float(x.var(0).sum() / Z[l + 1][te].var(0).sum()), 3))
    return ratios, round(float(xa.var(0).sum() / Z[-1][te].var(0).sum()), 3)


def ar1_control(N, tr, te, rng, drift_mult):
    """A stationary AR(1) cloud in 16 dimensions, translating by drift_mult noise scales per step, through the
    same rollout."""
    D, phi = 16, 0.7
    sig = np.exp(np.linspace(np.log(1.0), np.log(4.0), D))
    noise = np.sqrt(1 - phi**2) * sig
    steps = rng.standard_normal((12, D))
    steps *= drift_mult * np.linalg.norm(noise) / np.linalg.norm(steps, axis=1, keepdims=True)
    M = np.vstack([np.zeros((1, D)), np.cumsum(steps, 0)])
    T = np.zeros((13, N, D), np.float32)
    T[0] = rng.standard_normal((N, D)) * sig
    for t in range(12):
        T[t + 1] = M[t + 1] + phi * (T[t] - M[t]) + noise * rng.standard_normal((N, D))
    return km_rollout(T, tr, te, np.random.default_rng(9))[0][-1]


def identity_rollout(Z, tr, te, rng):
    """The rollout with one training token's residual path per simulated trajectory instead of fresh noise, and
    the fresh-noise rollout beside it."""
    steps, resid = [], []
    for l in range(12):
        d = Z[l + 1][tr] - Z[l][tr]
        resid.append(d - d[knn(Z[l][tr], Z[l][tr], 65)[:, 1:]].mean(1))
        steps.append(d)
    w = rng.choice(len(tr), len(te))
    x_id = x_iid = Z[0][te].copy()
    out = {"identity_path": [], "iid": []}
    for l in range(12):
        x_id = x_id + steps[l][knn(Z[l][tr], x_id, 64)].mean(1) + resid[l][w]
        nb = steps[l][knn(Z[l][tr], x_iid, 64)]
        x_iid = x_iid + nb.mean(1) + rng.standard_normal(x_iid.shape).astype(np.float32) * np.sqrt(nb.var(1))
        real = Z[l + 1][te].var(0).sum()
        out["identity_path"].append(round(float(x_id.var(0).sum() / real), 3))
        out["iid"].append(round(float(x_iid.var(0).sum() / real), 3))
    return out


S = load_ladder(data("ladder_160m_wikitext.npz"))
N = S.shape[1]
res = {"n_trajectories": int(N), "seeds": {}}
for seed in (0, 1, 2):
    mu, Vt, tr, te, _ = camera(S, seed)
    rows = likelihood_rows(project(S, mu, Vt, 16), tr, te, np.random.default_rng(11))
    res["seeds"][seed] = dict(kmfull_beats_affinefull=sum(r["margin"] > 0 for r in rows), rows=rows)
    print(f"[seed {seed}] KM-full beats affine-full {res['seeds'][seed]['kmfull_beats_affinefull']}/12", flush=True)

mu, Vt, tr, te, _ = camera(S)
res["n_train"] = int(len(tr))
S4 = load_ladder(data("ladder_410m_wikitext.npz"))
mu4, Vt4, tr4, te4, _ = camera(S4)
res["pythia410m"] = {"n_trajectories": int(S4.shape[1]), "n_train": int(len(tr4)), "widths": {}}
res["widths"], res["matched_budget"] = {}, {}
sub = np.random.default_rng(99)
for D in (8, 16, 32, 64):
    Z, res["widths"][D] = widths(S, mu, Vt, tr, te, D)
    res["widths"][D]["knn_wins"] = sum(
        r2(
            Z[l][te] + (Z[l + 1][tr] - Z[l][tr])[knn(Z[l][tr], Z[l][te], 64)].mean(1),
            Z[l][te],
            Z[l + 1][te],
            (Z[l + 1][tr] - Z[l][tr]).mean(0),
        )
        > res["widths"][D]["r2_affine"][l]
        for l in range(12)
    )
    far = Z[6][tr][knn(Z[6][tr], Z[6][te], 64)[:, -1]]
    res["widths"][D]["locality"] = round(
        float(
            np.median(np.linalg.norm(far - Z[6][te], axis=1))
            / np.sqrt(((Z[6][te] - Z[6][te].mean(0)) ** 2).sum(1).mean())
        ),
        3,
    )
    res["pythia410m"]["widths"][D] = widths(S4, mu4, Vt4, tr4, te4, D)[1]
    if D >= 32:
        res["matched_budget"][D] = widths(S, mu, Vt, sub.choice(tr, len(tr4), replace=False), te, D)[1]
    print(
        f"[width {D}] quad {res['widths'][D]['quad_wins']}/12 knn {res['widths'][D]['knn_wins']}/12 "
        f"locality {res['widths'][D]['locality']} | 410M quad {res['pythia410m']['widths'][D]['quad_wins']}/24",
        flush=True,
    )
Z64 = project(S, mu, Vt, 64)
res["ksweep_64"] = {
    k: sum(
        r2(
            Z64[l][te] + (Z64[l + 1][tr] - Z64[l][tr])[knn(Z64[l][tr], Z64[l][te], k)].mean(1),
            Z64[l][te],
            Z64[l + 1][te],
            (Z64[l + 1][tr] - Z64[l][tr]).mean(0),
        )
        > res["widths"][64]["r2_affine"][l]
        for l in range(12)
    )
    for k in (256, 1024)
}

pool = S[:, np.random.default_rng(0).choice(N, 8000, replace=False)].reshape(-1, S.shape[2]) - mu
res["retention"] = dict(
    pooled_16=round(float((pool @ Vt[:16].T).var(0).sum() / pool.var(0).sum()), 4),
    per_layer_16=[
        round(float(((S[l] - mu) @ Vt[:16].T).var(0).sum() / (S[l] - mu).var(0).sum()), 4) for l in range(13)
    ],
)

Z, Z32 = project(S, mu, Vt, 16), project(S, mu, Vt, 32)
hist16 = hist32 = 0
for l in range(1, 12):
    d, dte = (Z[l + 1] - Z[l])[tr], (Z[l + 1] - Z[l])[te]
    ll = lambda Qtr, Qte, d=d, dte=dte: float(km_ll(d[knn(Qtr, Qte, 64)], dte).mean())
    llh = ll(np.hstack([Z[l][tr], Z[l - 1][tr]]), np.hstack([Z[l][te], Z[l - 1][te]]))
    hist16 += llh > ll(Z[l][tr], Z[l][te])
    hist32 += llh > ll(Z32[l][tr], Z32[l][te])
res["markov"] = dict(history_beats_16=int(hist16), history_beats_32=int(hist32))

km, _ = km_rollout(Z, tr, te, np.random.default_rng(7))
_, affine = km_rollout(Z, tr, te, np.random.default_rng(7), affine=True)
g = np.random.default_rng(2)
res["rollout"] = dict(
    km_per_step=km,
    affine_final=affine,
    ar1_final=ar1_control(N, tr, te, g, 0.0),
    ar1_translating_final=ar1_control(N, tr, te, g, 5.0),
    **identity_rollout(Z, tr, te, np.random.default_rng(7)),
)
print("rollout:", res["rollout"], flush=True)
save("ladder", res)
