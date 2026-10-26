"""The replication on the Pile (Sections 4 and 5): the contests at 16 components, the width ladder and the
density-gradient test with the translation removed, on the Pile ladder with its own camera."""

import numpy as np

from depthflow import data, save
from depthflow.instruments import (
    affine_features,
    alignment_gap,
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
    score_dir,
)

S = load_ladder(data("ladder_160m_pile.npz"))
mu, Vt, tr, te, _ = camera(S)
res = {"n_trajectories": int(S.shape[1]), "transitions": [], "widths": {}}
Z = project(S, mu, Vt, 16)
for l in range(12):
    Ztr, Ztr1, Zte, Zte1 = Z[l][tr], Z[l + 1][tr], Z[l][te], Z[l + 1][te]
    dtr, dte, step = Ztr1 - Ztr, Zte1 - Zte, (Ztr1 - Ztr).mean(0)
    nb = dtr[knn(Ztr, Zte, 64)]
    Wa, Wq = fit(affine_features, Ztr, Ztr1), fit(quad_features, Ztr, Ztr1)
    pa, Rtr = affine_features(Zte) @ Wa, Ztr1 - affine_features(Ztr) @ Wa
    res["transitions"].append(
        dict(
            r2_km=round(r2(Zte + nb.mean(1), Zte, Zte1, step), 3),
            r2_affine=round(r2(pa, Zte, Zte1, step), 3),
            r2_quad=round(r2(quad_features(Zte) @ Wq, Zte, Zte1, step), 3),
            ll_km=round(float(km_ll(nb, dte).mean()), 2),
            ll_null_a=round(float(gauss_diag_ll(dtr, dte).mean()), 2),
            ll_null_c=round(float(gauss_full_ll(dtr, dte).mean()), 2),
            ll_affine=round(float(gauss_diag_ll(Rtr, Zte1 - pa).mean()), 2),
            margin=round(float((km_ll(nb, dte, full=True) - gauss_full_ll(Rtr, Zte1 - pa)).mean()), 2),
        )
    )
print("quad beats affine", sum(t["r2_quad"] > t["r2_affine"] for t in res["transitions"]), "/12", flush=True)

for D in (8, 16, 32, 64):
    ZD = project(S, mu, Vt, D)
    ra, rq = contest(ZD[:-1], ZD[1:], tr, te)
    rk = [
        r2(
            ZD[l][te] + (ZD[l + 1][tr] - ZD[l][tr])[knn(ZD[l][tr], ZD[l][te], 64)].mean(1),
            ZD[l][te],
            ZD[l + 1][te],
            (ZD[l + 1][tr] - ZD[l][tr]).mean(0),
        )
        for l in range(12)
    ]
    res["widths"][D] = dict(quad_wins=int((rq > ra).sum()), km_wins=int((np.array(rk) > ra).sum()))
    print(f"[width {D}]", res["widths"][D], flush=True)

res["gradient"] = []
for l in range(12):
    inc = Z[l + 1] - Z[l]
    g = score_dir(Z[l], exclude_self=False)
    res["gradient"].append(
        dict(
            raw_gap=round(alignment_gap(g, inc, np.random.default_rng(0)), 4),
            centred_gap=round(alignment_gap(g, inc - inc.mean(0), np.random.default_rng(0)), 4),
        )
    )
print("centred gaps:", [t["centred_gap"] for t in res["gradient"]], flush=True)
save("pile", res)
