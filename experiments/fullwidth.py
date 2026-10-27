"""The full-width contest (Section 4, Table 5): how much of the step is not affine, on the mean trajectories or on
64,000 individual states, at the raw width or in the principal basis at a given width.
Usage: uv run python experiments/fullwidth.py means 768       uv run python experiments/fullwidth.py positions 768

Per transition, four next-state arms share the real current states. The real arm is the real next states. The null
is the full-width affine prediction plus another state's affine residual, which is affine by construction. The two
planted arms add to the null a non-affine term, with every affine function of the state regressed out, at 5% and
10% of the step variance. Three models are scored by held-out R2 of the step: a ridge affine map, the affine map
plus a ridge on 2048 random GELU features of the residual, and the affine map plus a 1024-unit GELU network on the
residual. The ridge strengths and the network's stopping point come from a tenth of the training states. The
non-affine share of the real flow is its gain over the null, divided by the recovered fraction of the planted term."""

import sys

import numpy as np
import torch

from depthflow import data, save
from depthflow.instruments import affine_features, basis, load_ladder

STATES, WIDTH = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(4)
LAMBDAS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_RF, RANK = 2048, 32
SHARES = {"planted5": 0.05, "planted10": 0.10}


def ridge(F, Y, fit, val):
    Ff, Yf = F[fit].astype(np.float64), Y[fit].astype(np.float64)
    e, V = np.linalg.eigh(Ff.T @ Ff / len(Ff))
    B = V.T @ (Ff.T @ Yf / len(Ff))
    Ws = [(V @ (B / (e[:, None] + lam))).astype(np.float32) for lam in LAMBDAS]
    return Ws[int(np.argmin([((Y[val] - F[val] @ W) ** 2).sum() for W in Ws]))]


def network(X, R, fit, val, seed):
    torch.manual_seed(seed)
    scale = float(R[fit].std()) + 1e-8
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, R.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    x, r, fit_t = torch.from_numpy(X), torch.from_numpy(R / scale), torch.from_numpy(fit)
    best, state, bad = np.inf, None, 0
    for _ in range(300):
        for idx in fit_t[torch.randperm(len(fit_t))].split(512):
            opt.zero_grad()
            ((net(x[idx]) - r[idx]) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            v = float(((net(x[val]) - r[val]) ** 2).mean())
        if v < best * (1 - 1e-4):
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad == 15:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(x).numpy() * scale


def score(X, X1, tr, te, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    val, fit = perm[: len(tr) // 10], perm[len(tr) // 10 :]
    Xs = ((X - X[fit].mean(0)) / (X[fit].std(0) + 1e-8)).astype(np.float32)
    d = (X1 - X).astype(np.float32)
    d = d - d[fit].mean(0)
    Fa = affine_features(Xs)
    Wa = ridge(Fa, d, fit, val)
    resid = d - Fa @ Wa
    H = torch.nn.functional.gelu(
        torch.from_numpy(
            Xs @ (rng.standard_normal((X.shape[1], N_RF)).astype(np.float32) / np.sqrt(X.shape[1]))
            + rng.uniform(-1, 1, N_RF).astype(np.float32)
        )
    ).numpy()
    H = (H - H[fit].mean(0)) / (H[fit].std(0) + 1e-8)
    r2 = lambda pred: round(1 - float(((d[te] - pred[te]) ** 2).sum()) / float((d[te] ** 2).sum()), 4)
    return dict(
        affine=r2(Fa @ Wa),
        rf=r2(Fa @ Wa + H @ ridge(H, resid, fit, val)),
        mlp=r2(Fa @ Wa + network(Xs, resid, fit, val, seed)),
    )


rng = np.random.default_rng(11)
mu, Vt, tr, te, _ = basis(load_ladder(data("ladder_160m_wikitext.npz")))
if STATES == "means":
    S = load_ladder(data("ladder_160m_wikitext.npz")).astype(np.float64)
else:
    z = np.load(data("positions_160m_wikitext.npz"))
    pos = np.sort(rng.choice(z["doc"].shape[0], 64_000, replace=False))
    doc, S = z["doc"][pos], z["H"][:, pos].astype(np.float64)
    test_docs = set(rng.choice(np.unique(doc), len(np.unique(doc)) // 5, replace=False).tolist())
    te = np.array([i for i, d_ in enumerate(doc) if d_ in test_docs])
    tr = np.array([i for i, d_ in enumerate(doc) if d_ not in test_docs])
N = S.shape[1]
basis = np.linalg.qr(rng.standard_normal((64, 64)))[0] @ Vt[:64].astype(np.float64)
U, V, Wd = basis[:RANK], basis[RANK : 2 * RANK], np.linalg.qr(rng.standard_normal((64, RANK)))[0].T @ Vt[:64]
P = np.eye(768) if WIDTH == 768 else Vt[:WIDTH].astype(np.float64).T
shift = 0 if WIDTH == 768 else mu

rows = []
for l in range(12):
    X, X1 = S[l], S[l + 1]
    m0 = X[tr].mean(0)
    F = affine_features(X - m0)
    pred = F @ np.linalg.lstsq(F[tr], X1[tr], rcond=None)[0]
    donor = np.empty(N, np.int64)
    donor[tr], donor[te] = rng.permutation(tr), rng.permutation(te)
    null = pred + (X1 - pred)[donor]
    a, b = (X - m0) @ U.T, (X - m0) @ V.T
    term = (a / a[tr].std(0)) * (b / b[tr].std(0)) @ Wd
    term -= F @ np.linalg.lstsq(F[tr], term[tr], rcond=None)[0]
    step = X1 - X - (X1 - X)[tr].mean(0)
    arms, planted_share = {"real": X1, "null": null}, {}
    for name, share in SHARES.items():
        g = np.sqrt(share * (step[tr] ** 2).sum() / (term[tr] ** 2).sum())
        arms[name] = null + g * term
        planted_share[name] = round(float(((g * term[te] @ P) ** 2).sum() / ((step[te] @ P) ** 2).sum()), 4)
    Xin = ((X - shift) @ P).astype(np.float32)
    row = dict(l=l, planted_share=planted_share)
    for name, Y in arms.items():
        row[name] = score(Xin, ((Y - shift) @ P).astype(np.float32), tr, te, l)
    rows.append(row)
    print(
        f"[{STATES} {WIDTH} L{l}->L{l + 1}] "
        + " | ".join(
            f"{k} affine {row[k]['affine']:.3f} network {row[k]['mlp'] - row[k]['affine']:+.3f}" for k in arms
        ),
        flush=True,
    )

gain = lambda arm: np.array([r[arm]["mlp"] - r[arm]["affine"] for r in rows])
recovery = (gain("planted10") - gain("null")) / np.array([r["planted_share"]["planted10"] for r in rows])
share = (gain("real") - gain("null")) / recovery
print(f"non-affine share: median {np.median(share):.3f}, range {share.min():.3f}-{share.max():.3f}", flush=True)
save(
    f"fullwidth_{STATES}_{WIDTH}",
    dict(
        per_transition=rows,
        n_states=int(N),
        n_train=int(len(tr)),
        n_test=int(len(te)),
        affine_r2_real=[r["real"]["affine"] for r in rows],
        recovery=recovery.tolist(),
        nonaffine_share=share.tolist(),
    ),
)
