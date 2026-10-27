"""The paper's instruments: the balanced principal basis, the drift fits, the local estimator, the likelihoods, the
score direction and the cubic potential. Random draws are made in the order of the original scripts so that every number
of the paper is reproduced to the last digit."""

import itertools

import numpy as np
from scipy.spatial import cKDTree


def load_ladder(path, mincount=5):
    z = np.load(path)
    S = z["S"].astype(np.float32)
    return S[:, np.where(z["cnt"] >= mincount)[0]]


def basis(S, seed=0):
    """Balanced principal basis (one vote per layer), pooled mean, 80/20 token split, and the generator afterwards."""
    rng = np.random.default_rng(seed)
    N = S.shape[1]
    samp = rng.choice(N, min(8000, N), replace=False)
    mu = S[:, samp].reshape(-1, S.shape[2]).mean(0)
    pool = []
    for l in range(S.shape[0]):
        A = S[l][samp] - S[l][samp].mean(0)
        pool.append(A / max(np.sqrt((A**2).sum(1).mean()), 1e-9))
    Xb = np.concatenate(pool, 0)
    Vt = np.linalg.svd(Xb[rng.choice(len(Xb), 20000, replace=False)], full_matrices=False)[2]
    tr = rng.choice(N, int(0.8 * N), replace=False)
    te = np.setdiff1d(np.arange(N), tr)
    return mu, Vt, tr, te, rng


def project(S, mu, Vt, D):
    return np.einsum("lnd,kd->lnk", S - mu, Vt[:D])


def affine_features(P):
    return np.hstack([P, np.ones((len(P), 1), np.float32)])


def quad_features(P):
    iu = np.triu_indices(P.shape[1])
    return np.hstack([P, P[:, iu[0]] * P[:, iu[1]], np.ones((len(P), 1), np.float32)])


def fit(features, Ztr, Ztr1):
    return np.linalg.lstsq(features(Ztr), Ztr1, rcond=None)[0]


def r2(pred, Zte, Zte1, mean_step):
    """Held-out R2 of the predicted next state, against the training mean step."""
    return float(1 - ((Zte1 - pred) ** 2).sum() / ((Zte1 - Zte - mean_step) ** 2).sum())


def contest(Z, Z1, tr, te):
    """Per-transition held-out R2 of the affine and the quadratic map. Z1[l] is the state after Z[l]."""
    out = {"affine": [], "quad": []}
    for l in range(len(Z)):
        step = (Z1[l][tr] - Z[l][tr]).mean(0)
        for name, f in (("affine", affine_features), ("quad", quad_features)):
            out[name].append(r2(f(Z[l][te]) @ fit(f, Z[l][tr], Z1[l][tr]), Z[l][te], Z1[l][te], step))
    return np.array(out["affine"]), np.array(out["quad"])


def knn(train, query, k, chunk=2000):
    """Indices of the k nearest training points, nearest first. Tree up to 32 dimensions, exact search above."""
    if train.shape[1] <= 32:
        return cKDTree(train).query(query, k=k)[1]
    sq = (train**2).sum(1)
    out = np.empty((len(query), k), np.int64)
    for a in range(0, len(query), chunk):
        d2 = sq[None] - 2 * query[a : a + chunk] @ train.T
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        out[a : a + chunk] = np.take_along_axis(part, np.argsort(np.take_along_axis(d2, part, 1), axis=1), 1)
    return out


def km_drift(Ztr, dtr, Zte, k=64):
    return dtr[knn(Ztr, Zte, k)].mean(1)


def gauss_diag_ll(Rtr, Rte):
    """Per-point log-likelihood of Rte under the diagonal Gaussian fitted on Rtr."""
    m, v = Rtr.mean(0), Rtr.var(0) + 1e-6
    return -0.5 * (np.log(2 * np.pi * v) + (Rte - m) ** 2 / v).sum(1)


def gauss_full_ll(Rtr, Rte):
    m = Rtr.mean(0)
    L = np.linalg.cholesky(np.cov((Rtr - m).T) + 1e-6 * np.eye(Rtr.shape[1]))
    r = np.linalg.solve(L, (Rte - m).T)
    return -0.5 * (r**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * Rtr.shape[1] * np.log(2 * np.pi)


def km_ll(nb, dte, full=False, shrink=0.1):
    """Per-point log-likelihood of the step dte under the local Gaussian of its k neighbours' steps nb."""
    m = nb.mean(1)
    if not full:
        v = nb.var(1) + 1e-6
        return -0.5 * (np.log(2 * np.pi * v) + (dte - m) ** 2 / v).sum(1)
    E = nb - m[:, None, :]
    C = np.einsum("nki,nkj->nij", E, E) / (nb.shape[1] - 1)
    d = np.einsum("nii->ni", C).copy()
    C *= 1 - shrink
    i = np.arange(C.shape[1])
    C[:, i, i] = d + 1e-6
    L = np.linalg.cholesky(C)
    r = np.linalg.solve(L, (dte - m)[:, :, None])[:, :, 0]
    return -0.5 * (r**2).sum(1) - np.log(np.einsum("nii->ni", L)).sum(1) - 0.5 * C.shape[1] * np.log(2 * np.pi)


def score_dir(pts, k=48, exclude_self=True):
    """Unit kNN mean-shift direction, the estimate of the direction of grad log p."""
    idx = knn(pts, pts, k + 1)[:, 1:] if exclude_self else knn(pts, pts, k)
    g = pts[idx].mean(1) - pts
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def alignment_gap(g, inc, rng, n=4000):
    """Mean cosine between increments and score directions, minus the same with shuffled pairing."""
    dn = inc / np.maximum(np.linalg.norm(inc, axis=1, keepdims=True), 1e-9)
    ref = np.sort(rng.choice(len(g), n, replace=False))
    sh = rng.permutation(len(ref))
    return float((dn[ref] * g[ref]).sum(1).mean() - (dn[ref] * g[ref][sh]).sum(1).mean())


class CubicPotential:
    """Gradient fields of cubic polynomial potentials in D dimensions."""

    def __init__(self, D):
        self.D = D
        monos = []
        for deg in (1, 2, 3):
            for c in itertools.combinations_with_replacement(range(D), deg):
                e = np.zeros(D, np.int8)
                for i in c:
                    e[i] += 1
                monos.append(e)
        self.E = np.array(monos)
        self.M = len(self.E)

    def design(self, Zb):
        G = np.zeros((len(Zb), self.D, self.M), np.float32)
        cache = {}
        for m, e in enumerate(self.E):
            for i in np.where(e > 0)[0]:
                e2 = e.copy()
                e2[i] -= 1
                key = tuple(e2)
                if key not in cache:
                    cache[key] = np.prod(Zb**e2, axis=1)
                G[:, i, m] = e[i] * cache[key]
        return G

    def r2(self, Ztr, dtr, Zte, dte, ridge=1e-6, relative=False):
        """Held-out R2 of the best gradient field. The ridge is absolute, or relative to the mean diagonal."""
        AtA = np.zeros((self.M, self.M), np.float64)
        Aty = np.zeros(self.M, np.float64)
        for a in range(0, len(Ztr), 2000):
            Gf = self.design(Ztr[a : a + 2000]).reshape(-1, self.M)
            AtA += Gf.T @ Gf
            Aty += Gf.T @ dtr[a : a + 2000].reshape(-1)
        reg = ridge * np.trace(AtA) / self.M if relative else ridge
        c = np.linalg.solve(AtA + reg * np.eye(self.M), Aty)
        sse = sum(
            float(((dte[a : a + 2000] - np.einsum("ndm,m->nd", self.design(Zte[a : a + 2000]), c)) ** 2).sum())
            for a in range(0, len(Zte), 2000)
        )
        return 1 - sse / float(((dte - dte.mean(0)) ** 2).sum())


def quad_field(P):
    D = P.shape[1]
    return np.column_stack([np.ones(len(P)), P] + [P[:, i : i + 1] * P[:, i:] for i in range(D)])


def circulation(Z, Z1, tr, te, pot, ridge=1e-6, relative=False, centre=True):
    """R2 of the quadratic field, of the gradient of a cubic potential, and 1 minus their ratio, on increments
    centred by the training mean step."""
    d = (Z1 - Z) - ((Z1 - Z)[tr].mean(0) if centre else 0)
    W = fit(quad_field, Z[tr], d[tr])
    r2f = float(1 - (d[te] - quad_field(Z[te]) @ W).var() / d[te].var())
    r2g = pot.r2(Z[tr], d[tr], Z[te], d[te], ridge, relative)
    return r2f, r2g, (1 - r2g / r2f if r2f > 0.01 else float("nan"))
