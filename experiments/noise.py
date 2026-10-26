"""The estimation noise of a corpus mean (Section 3): from the sums over two random halves of each token's
occurrences at the last state, the noise of the full mean as a fraction of the spread between tokens, by
occurrence count."""

import numpy as np

from depthflow import data, save

z = np.load(data("ladder_160m_wikitext.npz"))
cnt, cA = z["cnt"], z["cnt_half"]
cB = cnt - cA
ok = (cA >= 1) & (cB >= 1) & (cnt >= 5)
M = z["S"][-1][ok].astype(np.float32)
mA = z["half"][ok] / cA[ok, None]
mB = (M * cnt[ok, None] - z["half"][ok]) / cB[ok, None]
noise2 = ((mA - mB) ** 2).sum(1) / (1.0 / cA[ok] + 1.0 / cB[ok]) / cnt[ok]
spread2 = float(((M - M.mean(0)) ** 2).sum(1).mean())
res = {"n_tokens": int(ok.sum()), "bins": {}}
for lo, hi in ((5, 10), (10, 100), (100, 1000), (1000, None)):
    m = (cnt[ok] >= lo) & (cnt[ok] < (hi or np.inf))
    res["bins"][f"{lo}-{hi or 'max'}"] = dict(
        n=int(m.sum()), noise_over_spread=round(float(np.median(np.sqrt(noise2[m] / spread2))), 4)
    )
print(res, flush=True)
save("noise", res)
