"""Hidden states of Pythia on a corpus, in the three forms the paper uses.

  ladder     next-token-conditioned corpus means: S[l, v] is the mean state after block l over the positions whose
             next token is v, with the occurrence count; the last state also keeps the sum over a random half of
             the occurrences, for the noise floor of Section 3
  positions  every position of the first documents, all states, in full width (the individual states of Section 4)
  reduced    every position of all documents, projected on the fly into the paper's 16-component camera

Documents are the first n of the corpus longer than 400 characters, truncated at 128 tokens.
Usage: python -m depthflow.data ladder 160m wikitext 25000
       python -m depthflow.data ladder 160m pile 25000
       python -m depthflow.data ladder 410m wikitext 5000
       python -m depthflow.data positions 160m wikitext 2000
       python -m depthflow.data reduced 160m wikitext 25000"""

import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from depthflow import data
from depthflow.instruments import camera, load_ladder

MODELS = {"160m": "EleutherAI/pythia-160m", "410m": "EleutherAI/pythia-410m"}
CORPORA = {"wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"), "pile": ("monology/pile-uncopyrighted",)}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def documents(corpus, n):
    kept = 0
    for t in load_dataset(*CORPORA[corpus], split="train", streaming=True):
        if len(t["text"]) > 400:
            yield t["text"]
            kept += 1
            if kept == n:
                return


def hidden_states(model, corpus, n_docs, dtype=torch.float16):
    """Per document: the states before each next token, (layers + 1, positions, width), and the next-token ids."""
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    mdl = AutoModelForCausalLM.from_pretrained(MODELS[model], dtype=dtype).eval().to(DEVICE)
    with torch.no_grad():
        for i, text in enumerate(documents(corpus, n_docs)):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            if ids.shape[1] < 2:
                continue
            hs = mdl(input_ids=ids.to(DEVICE), output_hidden_states=True).hidden_states
            if i % 2000 == 0:
                print(f"[{model} {corpus}] document {i}/{n_docs}", flush=True)
            yield np.stack([h[0, :-1].float().cpu().numpy() for h in hs]), ids[0, 1:].numpy()


def ladder(model, corpus, n_docs):
    V = AutoModelForCausalLM.from_pretrained(MODELS[model]).config.vocab_size
    rng = np.random.default_rng(0)
    S = cnt = half = cnt_half = None
    for hs, nxt in hidden_states(model, corpus, n_docs):
        if S is None:
            S, half = np.zeros((len(hs), V, hs.shape[2]), np.float32), np.zeros((V, hs.shape[2]), np.float32)
            cnt, cnt_half = np.zeros(V, np.int64), np.zeros(V, np.int64)
        cnt += np.bincount(nxt, minlength=V)
        for l in range(len(hs)):
            np.add.at(S[l], nxt, hs[l])
        h = rng.integers(0, 2, len(nxt)).astype(bool)
        np.add.at(half, nxt[h], hs[-1][h])
        cnt_half += np.bincount(nxt[h], minlength=V)
    S /= np.maximum(cnt, 1)[None, :, None]
    np.savez_compressed(
        data(f"ladder_{model}_{corpus}.npz"), S=S.astype(np.float16), cnt=cnt, half=half, cnt_half=cnt_half
    )


def positions(model, corpus, n_docs, max_positions=260_000):
    H, NX, DC = [], [], []
    for d, (hs, nxt) in enumerate(hidden_states(model, corpus, n_docs, torch.float32)):
        if sum(len(x) for x in NX) + len(nxt) > max_positions:
            continue
        H.append(hs.astype(np.float16))
        NX.append(nxt)
        DC.append(np.full(len(nxt), d, np.int32))
    np.savez(
        data(f"positions_{model}_{corpus}.npz"), H=np.concatenate(H, 1), nxt=np.concatenate(NX), doc=np.concatenate(DC)
    )


def reduced(model, corpus, n_docs, width=16):
    mu, Vt, *_ = camera(load_ladder(data(f"ladder_{model}_{corpus}.npz")))
    P = Vt[:width].T
    Z, NX, DC = [], [], []
    for d, (hs, nxt) in enumerate(hidden_states(model, corpus, n_docs)):
        Z.append(((hs - mu) @ P).astype(np.float16))
        NX.append(nxt)
        DC.append(np.full(len(nxt), d, np.int32))
    np.savez_compressed(
        data(f"reduced_{model}_{corpus}.npz"), Z=np.concatenate(Z, 1), nxt=np.concatenate(NX), doc=np.concatenate(DC)
    )


if __name__ == "__main__":
    kind, model, corpus, n_docs = sys.argv[1:5]
    {"ladder": ladder, "positions": positions, "reduced": reduced}[kind](model, corpus, int(n_docs))
