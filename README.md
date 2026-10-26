# The Depth Flow of Token Representations Is Nonlinear and Does Not Descend Its Own Density

Alexandre Quemy. NeurReps 2026, Proceedings track (PMLR).

`paper/` contains the camera-ready source while the rest of the repository contains the experiments that reproduce its numbers, tables and figures.

## Layout

```
depthflow/instruments.py   the camera, the drift fits, the local estimator, the likelihoods, the score direction,
                           the cubic potential
depthflow/data.py          the hidden states from Hugging Face, in the three forms the paper uses
experiments/               one script per result, each writes results/<name>.json
figures/                   the three figures, written to paper/figures/
tables.py                  the appendix tables and the numbers quoted in the text, written to paper/tables/
results/                   the JSON files the paper was built from
paper/                     main.tex, the class files, references.bib, figures/, tables/
```

## Reproduction

```bash
uv sync
uv run python -m depthflow.data ladder 160m wikitext 25000      # Pythia-160M on WikiText-103, the main object
uv run python -m depthflow.data ladder 160m pile 25000          # the Pile replication
uv run python -m depthflow.data ladder 410m wikitext 5000       # the Pythia-410M replication
uv run python -m depthflow.data positions 160m wikitext 2000    # 242,000 individual states at full width
uv run python -m depthflow.data reduced 160m wikitext 25000     # 3.0M individual states in the 16-component camera
```

Data lands in `data/`, about 8 GB. Set `DEPTHFLOW_DATA` to put it elsewhere. The streams run on Apple MPS when it
is available and on the CPU otherwise. Each script then runs with `uv run python`:

| Script | What it reproduces | Needs |
|---|---|---|
| `experiments/ladder.py` | Table 1, Table 6, the width ladder and the neighbourhood sweep, the variance retained by the camera, the Markov test, the rollouts and their controls (Section 4) | the three ladders |
| `experiments/density.py` | the density-gradient test and its calibration, Table 3 (Section 5, Appendix A) | 160M and 410M ladders |
| `experiments/potential.py` | the rotational part, its controls, the ridge sensitivity, the symmetric share (Section 5) | 160M ladder |
| `experiments/projection.py` | the projection control, Table 4 (Section 4) | 160M ladder |
| `experiments/fullwidth.py means 768` | the full-width contest on the mean trajectories, Table 5 (also at 256, 64, 16) | 160M ladder |
| `experiments/fullwidth.py positions 768` | the same on 64,000 individual states, Table 5 (also at 64, 16) | 160M ladder, positions |
| `experiments/preserve.py` | what the flow preserves (Section 6) | the three ladders |
| `experiments/pile.py` | the Pile replication (Sections 4 and 5) | Pile ladder |
| `experiments/positions.py` | the contests on 3.0M un-averaged positions (Sections 4 and 5) | reduced positions |
| `experiments/noise.py` | the estimation noise of a corpus mean by occurrence count (Section 3) | 160M ladder |
| `figures/river.py`, `figures/quadfield.py`, `figures/helm.py` | Figures 1 to 3 | 160M ladder |
| `tables.py` | `paper/tables/*.tex` from `results/` | the results above |

The scripts draw their random numbers in the order of the original runs. On the same caches they reproduce the
committed `results/` and the paper's numbers to the printed digit. The caches themselves depend on the hardware's
float16 arithmetic, and a rebuild from Hugging Face reproduces them to a few decimals.

## Paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
