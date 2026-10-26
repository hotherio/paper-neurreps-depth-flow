"""The tables of the appendix and the numbers quoted in the text, written from results/ into paper/tables/.
Table 1 and Table 6 are typed in the paper from results/ladder.json."""

import os

import numpy as np

from depthflow import ROOT, load

OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
macros = {}


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def table(name, caption, label, header, rows, size="small", columns=None):
    columns = columns or "l" + "c" * header.count("&")
    open(f"{OUT}/{name}.tex", "w").write(
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\{size}\n"
        f"\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
        + header
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


P = load("projection")
rows = []
for arm, label in (
    ("real", "real flow"),
    ("affine", "affine at full width"),
    ("crossfit", "affine, cross-fitted"),
    ("gaussian", "affine, Gaussian cloud"),
    ("symmetric", "gradient at full width"),
):
    cells = " & ".join(
        f"{P[arm][w]['quad_wins']}/12 & {np.median(P[arm][w]['margin']):+.3f}" for w in ("16", "32", "64")
    )
    rot = P[arm].get("rotational")
    rows.append(f"{label} & {cells} & " + (f"{min(rot):.2f}--{max(rot):.2f}" if rot else "--") + " \\\\")
macros["crProjRatioSixteen"] = pct(sum(P["affine"]["16"]["margin"]) / sum(P["real"]["16"]["margin"]))
macros["crFullAffMin"], macros["crFullAffMax"] = (f"{f(P['fullwidth_affine_r2']):.2f}" for f in (min, max))
for name, arm in (("Aff", "affine"), ("Grad", "symmetric")):
    macros[f"crProj{name}CircMin"], macros[f"crProj{name}CircMax"] = (pct(f(P[arm]["rotational"])) for f in (min, max))
table(
    "projection",
    "Quadratic against affine map on the real flow and on four surrogates whose full-width step is "
    "known. Wins are transitions where the quadratic map explains more held-out increment variance, margins are "
    "medians over transitions, and the last column is the range of the circulation share at $16$ components. "
    "Cross-fitted: every training token predicted by a map fitted on the other half.",
    "tab:projection",
    "& \\multicolumn{2}{c}{16 components} & \\multicolumn{2}{c}{32} & \\multicolumn{2}{c}{64} & circulation (16) \\\\\n"
    "flow & wins & margin & wins & margin & wins & margin & \\\\",
    rows,
    columns="lcccccc c",
)


def fullwidth_rows(states, widths):
    rows = []
    for w in widths:
        r = load(f"fullwidth_{states}_{w}")
        gain = lambda arm, r=r: np.array([x[arm]["mlp"] - x[arm]["affine"] for x in r["per_transition"]])
        aff, share = np.array(r["affine_r2_real"]), np.array(r["nonaffine_share"])
        rows.append(
            f"{'raw $768$' if w == 768 else w} & {aff.min():.2f}--{aff.max():.2f} & "
            f"{gain('null').min():+.3f} to {gain('null').max():+.3f} & {gain('real').min():+.3f} to "
            f"{gain('real').max():+.3f} & {np.median(gain('real') - gain('null')):+.3f} & "
            f"{np.median(r['recovery']):.2f} & {np.median(share) * 100:.1f}\\% \\\\"
        )
        if w == 768:
            key = "crFw" if states == "means" else "crOccFw"
            macros[key + "GainMin"], macros[key + "GainMax"] = (
                f"{gain('real').min():+.3f}",
                f"{gain('real').max():+.3f}",
            )
            nd = 1 if states == "means" else 0
            macros[key + "ShareMin"], macros[key + "ShareMax"] = pct(share.min(), nd), pct(share.max(), nd)
            macros[key + "ShareMed"] = pct(np.median(share), nd)
            if states == "means":
                macros["crFwShareMedRound"] = pct(np.median(share))
            else:
                macros["crOccFullAffMin"], macros["crOccFullAffMax"] = f"{aff.min():.2f}", f"{aff.max():.2f}"
                macros["crOccFwPositions"] = f"{r['n_states']:,}".replace(",", "{,}")
    return rows


table(
    "fullwidth",
    "The non-affine part of the step per state width, on the mean trajectories and on $64{,}000$ "
    "individual states. Affine $R^2$ is held out; gains are what the network adds to the affine map on the null and "
    "on the real flow; recovery is the fraction of a planted $10\\%$ non-affine term the network finds; the last "
    "column is the real gain over the null read through the recovery. Ranges and medians over the twelve "
    "transitions of Pythia-160M.",
    "tab:fullwidth",
    "width & affine $R^2$ & gain on null & gain on real & real $-$ null & recovery & non-affine \\\\",
    ["\\multicolumn{7}{l}{mean trajectories} \\\\"]
    + fullwidth_rows("means", (768, 256, 64, 16))
    + ["\\midrule", "\\multicolumn{7}{l}{individual states} \\\\"]
    + fullwidth_rows("positions", (768, 64, 16)),
    size="footnotesize",
)

D = load("density")
rows = []
for w in ("16", "32", "64", "128"):
    pos = np.array([t["drift"]["pos"] for t in D["sign_test"][w]])
    nul = np.array([t["drift"]["pos_null"] for t in D["sign_test"][w]])
    posc = np.array([t["drift_centred"]["pos"] for t in D["sign_test"][w]])
    gD = np.array([t["drift"]["cos_gap_D"] for t in D["sign_test"][w]])
    rows.append(
        f"{w} & {min(c['drift']['pos'] for c in D['sign_controls'][w].values()):.2f} & {pos.mean():.3f} & "
        f"{nul.mean():.3f} & {pos.min():.2f}--{pos.max():.2f} & {posc.min():.2f}--{posc.max():.2f} & "
        f"{gD.mean():+.3f} / {gD.max():+.3f} \\\\"
    )
    if w == "16":
        macros["crPosReal"], macros["crPosNull"] = pct(pos.mean()), pct(nul.mean())
macros["crGradCeiling"] = f"{D['ou_control']['gap']:.2f}"
table(
    "density",
    "The sign test at four widths: the fraction of held-out tokens whose drift has a positive inner "
    "product with the estimated score. Controls is the smallest value over the six controls of "
    "Appendix~\\ref{app:density}, real and shuffled are means over transitions, then the per-transition range, the "
    "same with the common translation removed, and the cosine gap once the score is multiplied by the fitted "
    "residual covariance.",
    "tab:density",
    "width & controls & real & shuffled & per transition & translation removed & cos.\\ gap with $\\hat D$ \\\\",
    rows,
)

R = load("preserve")
for s, name in (("wikitext_160m", "WikiSmall"), ("pile_160m", "PileSmall"), ("wikitext_410m", "WikiLarge")):
    macros["crPres" + name] = f"{R[s]['cos_mean']['icc']:.2f}"
dens = [R[s]["density"]["icc"] for s in ("wikitext_160m", "pile_160m", "wikitext_410m")]
macros["crDensIccMin"], macros["crDensIccMax"] = f"{min(dens):.2f}", f"{max(dens):.2f}"
macros["crNormFourTen"] = f"{R['wikitext_410m']['norm']['spearman_entry_exit']:+.2f}"
macros["crPartFourTen"] = f"{R['wikitext_410m']['participation']['spearman_entry_exit']:+.2f}"

occ, pile, L, pot = load("positions"), load("pile"), load("ladder"), load("potential")
macros["crOccPositions"] = f"{occ['n_positions']:,}".replace(",", "{,}")
macros["crOccQuadWins"] = str(sum(t["r2_quad"] > t["r2_affine"] for t in occ["transitions"]))
macros["crOccGradMean"] = f"{np.mean([t['centred_gap'] for t in occ['gradient']]):.2f}"
macros["crPileQuadWinsMin"] = str(min(w["quad_wins"] for w in pile["widths"].values()))
macros["crPileGradMean"] = f"{np.mean([t['centred_gap'] for t in pile['gradient']]):+.3f}"
macros["crPileGradMax"] = f"{max(t['centred_gap'] for t in pile['gradient']):+.3f}"
roll = L["rollout"]
macros["crIdPathMin"], macros["crIdPathMax"] = (
    f"{min(roll['identity_path'][:11]):.2f}",
    f"{max(roll['identity_path'][:11]):.2f}",
)
macros["crIdPathFused"] = f"{roll['identity_path'][11]:.2f}"
macros["crIidMin"], macros["crIidMax"] = f"{min(roll['iid'][:11]):.2f}", f"{max(roll['iid'][:11]):.2f}"
macros["crRidgeExitLoose"] = pct(pot["ridge"]["0.0001"][11]["rotational"])
macros["crHelmMin"] = pct(min(t["rotational"] for t in pot["transitions"]))

open(f"{OUT}/numbers.tex", "w").write(
    "% generated by tables.py from results/\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
)
print("\n".join(f"{k} = {v}" for k, v in sorted(macros.items())))
