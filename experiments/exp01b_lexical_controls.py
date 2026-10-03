"""
Experiment 01b - Lexical controls for the matrix-pull finding (no model needed).

Reads the word-level probe predictions saved by exp01 and asks whether the
matrix-language pull survives two lexical confounds:

  1. Shared / function words. Many LinCE "switches" are function words
     (من، في، على، و) that annotators label MSA in one tweet and EGY in another.
     -> split every word type by its label purity over ALL of LinCE
        (train + validation): "unambiguous" types (>= 95% one label, >= 3 uses)
        vs "shared" types.
  2. Word identity. An embedded word may simply be a weak cue of its variety.
     -> TYPE-MATCHED CONTEXT EFFECT: for each embedded word, subtract the mean
        probe score of the SAME word type in monolingual sentences of its own
        variety. What remains is the effect of the code-switched context alone.
        Reported per layer with 95% sentence-bootstrap CIs.

Outputs (results/exp01/<model>/):
  F_context_effect.csv/.png      type-matched context effect, MSA vs EGY embedded
  G_embedded_by_type_class.csv/.png   embedded words: unambiguous vs shared types
  H_switch_unambiguous.csv/.png  switch dynamics, only switches whose first
                                 post-switch word is an unambiguous type

Usage:
    python exp01b_lexical_controls.py --model Qwen/Qwen3-8B-Base
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
LINCE_DIR = PROJECT_DIR / "data" / "raw" / "lince_msaea" / "lid_msaea"
OFFSETS = list(range(-3, 4))

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"


def type_purity():
    cnt = defaultdict(Counter)
    for split in ["train", "validation"]:
        df = pd.read_parquet(LINCE_DIR / f"{split}.parquet")
        for ws, ls in zip(df.words, df.lid):
            for w, l in zip(ws, ls):
                if l in ("lang1", "lang2"):
                    cnt[w][l] += 1
    rows = [{"word": w, "n": c["lang1"] + c["lang2"], "share_egy": c["lang2"] / (c["lang1"] + c["lang2"])}
            for w, c in cnt.items()]
    return pd.DataFrame(rows).set_index("word")


def boot_ci(values, groups, n_boot=2000, seed=0):
    codes = pd.factorize(groups)[0]
    sums = np.bincount(codes, weights=values)
    cnts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnts), (n_boot, len(cnts)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--purity", type=float, default=0.95)
    ap.add_argument("--min_count", type=int, default=3)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / "exp01" / slug
    df = pd.read_parquet(out / "word_predictions.parquet")
    layers = sorted(int(c.split("L")[-1]) for c in df.columns if c.startswith("p_egy_L"))

    pur = type_purity()
    df = df.join(pur, on="word")
    df["type_class"] = "shared"
    frequent = df.n >= args.min_count
    df.loc[frequent & (df.share_egy <= 1 - args.purity), "type_class"] = "unamb_MSA"
    df.loc[frequent & (df.share_egy >= args.purity), "type_class"] = "unamb_EGY"

    def p_own(d, L):
        p = d[f"p_egy_L{L}"].values
        return np.where(d.y.values == 1, p, 1 - p)

    # matrix / embedded roles
    cs = df[df.split == "cs"].copy()
    c = cs.groupby("sid").y.agg(["sum", "count"])
    c["matrix"] = np.where(c["sum"] * 2 > c["count"], 1, np.where(c["sum"] * 2 < c["count"], 0, -1))
    cs["matrix"] = cs.sid.map(c["matrix"])
    cs = cs[cs.matrix >= 0]
    cs["role"] = np.where(cs.y == cs.matrix, "matrix", "embedded")
    emb = cs[cs.role == "embedded"]
    mono = df[df.split != "cs"]

    # ---------------- F. type-matched context effect ----------------
    f_rows = []
    for v, name in [(0, "MSA embedded in EGY"), (1, "EGY embedded in MSA")]:
        ref_all = mono[mono.y == v]
        for L in layers:
            ref = ref_all.assign(po=p_own(ref_all, L)).groupby("word").po.mean()
            for tclass in ["all", "unambiguous"]:
                d = emb[(emb.y == v) & emb.word.isin(ref.index)]
                if tclass == "unambiguous":
                    d = d[d.type_class.str.startswith("unamb")]
                if len(d) < 20:
                    continue
                diff = p_own(d, L) - d.word.map(ref).values
                m, lo, hi = boot_ci(diff, d.sid.values)
                f_rows.append({"condition": name, "types": tclass, "layer": L, "n": len(d),
                               "effect": m, "ci_lo": lo, "ci_hi": hi})
    f_df = pd.DataFrame(f_rows)
    f_df.to_csv(out / "F_context_effect.csv", index=False)

    # ---------------- G. embedded words by type class ----------------
    g_rows = []
    for v, var in [(0, "MSA"), (1, "EGY")]:
        for tclass in ["unambiguous", "shared"]:
            d = emb[emb.y == v]
            d = d[d.type_class.str.startswith("unamb")] if tclass == "unambiguous" else d[d.type_class == "shared"]
            if len(d) < 20:
                continue
            for L in layers:
                m, lo, hi = boot_ci(p_own(d, L), d.sid.values)
                g_rows.append({"variety": var, "types": tclass, "layer": L, "n": len(d),
                               "p_own": m, "ci_lo": lo, "ci_hi": hi})
    g_df = pd.DataFrame(g_rows)
    g_df.to_csv(out / "G_embedded_by_type_class.csv", index=False)

    # ---------------- H. switch dynamics with unambiguous post-switch word ----------------
    acc = defaultdict(list)
    for sid, g in df[df.split == "cs"].groupby("sid"):
        g = g.sort_values("lang_pos")
        ys, idx, lp, tc = g.y.values, g.index.values, g.lang_pos.values, g.type_class.values
        for j in range(1, len(g)):
            if ys[j] != ys[j - 1] and lp[j] == lp[j - 1] + 1 and tc[j].startswith("unamb"):
                d = "MSA->EGY" if ys[j] == 1 else "EGY->MSA"
                for off in OFFSETS:
                    k = j + off
                    if 0 <= k < len(g):
                        acc[(d, off)].append(idx[k])
    h_rows = [{"direction": d, "offset": off, "layer": L, "n": len(ids),
               "p_egy": float(df.loc[ids, f"p_egy_L{L}"].mean())}
              for (d, off), ids in acc.items() for L in layers]
    h_df = pd.DataFrame(h_rows)
    h_df.to_csv(out / "H_switch_unambiguous.csv", index=False)

    make_figures(f_df, g_df, h_df, out, slug)

    # ---------------- console summary ----------------
    print(f"\nType classes among embedded words:\n{emb.groupby(['y', 'type_class']).size()}")
    last = layers[-1]
    print(f"\nType-matched context effect at layers 0 / 4 / {last}:")
    for (cond, t), d in f_df.groupby(["condition", "types"]):
        s = d.set_index("layer")
        print(f"  {cond:<22} [{t:<11}] n={int(s.n.iloc[0]):>4}  " + "  ".join(
            f"L{L}={s.loc[L, 'effect']:+.3f} [{s.loc[L, 'ci_lo']:+.3f},{s.loc[L, 'ci_hi']:+.3f}]"
            for L in [0, 4, last] if L in s.index))
    print(f"\nEmbedded words, P(own variety) at layer {last}:")
    for (v, t), d in g_df.groupby(["variety", "types"]):
        r = d[d.layer == last].iloc[0]
        print(f"  {v} {t:<11} n={int(r.n):>4}  {r.p_own:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]")
    for d in ["MSA->EGY", "EGY->MSA"]:
        if (d, 0) in acc:
            s = h_df[(h_df.direction == d) & (h_df.offset == 0)].set_index("layer")
            print(f"\n{d} (unambiguous first word, n={len(acc[(d, 0)])}): P(EGY) at offset 0: "
                  + "  ".join(f"L{L}={s.loc[L, 'p_egy']:.3f}" for L in [0, 4, 12, last] if L in s.index))
    print(f"\nResults in {out}")


def make_figures(f_df, g_df, h_df, out, slug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    tkw = dict(color=C_TEXT, fontsize=11, loc="left")

    # F
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for cond, col in [("MSA embedded in EGY", C_BLUE), ("EGY embedded in MSA", C_ORANGE)]:
        for t, ls in [("all", "-"), ("unambiguous", "--")]:
            d = f_df[(f_df.condition == cond) & (f_df.types == t)]
            if not len(d):
                continue
            if t == "all":
                ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
            ax.plot(d.layer, d.effect, color=col, lw=2, ls=ls,
                    label=f"{cond} — {'all types' if t == 'all' else 'unambiguous types'} (n={int(d.n.iloc[0])})")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Δ P(own variety) vs. same word\nin monolingual context")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title(f"F. Type-matched context effect of code-switching — {slug}\n(bands = 95% sentence-bootstrap CI)", **tkw)
    fig.tight_layout()
    fig.savefig(out / "F_context_effect.png", dpi=200)
    plt.close(fig)

    # G
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, col in [("MSA", C_BLUE), ("EGY", C_ORANGE)]:
        for t, ls in [("unambiguous", "-"), ("shared", ":")]:
            d = g_df[(g_df.variety == v) & (g_df.types == t)]
            if not len(d):
                continue
            ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.12, lw=0)
            ax.plot(d.layer, d.p_own, color=col, lw=2, ls=ls, label=f"Embedded {v} — {t} types (n={int(d.n.iloc[0])})")
    ax.axhline(0.5, color=C_MUTED, lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Mean P(word's own variety)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title(f"G. Embedded words by lexical ambiguity — {slug}", **tkw)
    fig.tight_layout()
    fig.savefig(out / "G_embedded_by_type_class.png", dpi=200)
    plt.close(fig)

    # H
    if len(h_df):
        cmap = LinearSegmentedColormap.from_list("msa_egy", [C_BLUE, "#e8e8e6", C_ORANGE])
        fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
        im = None
        for ax, d in zip(axes, ["MSA->EGY", "EGY->MSA"]):
            sub = h_df[h_df.direction == d]
            if not len(sub):
                continue
            piv = sub.pivot(index="layer", columns="offset", values="p_egy")
            im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
                           extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5, -0.5, piv.shape[0] - 0.5])
            ax.axvline(-0.5, color=C_TEXT, lw=1.2)
            ax.set_xticks(list(piv.columns))
            ax.set_xlabel("Word offset (0 = first word after switch)")
            n = int(sub[sub.offset == 0].n.iloc[0])
            ax.set_title(f"{d}  (n={n})", color=C_TEXT, fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_ylabel("Layer")
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.8).set_label("Probe P(EGY)")
        fig.suptitle(f"H. Switch dynamics, unambiguous post-switch word only — {slug}",
                     x=0.02, ha="left", color=C_TEXT, fontsize=11)
        fig.savefig(out / "H_switch_unambiguous.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
