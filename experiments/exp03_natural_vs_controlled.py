"""
Experiment 03 - Why is the asymmetry strong in natural tweets (exp01) but absent
in controlled minimal pairs (exp02)?

For every NATURAL embedded word w (unambiguous type, from a LinCE code-switched
tweet) with its real left context C (K tokens), build three inputs that end in
the same word w and have the same number of prefix tokens:

    natural     : C + w                                   (real, integrated context)
    own_random  : random K-token prefix from a monolingual tweet of w's OWN variety + w
    mat_random  : random K-token prefix from a monolingual tweet of the MATRIX variety + w

    natural pull    = s(natural)    - s(own_random)     (signed toward the matrix variety)
    unintegrated    = s(mat_random) - s(own_random)     (same variety, but no syntactic link)

Readings
  * natural pull asymmetric AND unintegrated pull asymmetric
        -> the asymmetry is lexical (which words get code-switched), not contextual
  * natural pull asymmetric, unintegrated pull symmetric
        -> the asymmetry needs the real, integrated matrix frame (MLF-consistent)
  * both symmetric -> exp01 asymmetry came from the probe read-out, not the representation

Read-out: same difference-of-means variety score as exp02 (differences only, so the
per-layer offset cancels).

Usage:
    python exp03_natural_vs_controlled.py                 # Qwen3-8B-Base
    python exp03_natural_vs_controlled.py --model <hf-id> --max_items 400
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from exp02_causal_patching import (PROJECT_DIR, LINCE_DIR, boot, get_layers, make_scorer,
                                   mono_sentences, run_scores, style, type_classes,
                                   variety_directions, with_bos, word_ids)

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_TEXT, C_MUTED = "#0b0b0b", "#52514e"
LANG = {"lang1": "MSA", "lang2": "EGY"}


def load_rows_with_ids():
    rows = []
    for split in ["train", "validation"]:
        df = pd.read_parquet(LINCE_DIR / f"{split}.parquet")
        for i, (ws, ls) in enumerate(zip(df.words, df.lid)):
            rows.append((f"{split}-{i}", list(ws), list(ls)))
    return rows


def prefix_pools_all_K(tok, sents, max_K):
    """variety pool: K -> list of token-id prefixes (word boundaries) of length exactly K."""
    pools = defaultdict(list)
    for ws, _ in sents:
        ws = [w for w in ws if not w.startswith(("http", "@"))]
        for j in range(1, len(ws) + 1):
            ids = tok(" ".join(ws[:j]), add_special_tokens=False).input_ids
            if len(ids) > max_K:
                break
            pools[len(ids)].append(ids)
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--max_items", type=int, default=400, help="per embedded variety")
    ap.add_argument("--min_K", type=int, default=3)
    ap.add_argument("--max_K", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / "exp03" / slug
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    n_layers = len(get_layers(model))
    R = int(round(0.8 * n_layers))

    rows = load_rows_with_ids()
    plain_rows = [(ws, ls) for _, ws, ls in rows]
    msa_types, egy_types = set(), set()
    m, e = type_classes(plain_rows)
    msa_types, egy_types = set(m), set(e)
    pools = {v: prefix_pools_all_K(tok, mono_sentences(plain_rows, v), args.max_K) for v in ["MSA", "EGY"]}

    # ---------------- natural embedded items ----------------
    items = {"MSA": [], "EGY": []}
    for sid, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs:
            continue
        n_egy, n_msa = langs.count("EGY"), langs.count("MSA")
        if n_egy == n_msa:
            continue
        matrix = "EGY" if n_egy > n_msa else "MSA"
        for i, (w, l) in enumerate(zip(ws, ls)):
            if l not in LANG or LANG[l] == matrix or i == 0:
                continue
            own = LANG[l]
            if (own == "MSA" and w not in msa_types) or (own == "EGY" and w not in egy_types):
                continue
            left = [x for x in ws[:i]]
            left_ids = tok(" ".join(left), add_special_tokens=False).input_ids
            K = len(left_ids)
            if K < args.min_K or K > args.max_K or not pools[own].get(K) or not pools[matrix].get(K):
                continue
            n_left_matrix = sum(1 for x in ls[:i] if x in LANG and LANG[x] == matrix)
            items[own].append({"sid": sid, "word": w, "own": own, "matrix": matrix, "K": K,
                               "n_left_matrix": n_left_matrix, "left_ids": left_ids})
    for v in items:
        if len(items[v]) > args.max_items:
            idx = rng.choice(len(items[v]), args.max_items, replace=False)
            items[v] = [items[v][i] for i in sorted(idx)]
    print(f"natural embedded items: MSA-in-EGY={len(items['MSA'])}  EGY-in-MSA={len(items['EGY'])}")

    print("Estimating variety directions ...")
    mu_msa, mu_egy = variety_directions(model, tok, plain_rows, device, seed=args.seed)
    score = make_scorer(mu_msa, mu_egy, device)

    # ---------------- run ----------------
    recs = []
    all_items = items["MSA"] + items["EGY"]
    for n, it in enumerate(all_items):
        w_ids = word_ids(tok, it["word"])
        K = it["K"]
        own_p = pools[it["own"]][K][rng.integers(len(pools[it["own"]][K]))]
        mat_p = pools[it["matrix"]][K][rng.integers(len(pools[it["matrix"]][K]))]
        s = {c: run_scores(model, [with_bos(tok, p + w_ids)], device, score)[0]
             for c, p in [("natural", it["left_ids"]), ("own_random", own_p), ("mat_random", mat_p)]}
        sign = 1.0 if it["matrix"] == "EGY" else -1.0   # positive = toward matrix variety
        for L in range(n_layers + 1):
            recs.append({"item": n, "own": it["own"], "word": it["word"], "K": K,
                         "n_left_matrix": it["n_left_matrix"], "layer": L,
                         "natural_pull": sign * (s["natural"][L] - s["own_random"][L]),
                         "unintegrated_pull": sign * (s["mat_random"][L] - s["own_random"][L])})
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(all_items)} ({time.time()-T0:.0f}s)")
    df = pd.DataFrame(recs)
    df.to_csv(out / "X_long.csv", index=False)

    agg = []
    for (own, L), d in df.groupby(["own", "layer"]):
        for metric in ["natural_pull", "unintegrated_pull"]:
            mean, lo, hi = boot(d[metric].values)
            agg.append({"embedded": own, "layer": L, "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "n": len(d)})
    agg = pd.DataFrame(agg)
    agg.to_csv(out / "X_pull_by_layer.csv", index=False)

    def asym(metric, L):
        a = df[(df.own == "MSA") & (df.layer == L)][metric].values
        b = df[(df.own == "EGY") & (df.layer == L)][metric].values
        brng = np.random.default_rng(1)
        diffs = [a[brng.integers(0, len(a), len(a))].mean() - b[brng.integers(0, len(b), len(b))].mean()
                 for _ in range(5000)]
        return {"MSA_in_EGY": float(a.mean()), "EGY_in_MSA": float(b.mean()),
                "difference": float(a.mean() - b.mean()),
                "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}

    summary = {"model": args.model, "readout_layer": R,
               "n_items": {v: len(items[v]) for v in items},
               "mean_prefix_tokens": {v: float(np.mean([i["K"] for i in items[v]])) if items[v] else None
                                      for v in items},
               "at_R": {m: asym(m, R) for m in ["natural_pull", "unintegrated_pull"]},
               "at_last_layer": {m: asym(m, n_layers) for m in ["natural_pull", "unintegrated_pull"]},
               "minutes": round((time.time() - T0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, own, title in [(axes[0], "MSA", "MSA word embedded in EGY tweet"),
                           (axes[1], "EGY", "EGY word embedded in MSA tweet")]:
        for metric, col, label in [("natural_pull", C_ORANGE if own == "MSA" else C_BLUE, "real (integrated) context"),
                                   ("unintegrated_pull", C_MUTED, "random matrix-variety prefix")]:
            d = agg[(agg.embedded == own) & (agg.metric == metric)]
            ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
            ax.plot(d.layer, d["mean"], color=col, lw=2, label=label)
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.axvline(R, color=C_MUTED, lw=0.8, ls=":")
        n = int(agg[agg.embedded == own].n.iloc[0]) if len(agg[agg.embedded == own]) else 0
        ax.set_title(f"{title} (n={n})", color=C_TEXT, fontsize=10)
        ax.set_xlabel("Layer")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        style(ax)
    axes[0].set_ylabel("Pull toward matrix variety\n(Δ variety score vs. own-variety prefix)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"X. Natural vs. unintegrated context — {slug}", x=0.01, ha="left", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "X_natural_vs_unintegrated.png", dpi=200)
    plt.close(fig)

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
