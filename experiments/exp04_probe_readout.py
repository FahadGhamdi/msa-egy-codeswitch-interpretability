"""
Experiment 04 - Same design as exp03, but with (i) a robust read-out and (ii) a
test of the "host composition" explanation of the asymmetry.

(i) Read-out. The difference-of-means score used in exp02/03 is dominated by a
few massive-activation dimensions in middle layers. Here every layer gets a
standardised logistic probe (z-scored features, as in exp01), trained on words
from monolingual LinCE tweets. All effects are reported in probe LOGITS
(positive = EGY), which are unbounded and comparable across directions.

(ii) Host composition. Natural MSA tweets that host an EGY word may themselves be
less "MSA" (informal register) than a random monolingual MSA tweet, and vice
versa. We therefore also score the LAST word of each prefix:
    host strength = logit at the prefix's last word, signed toward the matrix variety.
If natural hosts are weaker than random hosts more for one direction, the
asymmetry is a property of the host tweets, not of how the model integrates.

Conditions per natural embedded word w (left-context length K tokens):
    natural     : real left context + w
    own_random  : random K-token prefix from a monolingual tweet of w's own variety + w
    mat_random  : random K-token prefix from a monolingual tweet of the matrix variety + w

Usage:
    python exp04_probe_readout.py                  # Qwen3-8B-Base
    python exp04_probe_readout.py --model <hf-id>
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from exp02_causal_patching import (PROJECT_DIR, boot, get_layers, mono_sentences, style, type_classes, with_bos, word_ids)
from exp03_natural_vs_controlled import LANG, load_rows_with_ids, prefix_pools_all_K

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_TEXT, C_MUTED = "#0b0b0b", "#52514e"


def word_vectors(model, tok, sents, lab, device, max_words):
    """Last-subword hidden states (all layers) of words labelled `lab` in the given tweets."""
    feats, n = [], 0
    for ws, ls in sents:
        text, spans = "", []
        for w in ws:
            if text:
                text += " "
            spans.append((len(text), len(text) + len(w)))
            text += w
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=128)
        offs = enc.pop("offset_mapping")[0].tolist()
        c2w = np.full(len(text) + 1, -1)
        for wi, (s, e) in enumerate(spans):
            c2w[s:e] = wi
        last = {}
        for ti, (s, e) in enumerate(offs):
            if e > s and c2w[e - 1] >= 0:
                last[int(c2w[e - 1])] = ti
        pos = [last[wi] for wi, l in enumerate(ls) if l == lab and wi in last]
        if not pos:
            continue
        with torch.no_grad():
            hs = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
        feats.append(torch.stack([h[0, pos] for h in hs], 1).float().clamp(-6e4, 6e4).half().cpu().numpy())
        n += len(pos)
        if n >= max_words:
            break
    return np.concatenate(feats, 0)  # [N, L+1, d]


def train_probes(X_msa, X_egy, seed):
    from sklearn.linear_model import LogisticRegression
    n_layers = X_msa.shape[1]
    probes = []
    for L in range(n_layers):
        X = np.concatenate([X_msa[:, L], X_egy[:, L]]).astype(np.float32)
        y = np.r_[np.zeros(len(X_msa)), np.ones(len(X_egy))]
        mu, sd = X.mean(0), X.std(0) + 1e-4
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit((X - mu) / sd, y)
        w = clf.coef_[0] / sd
        b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
        probes.append((torch.tensor(w, dtype=torch.float32), b))
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--max_items", type=int, default=400)
    ap.add_argument("--probe_words", type=int, default=6000, help="training words per variety")
    ap.add_argument("--min_K", type=int, default=3)
    ap.add_argument("--max_K", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / "exp04" / slug
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
    plain = [(ws, ls) for _, ws, ls in rows]
    m, e = type_classes(plain)
    msa_types, egy_types = set(m), set(e)
    pools = {v: prefix_pools_all_K(tok, mono_sentences(plain, v), args.max_K) for v in ["MSA", "EGY"]}

    # ---------------- probes (monolingual tweets, excluding CS sentences) ----------------
    print("Training per-layer probes ...")
    mono = {v: mono_sentences(plain, v) for v in ["MSA", "EGY"]}
    for v in mono:
        idx = rng.permutation(len(mono[v]))
        mono[v] = [mono[v][i] for i in idx]
    X_msa = word_vectors(model, tok, mono["MSA"], "lang1", device, args.probe_words)
    X_egy = word_vectors(model, tok, mono["EGY"], "lang2", device, args.probe_words)
    # hold out 20% of the monolingual words to describe each class's logit distribution
    cut_m, cut_e = int(0.8 * len(X_msa)), int(0.8 * len(X_egy))
    probes = train_probes(X_msa[:cut_m], X_egy[:cut_e], args.seed)
    W = torch.stack([w for w, _ in probes]).to(device)          # [L+1, d]
    B = torch.tensor([b for _, b in probes], device=device)     # [L+1]
    Wn, Bn = W.cpu().numpy(), B.cpu().numpy()
    mono_stats = []
    for L in range(Wn.shape[0]):
        zm = X_msa[cut_m:, L].astype(np.float32) @ Wn[L] + Bn[L]
        ze = X_egy[cut_e:, L].astype(np.float32) @ Wn[L] + Bn[L]
        mono_stats.append({"layer": L, "msa_own_mean": float((-zm).mean()), "msa_own_sd": float(zm.std()),
                           "egy_own_mean": float(ze.mean()), "egy_own_sd": float(ze.std()),
                           "heldout_acc": float(((zm < 0).sum() + (ze > 0).sum()) / (len(zm) + len(ze)))})
    mono_stats = pd.DataFrame(mono_stats)
    mono_stats.to_csv(out / "Y_mono_logit_stats.csv", index=False)
    del X_msa, X_egy
    print(f"  probes trained ({time.time()-T0:.0f}s)")

    def logits_at(ids, positions):
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
        H = torch.stack([h[0, positions].float() for h in hs])   # [L+1, P, d]
        return ((H * W[:, None, :]).sum(-1) + B[:, None]).cpu().numpy()  # [L+1, P]

    # ---------------- natural embedded items (same selection as exp03) ----------------
    items = {"MSA": [], "EGY": []}
    for sid, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs or langs.count("EGY") == langs.count("MSA"):
            continue
        matrix = "EGY" if langs.count("EGY") > langs.count("MSA") else "MSA"
        for i, (w, l) in enumerate(zip(ws, ls)):
            if l not in LANG or LANG[l] == matrix or i == 0:
                continue
            own = LANG[l]
            if w not in (msa_types if own == "MSA" else egy_types):
                continue
            left_ids = tok(" ".join(ws[:i]), add_special_tokens=False).input_ids
            K = len(left_ids)
            if K < args.min_K or K > args.max_K or not pools[own].get(K) or not pools[matrix].get(K):
                continue
            left_lang = [LANG[x] for x in ls[:i] if x in LANG]
            items[own].append({"sid": sid, "word": w, "own": own, "matrix": matrix, "K": K,
                               "n_left_matrix": left_lang.count(matrix),
                               "n_left_own": left_lang.count(own), "left_ids": left_ids})
    for v in items:
        if len(items[v]) > args.max_items:
            keep = rng.choice(len(items[v]), args.max_items, replace=False)
            items[v] = [items[v][i] for i in sorted(keep)]
    print(f"natural embedded items: MSA-in-EGY={len(items['MSA'])}  EGY-in-MSA={len(items['EGY'])}")

    recs = []
    all_items = items["MSA"] + items["EGY"]
    for n, it in enumerate(all_items):
        w_ids = word_ids(tok, it["word"])
        K = it["K"]
        prefixes = {"natural": it["left_ids"],
                    "own_random": pools[it["own"]][K][rng.integers(len(pools[it["own"]][K]))],
                    "mat_random": pools[it["matrix"]][K][rng.integers(len(pools[it["matrix"]][K]))]}
        sign = 1.0 if it["matrix"] == "EGY" else -1.0          # positive = toward matrix variety
        z = {}
        for c, p in prefixes.items():
            ids = with_bos(tok, p + w_ids)
            z[c] = logits_at(ids, [len(ids) - 1, len(ids) - 1 - len(w_ids)])  # target, last prefix token
        for L in range(n_layers + 1):
            recs.append({
                "item": n, "own": it["own"], "word": it["word"], "K": K, "layer": L,
                "n_left_matrix": it["n_left_matrix"], "n_left_own": it["n_left_own"],
                "natural_pull": sign * (z["natural"][L, 0] - z["own_random"][L, 0]),
                "unintegrated_pull": sign * (z["mat_random"][L, 0] - z["own_random"][L, 0]),
                "host_natural": sign * z["natural"][L, 1],
                "host_random": sign * z["mat_random"][L, 1],
                "target_own_logit": -sign * z["own_random"][L, 0],   # how strongly w signals its own variety
            })
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(all_items)} ({time.time()-T0:.0f}s)")
    df = pd.DataFrame(recs)
    df.to_csv(out / "Y_long.csv", index=False)

    # ---------------- aggregates ----------------
    metrics = ["natural_pull", "unintegrated_pull", "host_natural", "host_random", "target_own_logit"]
    agg = []
    for subset, dsub in [("all", df), ("matrix_left>=3", df[df.n_left_matrix >= 3])]:
        for (own, L), d in dsub.groupby(["own", "layer"]):
            for mtr in metrics:
                mean, lo, hi = boot(d[mtr].values)
                agg.append({"subset": subset, "embedded": own, "layer": L, "metric": mtr,
                            "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(d)})
    agg = pd.DataFrame(agg)
    agg.to_csv(out / "Y_by_layer.csv", index=False)

    def contrast(d, metric):
        a = d[d.own == "MSA"][metric].values
        b = d[d.own == "EGY"][metric].values
        if len(a) < 5 or len(b) < 5:
            return None
        brng = np.random.default_rng(1)
        diffs = [a[brng.integers(0, len(a), len(a))].mean() - b[brng.integers(0, len(b), len(b))].mean()
                 for _ in range(5000)]
        return {"MSA_in_EGY": float(a.mean()), "EGY_in_MSA": float(b.mean()), "difference": float(a.mean() - b.mean()),
                "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5)),
                "n": [int(len(a)), int(len(b))]}

    summary = {"model": args.model, "readout_layer": R, "units": "probe logits, positive = toward matrix variety"}
    for subset, dsub in [("all", df), ("matrix_left>=3", df[df.n_left_matrix >= 3])]:
        for L in [R, n_layers - 1]:
            d = dsub[dsub.layer == L]
            summary[f"{subset}@L{L}"] = {m: contrast(d, m) for m in metrics}
    # robustness: is the markedness asymmetry just the probe's intercept/scale?
    rob = {}
    for L in [R, n_layers - 1]:
        st = mono_stats.set_index("layer").loc[L]
        d = df[(df.layer == L) & (df.n_left_matrix >= 3)]
        rob[f"L{L}"] = {
            "heldout_probe_accuracy": float(st.heldout_acc),
            "monolingual_own_logit": {"MSA": [float(st.msa_own_mean), float(st.msa_own_sd)],
                                      "EGY": [float(st.egy_own_mean), float(st.egy_own_sd)]},
            "target_own_in_sd_units": {
                "MSA": float(d[d.own == "MSA"].target_own_logit.mean() / st.msa_own_sd),
                "EGY": float(d[d.own == "EGY"].target_own_logit.mean() / st.egy_own_sd)},
            "pull_natural_in_sd_units": {
                "MSA": float(d[d.own == "MSA"].natural_pull.mean() / st.msa_own_sd),
                "EGY": float(d[d.own == "EGY"].natural_pull.mean() / st.egy_own_sd)},
            "flip_rate_natural": {v: float(((d[d.own == v].target_own_logit - d[d.own == v].natural_pull) < 0).mean())
                                  for v in ["MSA", "EGY"]},
            "flip_rate_random_matrix_prefix": {
                v: float(((d[d.own == v].target_own_logit - d[d.own == v].unintegrated_pull) < 0).mean())
                for v in ["MSA", "EGY"]},
        }
    summary["robustness"] = rob
    summary["minutes"] = round((time.time() - T0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("natural_pull", "Pull: real context"), ("unintegrated_pull", "Pull: random matrix-variety prefix"),
              ("host", "Host strength: last prefix word")]
    sub = agg[agg.subset == "matrix_left>=3"]
    for ax, (mtr, title) in zip(axes, panels):
        for own, col, lab in [("MSA", C_ORANGE, "MSA word in EGY tweet"), ("EGY", C_BLUE, "EGY word in MSA tweet")]:
            if mtr == "host":
                for hm, ls, hl in [("host_natural", "-", "natural host"), ("host_random", "--", "random host")]:
                    d = sub[(sub.embedded == own) & (sub.metric == hm)]
                    ax.plot(d.layer, d["mean"], color=col, lw=2, ls=ls, label=f"{lab}: {hl}")
            else:
                d = sub[(sub.embedded == own) & (sub.metric == mtr)]
                ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
                ax.plot(d.layer, d["mean"], color=col, lw=2, label=f"{lab} (n={int(d.n.iloc[0])})" if len(d) else lab)
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.axvline(R, color=C_MUTED, lw=0.8, ls=":")
        ax.set_title(title, color=C_TEXT, fontsize=10)
        ax.set_xlabel("Layer")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Probe logit, toward matrix variety")
    fig.suptitle(f"Y. Probe read-out, items with ≥3 matrix words on the left — {slug}",
                 x=0.01, ha="left", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "Y_probe_readout.png", dpi=200)
    plt.close(fig)

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
