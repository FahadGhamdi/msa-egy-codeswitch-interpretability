"""
Experiment 01 - Where does the model encode "which variety is this word in",
and what happens to that signal at a code-switch point?

Data  : LinCE lid_msaea (train + validation; natural MSA <-> Egyptian tweets)
        lang1 = MSA, lang2 = Egyptian Arabic (EGY)
Model : any Hugging Face causal LM (default Qwen/Qwen3-8B-Base)

Analyses (one linear probe per layer, trained ONLY on monolingual sentences)
  A. Layer-wise probe accuracy on:
       - held-out monolingual sentences
       - held-out words whose word TYPE never appears in probe training
       - "ambiguous forms": word types that occur with BOTH labels in LinCE
         (the surface form alone cannot decide; only context can)
       - words inside code-switched sentences
     compared with a lexical look-up baseline (majority label per word type).
  B. Matrix-language pull (MLF): in code-switched sentences, words from the
     minority variety ("embedded") vs. the majority variety ("matrix").
     Does the representation of an embedded word drift toward the matrix
     variety as depth increases?
  C. Switch-point dynamics: probe P(EGY) at word offsets -3..+3 around every
     switch point, per layer, separately for MSA->EGY and EGY->MSA.
  D. Control: a "local" probe (trained with word-level labels incl. half of the
     CS sentences) vs. the monolingual probe on held-out CS sentences — is the
     embedded word's variety LOST, or present but not dominant?
  E. Pull as a function of how many matrix words precede the embedded word.

Usage (project folder, environment active):
    python exp01_probe_switch.py                              # full run
    python exp01_probe_switch.py --model Qwen/Qwen2.5-0.5B-Instruct --n_mono 150   # quick smoke test
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

LINCE_DIR = PROJECT_DIR / "data" / "raw" / "lince_msaea" / "lid_msaea"
LANGS = {"lang1": 0, "lang2": 1}  # 0 = MSA, 1 = EGY
OFFSETS = list(range(-3, 4))

# Figure colours (validated categorical slots, light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_sentences(n_mono, seed):
    rows = []
    for split in ["train", "validation"]:
        df = pd.read_parquet(LINCE_DIR / f"{split}.parquet")
        for _, r in df.iterrows():
            rows.append({"sid": f"{split}-{r['idx']}", "words": list(r["words"]), "labels": list(r["lid"])})
    sents = pd.DataFrame(rows)

    def category(labels):
        s = set(labels)
        if {"lang1", "lang2"} <= s:
            return "cs"
        if "lang2" in s:
            return "egy_only"
        if "lang1" in s:
            return "msa_only"
        return "none"

    sents["cat"] = sents["labels"].map(category)
    rng = np.random.default_rng(seed)
    keep = [sents[sents.cat == "cs"]]
    for cat in ["msa_only", "egy_only"]:
        sub = sents[sents.cat == cat]
        n = min(n_mono, len(sub))
        keep.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
    out = pd.concat(keep, ignore_index=True)
    print("Sentences available:", dict(sents.cat.value_counts()))
    print("Sentences used     :", dict(out.cat.value_counts()))
    return out


# ----------------------------------------------------------------------
# Hidden-state extraction (one vector per word = its last sub-word token)
# ----------------------------------------------------------------------
def extract(model, tok, sents, device, max_tokens=256):
    all_h, meta = [], []
    t0 = time.time()
    for i, r in enumerate(sents.itertuples()):
        words, labels = r.words, r.labels
        text, spans, pos = "", [], 0
        for w in words:
            if text:
                text += " "
            start = len(text)
            text += w
            spans.append((start, len(text)))
        char2word = np.full(len(text) + 1, -1)
        for wi, (s, e) in enumerate(spans):
            char2word[s:e] = wi

        enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
                  truncation=True, max_length=max_tokens)
        offsets = enc.pop("offset_mapping")[0].tolist()
        last_tok = {}
        for ti, (s, e) in enumerate(offsets):
            if e > s and char2word[e - 1] >= 0:
                last_tok[int(char2word[e - 1])] = ti

        keep = [wi for wi, l in enumerate(labels) if l in LANGS and wi in last_tok]
        if not keep:
            continue
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        idx = torch.tensor([last_tok[wi] for wi in keep], device=device)
        hs = torch.stack([h[0, idx] for h in out.hidden_states])  # [L+1, n_words, d]
        hs = hs.float().clamp(-6e4, 6e4).to(torch.float16).cpu().numpy()
        all_h.append(hs)

        lang_seq = [wi for wi, l in enumerate(labels) if l in LANGS]
        for wi in keep:
            meta.append({
                "sid": r.sid, "cat": r.cat, "word_idx": wi,
                "lang_pos": lang_seq.index(wi),  # position among lang1/lang2 words
                "word": words[wi], "y": LANGS[labels[wi]],
            })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sents)} sentences  ({time.time()-t0:.0f}s)")
    H = np.concatenate(all_h, axis=1)  # [L+1, N, d]
    return H, pd.DataFrame(meta)


# ----------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------
def fit_probe(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=seed))
    clf.fit(X.astype(np.float32), y)
    return clf


def balanced_acc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (p >= 0.5).astype(int))) if len(set(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--n_mono", type=int, default=1000, help="monolingual sentences per variety")
    ap.add_argument("--max_train_words", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out_dir = PROJECT_DIR / "results" / "exp01" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    # ---------------- data + activations ----------------
    sents = load_sentences(args.n_mono, args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    H, meta = extract(model, tok, sents, device)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    n_layers = H.shape[0]
    print(f"Activations: {H.shape} (layers incl. embeddings, words, dim)")

    # ---------------- splits ----------------
    mono_sids = meta.loc[meta.cat != "cs", "sid"].unique()
    test_sids = set(rng.choice(mono_sids, int(0.2 * len(mono_sids)), replace=False))
    meta["split"] = np.where(meta.cat == "cs", "cs",
                             np.where(meta.sid.isin(test_sids), "mono_test", "mono_train"))
    tr = np.where(meta.split == "mono_train")[0]
    if len(tr) > args.max_train_words:
        tr = rng.choice(tr, args.max_train_words, replace=False)
    train_types = set(meta.word.iloc[tr])

    # word types that occur with both labels anywhere in LinCE (train + validation)
    label_by_type = defaultdict(Counter)
    for split in ["train", "validation"]:
        df = pd.read_parquet(LINCE_DIR / f"{split}.parquet")
        for ws, ls in zip(df.words, df.lid):
            for w, l in zip(ws, ls):
                if l in LANGS:
                    label_by_type[w][l] += 1
    ambiguous = {w for w, c in label_by_type.items() if c["lang1"] >= 2 and c["lang2"] >= 2}

    mono_test = meta.split == "mono_test"
    eval_sets = {
        "mono_test": mono_test.values,
        "unseen_types": (mono_test & ~meta.word.isin(train_types)).values,
        "ambiguous_forms": ((meta.split != "mono_train") & meta.word.isin(ambiguous)).values,
        "cs_words": (meta.split == "cs").values,
    }
    print("Eval set sizes:", {k: int(v.sum()) for k, v in eval_sets.items()})

    # lexical look-up baseline (majority label per type in probe training data)
    tr_df = meta.iloc[tr]
    type_major = tr_df.groupby("word").y.agg(lambda s: int(s.mean() >= 0.5)).to_dict()
    global_major = int(tr_df.y.mean() >= 0.5)
    lex_pred = meta.word.map(type_major).fillna(global_major).astype(float).values
    baseline = {k: balanced_acc(meta.y.values[m], lex_pred[m]) for k, m in eval_sets.items()}
    print("Lexical baseline:", {k: round(v, 3) for k, v in baseline.items()})

    # ---------------- probes per layer ----------------
    y = meta.y.values
    P = np.zeros((n_layers, len(meta)), dtype=np.float32)  # P(EGY) for every word, every layer
    rows = []
    for L in range(n_layers):
        t = time.time()
        clf = fit_probe(H[L, tr], y[tr], args.seed)
        P[L] = clf.predict_proba(H[L].astype(np.float32))[:, 1]
        row = {"layer": L}
        for k, m in eval_sets.items():
            row[k] = balanced_acc(y[m], P[L, m])
        rows.append(row)
        print(f"  layer {L:02d}: " + "  ".join(f"{k}={row[k]:.3f}" for k in eval_sets) + f"  ({time.time()-t:.0f}s)")
    probe_df = pd.DataFrame(rows)
    for k, v in baseline.items():
        probe_df[f"baseline_{k}"] = v
    probe_df.to_csv(out_dir / "A_probe_by_layer.csv", index=False)

    # ---------------- B. matrix-language pull ----------------
    cs = meta[meta.split == "cs"].copy()
    counts = cs.groupby("sid").y.agg(["sum", "count"])
    counts["matrix"] = np.where(counts["sum"] * 2 > counts["count"], 1,
                                np.where(counts["sum"] * 2 < counts["count"], 0, -1))
    cs["matrix"] = cs.sid.map(counts["matrix"])
    cs = cs[cs.matrix >= 0]
    cs["role"] = np.where(cs.y == cs.matrix, "matrix", "embedded")
    # number of matrix-variety words to the LEFT of each word (causal context)
    cs = cs.sort_values(["sid", "lang_pos"])
    cs["is_matrix_word"] = (cs.y == cs.matrix).astype(int)
    cs["n_left_matrix"] = cs.groupby("sid").is_matrix_word.cumsum() - cs.is_matrix_word
    sent_codes, sent_index = pd.factorize(cs.sid)

    def cluster_boot_ci(values, mask, n_boot=1000):
        """95% CI of the mean, resampling whole sentences (cluster bootstrap)."""
        v, s = values[mask], sent_codes[mask]
        n_s = len(sent_index)
        sums = np.bincount(s, weights=v, minlength=n_s)
        cnts = np.bincount(s, minlength=n_s).astype(float)
        present = np.where(cnts > 0)[0]
        brng = np.random.default_rng(args.seed)
        draws = brng.choice(present, (n_boot, len(present)), replace=True)
        means = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    b_rows = []
    for L in range(n_layers):
        p_egy = P[L, cs.index.values]
        p_own = np.where(cs.y.values == 1, p_egy, 1 - p_egy)
        for role in ["matrix", "embedded"]:
            m = (cs.role == role).values
            for var, vy in [("MSA", 0), ("EGY", 1), ("all", None)]:
                mm = m if vy is None else m & (cs.y.values == vy)
                if mm.sum():
                    lo, hi = cluster_boot_ci(p_own, mm)
                    b_rows.append({"layer": L, "role": role, "variety": var, "n": int(mm.sum()),
                                   "p_own_label": float(p_own[mm].mean()), "ci_lo": lo, "ci_hi": hi,
                                   "acc_own_label": float((p_own[mm] >= 0.5).mean())})
    b_df = pd.DataFrame(b_rows)
    b_df.to_csv(out_dir / "B_matrix_pull.csv", index=False)

    # ---------------- D. is the embedded word's identity LOST or only NOT DOMINANT? ----------------
    # A "local" probe is trained on word-level labels that include half of the
    # code-switched sentences, so it is rewarded for recovering each word's own
    # variety even against its sentence. If the local probe recovers embedded
    # words that the monolingual probe misses, the information is still present
    # in the residual stream but is not the dominant variety direction.
    cs_sids = cs.sid.unique()
    cs_train_sids = set(rng.choice(cs_sids, len(cs_sids) // 2, replace=False))
    cs_tr = cs.index[cs.sid.isin(cs_train_sids)].values
    cs_te = cs[~cs.sid.isin(cs_train_sids)]
    local_tr = np.concatenate([tr, cs_tr])
    d_rows = []
    for L in range(n_layers):
        t = time.time()
        clf = fit_probe(H[L, local_tr], y[local_tr], args.seed)
        p_loc = clf.predict_proba(H[L, cs_te.index.values].astype(np.float32))[:, 1]
        p_mono = P[L, cs_te.index.values]
        yy = cs_te.y.values
        for role in ["matrix", "embedded"]:
            m = (cs_te.role == role).values
            own = lambda p: np.where(yy == 1, p, 1 - p)  # noqa: E731
            d_rows.append({"layer": L, "role": role, "n": int(m.sum()),
                           "p_own_mono_probe": float(own(p_mono)[m].mean()),
                           "p_own_local_probe": float(own(p_loc)[m].mean()),
                           "acc_local_probe": float((own(p_loc)[m] >= 0.5).mean())})
        print(f"  local probe layer {L:02d} ({time.time()-t:.0f}s)")
    d_df = pd.DataFrame(d_rows)
    d_df.to_csv(out_dir / "D_local_vs_mono_probe.csv", index=False)

    # ---------------- E. does the pull grow with matrix context on the left? ----------------
    emb = cs[cs.role == "embedded"].copy()
    emb["left_bin"] = emb.n_left_matrix.clip(upper=3)
    e_rows = []
    for L in range(n_layers):
        p_egy = P[L, emb.index.values]
        p_own = np.where(emb.y.values == 1, p_egy, 1 - p_egy)
        for b in range(4):
            m = (emb.left_bin == b).values
            if m.sum():
                e_rows.append({"layer": L, "n_left_matrix": "3+" if b == 3 else str(b),
                               "n": int(m.sum()), "p_own_label": float(p_own[m].mean())})
    e_df = pd.DataFrame(e_rows)
    e_df.to_csv(out_dir / "E_pull_by_left_context.csv", index=False)

    # ---------------- C. switch-point dynamics ----------------
    c_acc = defaultdict(list)
    for sid, g in meta[meta.split == "cs"].groupby("sid"):
        g = g.sort_values("lang_pos")
        ys, idxs, lp = g.y.values, g.index.values, g.lang_pos.values
        for j in range(1, len(g)):
            if ys[j] != ys[j - 1] and lp[j] == lp[j - 1] + 1:
                direction = "MSA->EGY" if ys[j] == 1 else "EGY->MSA"
                for off in OFFSETS:
                    k = j + off
                    if 0 <= k < len(g):
                        c_acc[(direction, off)].append(idxs[k])
    c_rows = []
    for (direction, off), ids in c_acc.items():
        for L in range(n_layers):
            c_rows.append({"direction": direction, "offset": off, "layer": L,
                           "n": len(ids), "p_egy": float(P[L, ids].mean())})
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(out_dir / "C_switch_dynamics.csv", index=False)

    # ---------------- figures ----------------
    make_figures(probe_df, b_df, c_df, d_df, e_df, out_dir, slug)

    meta.assign(**{f"p_egy_L{L}": P[L] for L in range(n_layers)}).to_parquet(out_dir / "word_predictions.parquet")
    summary = {
        "model": args.model, "device": device, "n_layers_incl_emb": n_layers, "hidden_dim": int(H.shape[2]),
        "n_words": int(len(meta)), "n_train_words": int(len(tr)),
        "eval_set_sizes": {k: int(v.sum()) for k, v in eval_sets.items()},
        "lexical_baseline": baseline,
        "best_layer": {k: int(probe_df[k].idxmax()) for k in eval_sets if probe_df[k].notna().any()},
        "best_score": {k: float(probe_df[k].max()) for k in eval_sets if probe_df[k].notna().any()},
        "n_switch_points": {d: len(c_acc[(d, 0)]) for d in ["MSA->EGY", "EGY->MSA"]},
        "minutes": round((time.time() - T0) / 60, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults in {out_dir}")


# ----------------------------------------------------------------------
def style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)


def make_figures(probe_df, b_df, c_df, d_df, e_df, out_dir, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    title_kw = dict(color=C_TEXT, fontsize=11, loc="left")

    # A. probe accuracy by layer
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    series = [("mono_test", "Monolingual (held-out)", C_BLUE),
              ("unseen_types", "Unseen word types", C_ORANGE),
              ("ambiguous_forms", "Ambiguous forms", C_AQUA),
              ("cs_words", "Words in CS sentences", C_VIOLET)]
    for key, label, col in series:
        ax.plot(probe_df.layer, probe_df[key], color=col, lw=2, label=label)
        ax.axhline(probe_df[f"baseline_{key}"].iloc[0], color=col, lw=1, ls=":", alpha=0.9)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Balanced accuracy (MSA vs EGY)")
    ax.set_ylim(0.45, 1.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"A. Variety probe by layer — {model_name}\n(dotted = lexical look-up baseline)", **title_kw)
    fig.tight_layout()
    fig.savefig(out_dir / "A_probe_by_layer.png", dpi=200)
    plt.close(fig)

    # B. matrix vs embedded
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for role, col, label in [("matrix", C_BLUE, "Matrix-variety words"),
                             ("embedded", C_ORANGE, "Embedded-variety words")]:
        d = b_df[(b_df.role == role) & (b_df.variety == "all")]
        ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
        ax.plot(d.layer, d.p_own_label, color=col, lw=2, label=label)
    ax.axhline(0.5, color=C_MUTED, lw=1, ls="--")
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Mean P(word's own variety)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"B. Matrix-language pull in code-switched sentences — {model_name}\n(bands = 95% sentence-bootstrap CI)", **title_kw)
    fig.tight_layout()
    fig.savefig(out_dir / "B_matrix_pull.png", dpi=200)
    plt.close(fig)

    # C. switch dynamics heatmaps (diverging: MSA blue <-> grey <-> EGY orange)
    cmap = LinearSegmentedColormap.from_list("msa_egy", [C_BLUE, "#e8e8e6", C_ORANGE])
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for ax, direction in zip(axes, ["MSA->EGY", "EGY->MSA"]):
        d = c_df[c_df.direction == direction].pivot(index="layer", columns="offset", values="p_egy")
        im = ax.imshow(d.values, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
                       extent=[min(OFFSETS) - 0.5, max(OFFSETS) + 0.5, -0.5, d.shape[0] - 0.5])
        ax.axvline(-0.5, color=C_TEXT, lw=1.2)
        ax.set_xticks(OFFSETS)
        ax.set_xlabel("Word offset (0 = first word after switch)")
        n = int(c_df[(c_df.direction == direction) & (c_df.offset == 0)].n.iloc[0])
        ax.set_title(f"{direction}  (n={n})", color=C_TEXT, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Layer")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("Probe P(EGY)")
    fig.suptitle("C. Variety signal around switch points", x=0.02, ha="left", color=C_TEXT, fontsize=11)
    fig.savefig(out_dir / "C_switch_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # D. monolingual vs local probe on embedded words
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for role, col in [("matrix", C_BLUE), ("embedded", C_ORANGE)]:
        d = d_df[d_df.role == role]
        ax.plot(d.layer, d.p_own_mono_probe, color=col, lw=2, ls="-", label=f"{role.capitalize()} — monolingual probe")
        ax.plot(d.layer, d.p_own_local_probe, color=col, lw=2, ls="--", label=f"{role.capitalize()} — local probe")
    ax.axhline(0.5, color=C_MUTED, lw=1, ls=":")
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Mean P(word's own variety), held-out CS sentences")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"D. Is the embedded word's variety lost, or only not dominant? — {model_name}", **title_kw)
    fig.tight_layout()
    fig.savefig(out_dir / "D_local_vs_mono_probe.png", dpi=200)
    plt.close(fig)

    # E. pull as a function of matrix words already seen (ordinal -> one-hue ramp)
    ramp = ["#9ec5f0", "#5d9ee6", "#2a78d6", "#154b8f"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for b, col in zip(["0", "1", "2", "3+"], ramp):
        d = e_df[e_df.n_left_matrix == b]
        if len(d):
            ax.plot(d.layer, d.p_own_label, color=col, lw=2, label=f"{b} matrix words to the left (n={int(d.n.iloc[0])})")
    ax.axhline(0.5, color=C_MUTED, lw=1, ls=":")
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Embedded words: mean P(own variety)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"E. Does the pull grow with matrix context? — {model_name}", **title_kw)
    fig.tight_layout()
    fig.savefig(out_dir / "E_pull_by_left_context.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
