"""
Experiment 02 - Causal test of the "sticky dialect" asymmetry.

Question
--------
Exp01 showed (correlationally) that an unambiguous MSA word placed after an
Egyptian context is re-represented as Egyptian, while an unambiguous EGY word
after an MSA context keeps its identity. Exp02 tests this CAUSALLY with
controlled minimal pairs and activation patching.

Design
------
Minimal pairs: the SAME target word w after a prefix of exactly K tokens taken
from a monolingual EGY tweet vs. from a monolingual MSA tweet (plus w alone).
    MSA-target pairs : w is an unambiguous MSA type  (tests "overwriting")
    EGY-target pairs : w is an unambiguous EGY type  (tests "resistance")

Read-out: difference-of-means "variety direction" per layer, estimated on
monolingual LinCE words:  s_L(h) = (h - mu_MSA)·(mu_EGY - mu_MSA) / ||mu_EGY - mu_MSA||^2
(0 = average MSA word, 1 = average EGY word).

  1. Context effect by layer: s_L(w | EGY prefix) vs s_L(w | MSA prefix) vs s_L(w alone).
  2. Activation patching (denoising). Clean run = EGY prefix, corrupt run = MSA
     prefix. At the target word position only, copy one component from the clean
     run into the corrupt run and measure s at the read-out layer R:
        resid  : residual stream entering layer L   (where is the context info?)
        attn   : attention output of layer L         (which layers MOVE it?)
        mlp    : MLP output of layer L               (which layers WRITE it?)
     Restoration = (s_patched - s_corrupt) / (s_clean - s_corrupt).
  3. Head-level patching in the top attention layers.

Usage (project folder, environment active):
    python exp02_causal_patching.py                         # Qwen3-8B-Base
    python exp02_causal_patching.py --model <hf-id> --n_pairs 150 --prefix_tokens 12
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
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_lince():
    rows = []
    for split in ["train", "validation"]:
        df = pd.read_parquet(LINCE_DIR / f"{split}.parquet")
        for ws, ls in zip(df.words, df.lid):
            rows.append((list(ws), list(ls)))
    return rows


def type_classes(rows, purity=0.95, min_count=5):
    cnt = defaultdict(Counter)
    for ws, ls in rows:
        for w, l in zip(ws, ls):
            if l in ("lang1", "lang2"):
                cnt[w][l] += 1
    msa, egy = [], []
    for w, c in cnt.items():
        n = c["lang1"] + c["lang2"]
        if n < min_count or w.startswith(("#", "@", "http")) or len(w) < 2:
            continue
        share = c["lang2"] / n
        if share <= 1 - purity:
            msa.append(w)
        elif share >= purity:
            egy.append(w)
    return sorted(msa), sorted(egy)


def mono_sentences(rows, variety):
    lab, other = ("lang1", "lang2") if variety == "MSA" else ("lang2", "lang1")
    return [(ws, ls) for ws, ls in rows if lab in ls and other not in ls]


def prefix_pool(tok, sents, K):
    """Word-boundary prefixes of monolingual tweets that tokenize to exactly K tokens."""
    pool = []
    for ws, _ in sents:
        ws = [w for w in ws if not w.startswith(("http", "@"))]
        for j in range(1, len(ws) + 1):
            ids = tok(" ".join(ws[:j]), add_special_tokens=False).input_ids
            if len(ids) == K:
                pool.append(ids)
                break
            if len(ids) > K:
                break
    return pool


def word_ids(tok, w):
    """Token ids of ' w' as it appears after a preceding word (works for BPE and SentencePiece)."""
    anchor = "و"
    a = tok(anchor, add_special_tokens=False).input_ids
    full = tok(anchor + " " + w, add_special_tokens=False).input_ids
    if full[:len(a)] == a and len(full) > len(a):
        return full[len(a):]
    return tok(" " + w, add_special_tokens=False).input_ids


def with_bos(tok, ids):
    bos = tok.bos_token_id
    add_bos = bos is not None and tok("a").input_ids[:1] == [bos]
    return ([bos] if add_bos else []) + ids


# ----------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------
def get_layers(model):
    return model.model.layers


def variety_directions(model, tok, rows, device, n_per=400, seed=0):
    """Mean last-subword hidden state of MSA / EGY words in monolingual tweets, per layer."""
    rng = np.random.default_rng(seed)
    sums, counts = {}, {}
    for var, lab in [("MSA", "lang1"), ("EGY", "lang2")]:
        sents = mono_sentences(rows, var)
        pick = rng.choice(len(sents), min(n_per, len(sents)), replace=False)
        acc, n = None, 0
        for i in pick:
            ws, ls = sents[i]
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
            h = torch.stack([x[0, pos].float().sum(0) for x in hs]).cpu()  # [L+1, d]
            acc = h if acc is None else acc + h
            n += len(pos)
        sums[var], counts[var] = acc, n
    mu_msa = sums["MSA"] / counts["MSA"]
    mu_egy = sums["EGY"] / counts["EGY"]
    return mu_msa, mu_egy


def make_scorer(mu_msa, mu_egy, device):
    d = (mu_egy - mu_msa)
    denom = (d * d).sum(-1)  # [L+1]
    mu_msa, d, denom = mu_msa.to(device), d.to(device), denom.to(device)

    def score(h, L):  # h: [..., d]
        return ((h.float() - mu_msa[L]) * d[L]).sum(-1) / denom[L]
    return score


def run_scores(model, ids_batch, device, score, layers=None):
    x = torch.tensor(ids_batch, device=device)
    with torch.no_grad():
        hs = model(input_ids=x, output_hidden_states=True).hidden_states
    Ls = range(len(hs)) if layers is None else layers
    return np.array([[float(score(hs[L][b, -1], L)) for L in Ls] for b in range(x.shape[0])])


def capture_clean(model, ids, device):
    """Cache resid_pre / attn_out / mlp_out / head inputs at the LAST position for every layer."""
    layers = get_layers(model)
    cache = {"resid": {}, "attn": {}, "mlp": {}, "heads": {}}
    hooks = []
    for L, layer in enumerate(layers):
        def pre(mod, args, kwargs, L=L):
            h = args[0] if args else kwargs["hidden_states"]
            cache["resid"][L] = h[0, -1].detach().clone()
        def attn_hook(mod, args, out, L=L):
            o = out[0] if isinstance(out, tuple) else out
            cache["attn"][L] = o[0, -1].detach().clone()
        def mlp_hook(mod, args, out, L=L):
            cache["mlp"][L] = out[0, -1].detach().clone()
        def oproj_pre(mod, args, L=L):
            cache["heads"][L] = args[0][0, -1].detach().clone()
        hooks += [layer.register_forward_pre_hook(pre, with_kwargs=True),
                  layer.self_attn.register_forward_hook(attn_hook),
                  layer.mlp.register_forward_hook(mlp_hook),
                  layer.self_attn.o_proj.register_forward_pre_hook(oproj_pre)]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device))
    finally:
        for h in hooks:
            h.remove()
    return cache


def patched_scores(model, ids, device, cache, component, targets, R, score, head_dim=None):
    """
    Batch element b gets ONE patch: targets[b] = layer L (resid/attn/mlp) or (L, head).
    Returns s_R for each batch element.
    """
    layers = get_layers(model)
    B = len(targets)
    hooks = []
    by_layer = defaultdict(list)
    for b, t in enumerate(targets):
        L = t[0] if isinstance(t, tuple) else t
        by_layer[L].append((b, t))
    for L, items in by_layer.items():
        layer = layers[L]
        if component == "resid":
            def pre(mod, args, kwargs, items=items, L=L):
                if args:
                    h = args[0].clone()
                    for b, _ in items:
                        h[b, -1] = cache["resid"][L]
                    return (h,) + tuple(args[1:]), kwargs
                h = kwargs["hidden_states"].clone()
                for b, _ in items:
                    h[b, -1] = cache["resid"][L]
                kwargs["hidden_states"] = h
                return args, kwargs
            hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        elif component == "attn":
            def fwd(mod, args, out, items=items, L=L):
                o = (out[0] if isinstance(out, tuple) else out).clone()
                for b, _ in items:
                    o[b, -1] = cache["attn"][L]
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            hooks.append(layer.self_attn.register_forward_hook(fwd))
        elif component == "mlp":
            def fwd(mod, args, out, items=items, L=L):
                o = out.clone()
                for b, _ in items:
                    o[b, -1] = cache["mlp"][L]
                return o
            hooks.append(layer.mlp.register_forward_hook(fwd))
        elif component == "head":
            def pre(mod, args, items=items, L=L):
                x = args[0].clone()
                for b, (_, hd) in items:
                    sl = slice(hd * head_dim, (hd + 1) * head_dim)
                    x[b, -1, sl] = cache["heads"][L][sl]
                return (x,) + tuple(args[1:])
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
    try:
        x = torch.tensor([ids] * B, device=device)
        with torch.no_grad():
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
        return np.array([float(score(hs[R][b, -1], R)) for b in range(B)])
    finally:
        for h in hooks:
            h.remove()


def boot(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--n_pairs", type=int, default=150, help="minimal pairs per target variety")
    ap.add_argument("--prefix_tokens", type=int, default=12)
    ap.add_argument("--top_attn_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reverse", action="store_true",
                    help="clean = MSA prefix, corrupt = EGY prefix (find heads that import MSA evidence)")
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / ("exp02_reverse" if args.reverse else "exp02") / slug
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    n_layers = len(get_layers(model))
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    R = int(round(0.8 * n_layers))  # read-out layer (hidden_states index)
    print(f"layers={n_layers} heads={n_heads} head_dim={head_dim} read-out layer R={R}")

    # ---------------- data ----------------
    rows = load_lince()
    msa_types, egy_types = type_classes(rows)
    pools = {v: prefix_pool(tok, mono_sentences(rows, v), args.prefix_tokens) for v in ["MSA", "EGY"]}
    print(f"unambiguous types: MSA={len(msa_types)} EGY={len(egy_types)}; "
          f"prefix pools (K={args.prefix_tokens}): MSA={len(pools['MSA'])} EGY={len(pools['EGY'])}")

    print("Estimating variety directions ...")
    mu_msa, mu_egy = variety_directions(model, tok, rows, device, seed=args.seed)
    score = make_scorer(mu_msa, mu_egy, device)
    # patching read-out: positive = toward the CLEAN run's variety (EGY normally, MSA with --reverse)
    score_p = (lambda h, L: -score(h, L)) if args.reverse else score

    pairs = []
    for target_var, types in [("MSA", msa_types), ("EGY", egy_types)]:
        words = rng.choice(types, min(args.n_pairs, len(types)), replace=len(types) < args.n_pairs)
        for w in words:
            w_ids = word_ids(tok, str(w))
            egy_p = pools["EGY"][rng.integers(len(pools["EGY"]))]
            msa_p = pools["MSA"][rng.integers(len(pools["MSA"]))]
            pairs.append({"target": target_var, "word": str(w),
                          "clean": with_bos(tok, (msa_p if args.reverse else egy_p) + w_ids),
                          "corrupt": with_bos(tok, (egy_p if args.reverse else msa_p) + w_ids),
                          "alone": with_bos(tok, tok(str(w), add_special_tokens=False).input_ids)})

    # ---------------- 1. context effect by layer ----------------
    print("Part 1: context effect by layer ...")
    s1 = []
    for i, p in enumerate(pairs):
        for cond in ["clean", "corrupt", "alone"]:
            sc = run_scores(model, [p[cond]], device, score)[0]
            for L, v in enumerate(sc):
                s1.append({"pair": i, "target": p["target"], "word": p["word"],
                           "prefix": {"clean": "MSA" if args.reverse else "EGY", "corrupt": "EGY" if args.reverse else "MSA", "alone": "none"}[cond],
                           "layer": L, "s": v})
    s1 = pd.DataFrame(s1)
    s1.to_csv(out / "P1_scores_long.csv", index=False)
    agg1 = []
    for (t, pre, L), d in s1.groupby(["target", "prefix", "layer"]):
        m, lo, hi = boot(d.s.values)
        agg1.append({"target": t, "prefix": pre, "layer": L, "s": m, "ci_lo": lo, "ci_hi": hi})
    agg1 = pd.DataFrame(agg1)
    agg1.to_csv(out / "P1_context_effect_by_layer.csv", index=False)

    # asymmetry statistic at read-out layer: context sensitivity of the SAME word,
    # s(w | EGY prefix) - s(w | MSA prefix). Hypothesis: larger for MSA targets.
    at_R = s1[s1.layer == R].pivot_table(index=["pair", "target"], columns="prefix", values="s").reset_index()
    at_R["pull_to_other"] = at_R.EGY - at_R.MSA
    asym = {t: boot(at_R[at_R.target == t].pull_to_other.values) for t in ["MSA", "EGY"]}
    d_msa, d_egy = at_R[at_R.target == "MSA"].pull_to_other.values, at_R[at_R.target == "EGY"].pull_to_other.values
    brng = np.random.default_rng(1)
    diff_boot = [d_msa[brng.integers(0, len(d_msa), len(d_msa))].mean() -
                 d_egy[brng.integers(0, len(d_egy), len(d_egy))].mean() for _ in range(5000)]
    asym_diff = (float(d_msa.mean() - d_egy.mean()), float(np.percentile(diff_boot, 2.5)),
                 float(np.percentile(diff_boot, 97.5)))

    # ---------------- 2. layer-level patching ----------------
    print("Part 2: layer-level activation patching ...")
    patch_layers = list(range(R))
    p2 = []
    for i, p in enumerate(pairs):
        s_clean = run_scores(model, [p["clean"]], device, score_p, [R])[0, 0]
        s_corr = run_scores(model, [p["corrupt"]], device, score_p, [R])[0, 0]
        cache = capture_clean(model, p["clean"], device)
        for comp in ["resid", "attn", "mlp"]:
            sp = patched_scores(model, p["corrupt"], device, cache, comp, patch_layers, R, score_p)
            for L, v in zip(patch_layers, sp):
                p2.append({"pair": i, "target": p["target"], "component": comp, "layer": L,
                           "s_clean": s_clean, "s_corrupt": s_corr, "s_patched": v})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} pairs ({time.time()-T0:.0f}s)")
    p2 = pd.DataFrame(p2)
    p2["delta"] = p2.s_patched - p2.s_corrupt
    p2["gap"] = p2.s_clean - p2.s_corrupt
    p2.to_csv(out / "P2_patching_long.csv", index=False)
    agg2 = []
    for (t, comp, L), d in p2.groupby(["target", "component", "layer"]):
        m, lo, hi = boot(d.delta.values)
        gap = d.gap.mean()
        agg2.append({"target": t, "component": comp, "layer": L, "delta": m, "ci_lo": lo, "ci_hi": hi,
                     "mean_gap": gap, "restoration": m / gap if abs(gap) > 1e-6 else np.nan})
    agg2 = pd.DataFrame(agg2)
    agg2.to_csv(out / "P2_patching_by_layer.csv", index=False)

    # ---------------- 3. head-level patching ----------------
    a = agg2[(agg2.target == "MSA") & (agg2.component == "attn")].sort_values("delta", ascending=False)
    top_layers = [int(L) for L in a.layer.head(args.top_attn_layers)]
    print(f"Part 3: head-level patching in layers {top_layers} ...")
    p3 = []
    msa_pairs = [(i, p) for i, p in enumerate(pairs) if p["target"] == "MSA"]
    for i, p in msa_pairs:
        s_corr = run_scores(model, [p["corrupt"]], device, score_p, [R])[0, 0]
        cache = capture_clean(model, p["clean"], device)
        for L in top_layers:
            sp = patched_scores(model, p["corrupt"], device, cache, "head",
                                [(L, h) for h in range(n_heads)], R, score_p, head_dim)
            for h, v in enumerate(sp):
                p3.append({"pair": i, "layer": L, "head": h, "delta": v - s_corr})
    p3 = pd.DataFrame(p3)
    agg3 = p3.groupby(["layer", "head"]).delta.agg(["mean", "std", "count"]).reset_index()
    agg3.to_csv(out / "P3_heads.csv", index=False)

    make_figures(agg1, agg2, agg3, R, out, slug)

    summary = {
        "model": args.model, "n_layers": n_layers, "readout_layer": R, "prefix_tokens": args.prefix_tokens,
        "n_pairs": {t: int(sum(p["target"] == t for p in pairs)) for t in ["MSA", "EGY"]},
        "n_unambiguous_types": {"MSA": len(msa_types), "EGY": len(egy_types)},
        "context_sensitivity_at_R (s_EGYprefix - s_MSAprefix)": {t: dict(zip(["mean", "ci_lo", "ci_hi"], v)) for t, v in asym.items()},
        "asymmetry_MSA_minus_EGY": dict(zip(["mean", "ci_lo", "ci_hi"], asym_diff)),
        "top_attn_layers_by_patching": top_layers,
        "top_heads": agg3.sort_values("mean", ascending=False).head(10)[["layer", "head", "mean"]].to_dict("records"),
        "peak_restoration": {
            f"{t}_{c}": {"layer": int(d.loc[d.delta.idxmax(), "layer"]), "restoration": float(d.restoration.max())}
            for (t, c), d in agg2.groupby(["target", "component"])},
        "minutes": round((time.time() - T0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults in {out}")


# ----------------------------------------------------------------------
def style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED)


def make_figures(agg1, agg2, agg3, R, out, slug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tkw = dict(color=C_TEXT, fontsize=11, loc="left")

    # P1: two panels, one per target variety
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, t in zip(axes, ["MSA", "EGY"]):
        for pre, col, ls in [("EGY", C_ORANGE, "-"), ("MSA", C_BLUE, "-")]:  # "word alone" omitted: position-0 attention-sink artefact
            d = agg1[(agg1.target == t) & (agg1.prefix == pre)]
            if pre != "none":
                ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
            ax.plot(d.layer, d.s, color=col, lw=2, ls=ls,
                    label={"EGY": "after EGY prefix", "MSA": "after MSA prefix", "none": "word alone"}[pre])
        ax.axhline(0, color=C_BLUE, lw=0.8, ls="--")
        ax.axhline(1, color=C_ORANGE, lw=0.8, ls="--")
        ax.axvline(R, color=C_MUTED, lw=0.8, ls=":")
        ax.set_title(f"Unambiguous {t} target word", color=C_TEXT, fontsize=10)
        ax.set_xlabel("Layer")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        style(ax)
    axes[0].set_ylabel("Variety score (0 = MSA mean, 1 = EGY mean)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"P1. Same word, different context — {slug}", x=0.01, ha="left", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "P1_context_effect.png", dpi=200)
    plt.close(fig)

    # P2: restoration curves for MSA targets (the overwriting case) + raw delta for EGY targets
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, t in zip(axes, ["MSA", "EGY"]):
        for comp, col in [("resid", C_TEXT), ("attn", C_ORANGE), ("mlp", C_AQUA)]:
            d = agg2[(agg2.target == t) & (agg2.component == comp)]
            ax.fill_between(d.layer, d.ci_lo, d.ci_hi, color=col, alpha=0.15, lw=0)
            ax.plot(d.layer, d.delta, color=col, lw=2, label={"resid": "residual stream", "attn": "attention output",
                                                               "mlp": "MLP output"}[comp])
        gap = agg2[(agg2.target == t)].mean_gap.iloc[0]
        ax.axhline(gap, color=C_MUTED, lw=1, ls="--")
        ax.axhline(0, color=C_MUTED, lw=0.8)
        ax.set_title(f"{t} target  (dashed = full context effect {gap:+.2f})", color=C_TEXT, fontsize=10)
        ax.set_xlabel("Patched layer (target position only)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        style(ax)
    axes[0].set_ylabel(f"Δ variety score at layer {R}\n(EGY-prefix → MSA-prefix run)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"P2. Where does the context effect enter the target word? — {slug}",
                 x=0.01, ha="left", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "P2_patching.png", dpi=200)
    plt.close(fig)

    # P3: heads heatmap (sequential single hue)
    if len(agg3):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", C_ORANGE, "#8a2e0c"])
        piv = agg3.pivot(index="layer", columns="head", values="mean")
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(piv) + 1.6))
        vmax = max(1e-6, float(np.nanmax(np.abs(piv.values))))
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([f"L{L}" for L in piv.index])
        ax.set_xlabel("Head")
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label("Δ variety score (MSA targets)")
        ax.set_title(f"P3. Heads that carry the EGY context into an MSA word — {slug}", **tkw)
        fig.tight_layout()
        fig.savefig(out / "P3_heads.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
