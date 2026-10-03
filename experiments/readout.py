"""
Shared read-out utilities for the revision experiments (exp06, exp07).

* ONE saved read-out probe per model (results/probes/<model>/probe_L<R>.pt), trained on
  words from monolingual LinCE tweets only, reused by every later script, so all flip
  rates and logits come from the same probe (fixes the exp04 vs exp05 discrepancy).
* Type purity computed from MONOLINGUAL tweets only (no code-switched labels), so the
  selection of "unambiguous" words is independent of the items being evaluated.
* Head tools: o_proj-input capture, mean/zero ablation at chosen positions, per-head
  output norms (for norm-matched random controls) and direct attribution onto the probe
  (with the Gemma-2 post-attention RMSNorm handled exactly).
"""
from collections import Counter, defaultdict

import numpy as np
import torch

from exp02_causal_patching import PROJECT_DIR, get_layers, mono_sentences, type_classes, with_bos, word_ids
from exp03_natural_vs_controlled import LANG, load_rows_with_ids

__all__ = ["PROJECT_DIR", "LANG", "load_rows_with_ids", "get_layers", "with_bos", "word_ids", "mono_sentences",
           "type_classes", "type_classes_mono", "word_positions", "get_probe", "head_geometry",
           "collect_head_inputs", "AblationHooks", "dla_matrix", "boot", "boot_diff", "readout_layer"]


def readout_layer(n_layers):
    return int(round(0.8 * n_layers))


# ----------------------------------------------------------------------------- data
def type_classes_mono(rows, purity=0.95, min_count=5):
    """Unambiguous types from MONOLINGUAL tweets only: share of a type's uses that occur
    in EGY-only tweets (labelled lang2) vs MSA-only tweets (labelled lang1)."""
    cnt = defaultdict(Counter)
    for ws, ls in rows:
        labs = set(ls)
        if ("lang1" in labs) == ("lang2" in labs):      # skip code-switched and label-less tweets
            continue
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


def word_positions(tok, ws, ls, lab, max_length=128):
    """input_ids of the tweet and the last-subword position of every word labelled `lab`
    (returns positions aligned with word indices through the dict `last`)."""
    text, spans = "", []
    for wd in ws:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(wd)))
        text += wd
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    c2w = np.full(len(text) + 1, -1)
    for wi, (a, e) in enumerate(spans):
        c2w[a:e] = wi
    last = {}
    for ti, (a, e) in enumerate(enc["offset_mapping"]):
        if e > a and c2w[e - 1] >= 0:
            last[int(c2w[e - 1])] = ti
    pos = [last[wi] for wi, l in enumerate(ls) if l == lab and wi in last]
    return enc["input_ids"], pos, last


# ----------------------------------------------------------------------------- probe
def _vectors_at_layer(model, tok, sents, lab, device, max_words, L):
    feats, n, used = [], 0, 0
    for ws, ls in sents:
        used += 1
        ids, pos, _ = word_positions(tok, ws, ls, lab)
        if not pos:
            continue
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
        feats.append(hs[L][0, pos].float().clamp(-6e4, 6e4).cpu().numpy())
        n += len(pos)
        if n >= max_words:
            break
    return np.concatenate(feats, 0), used


def get_probe(model, tok, rows, slug, R, device, probe_words=6000, seed=13, refit=False):
    """Standardised logistic probe at layer R (positive logit = EGY). Cached on disk.
    Returns dict(w [d] float32 tensor on device, b float, stats, n_used={MSA,EGY})."""
    path = PROJECT_DIR / "results" / "probes" / slug / f"probe_L{R}_seed{seed}.pt"
    if path.exists() and not refit:
        p = torch.load(path, weights_only=False)
        p["w"] = p["w"].to(device)
        print(f"Loaded saved probe {path.relative_to(PROJECT_DIR)}  (held-out acc {p['stats']['heldout_acc']:.3f})")
        return p
    from sklearn.linear_model import LogisticRegression
    print(f"Training read-out probe at layer {R} ...")
    rng = np.random.default_rng(seed)
    mono, X, used = {}, {}, {}
    for v, lab in [("MSA", "lang1"), ("EGY", "lang2")]:
        s = mono_sentences(rows, v)
        mono[v] = [s[i] for i in rng.permutation(len(s))]
        X[v], used[v] = _vectors_at_layer(model, tok, mono[v], lab, device, probe_words, R)
    cut = {v: int(0.8 * len(X[v])) for v in X}
    Xtr = np.concatenate([X["MSA"][:cut["MSA"]], X["EGY"][:cut["EGY"]]])
    ytr = np.r_[np.zeros(cut["MSA"]), np.ones(cut["EGY"])]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit((Xtr - mu) / sd, ytr)
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    zm = X["MSA"][cut["MSA"]:] @ w + b
    ze = X["EGY"][cut["EGY"]:] @ w + b
    stats = {"heldout_acc": float(((zm < 0).sum() + (ze > 0).sum()) / (len(zm) + len(ze))),
             "msa_own_mean": float((-zm).mean()), "msa_own_sd": float(zm.std()),
             "egy_own_mean": float(ze.mean()), "egy_own_sd": float(ze.std()),
             "n_train": {"MSA": int(cut["MSA"]), "EGY": int(cut["EGY"])}}
    p = {"w": torch.tensor(w, dtype=torch.float32), "b": b, "layer": R, "seed": seed,
         "stats": stats, "n_used_tweets": used, "probe_words": probe_words}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(p, path)
    print(f"  saved {path.relative_to(PROJECT_DIR)}  (held-out acc {stats['heldout_acc']:.3f})")
    p["w"] = p["w"].to(device)
    return p


def probe_split(rows, n_used, seed=13):
    """Monolingual tweets NOT used to train the probe (same permutation as get_probe)."""
    rng = np.random.default_rng(seed)
    out = {}
    for v in ["MSA", "EGY"]:
        s = mono_sentences(rows, v)
        perm = [s[i] for i in rng.permutation(len(s))]
        out[v] = (perm[:n_used[v]], perm[n_used[v]:])
    return out


# ----------------------------------------------------------------------------- heads
def head_geometry(model):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_heads, head_dim


def collect_head_inputs(model, ids, positions, layers_needed, device):
    """o_proj inputs (all heads concatenated) at `positions`, for each layer in layers_needed."""
    layers = get_layers(model)
    cap = {}
    hooks = [layers[L].self_attn.o_proj.register_forward_pre_hook(
        lambda mod, a, L=L: cap.__setitem__(L, a[0][0, positions].float().cpu())) for L in layers_needed]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device))
    finally:
        for hk in hooks:
            hk.remove()
    return cap


class AblationHooks:
    """Per-batch-element head ablation at one position per element.

    plan: list over batch elements of (position, [(L, h), ...]); ref_mean: {L: tensor[H*hd]} or None (zero).
    """

    def __init__(self, model, plan, head_dim, ref_mean=None):
        self.model, self.plan, self.hd, self.ref = model, plan, head_dim, ref_mean
        self.hooks = []

    def __enter__(self):
        layers = get_layers(self.model)
        by_layer = defaultdict(list)
        for b, (pos, heads) in enumerate(self.plan):
            for L, h in heads:
                by_layer[L].append((b, pos, h))
        for L, todo in by_layer.items():
            def pre(mod, a, L=L, todo=todo):
                x = a[0].clone()
                for b, pos, h in todo:
                    sl = slice(h * self.hd, (h + 1) * self.hd)
                    x[b, pos, sl] = 0 if self.ref is None else self.ref[L][sl].to(x.dtype).to(x.device)
                return (x,) + tuple(a[1:])
            self.hooks.append(layers[L].self_attn.o_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for hk in self.hooks:
            hk.remove()


def dla_matrix(model, L, h, x, head_dim, w_cpu):
    """Direct attribution of head (L,h) onto the probe direction for o_proj inputs x [n, H*hd].
    Gemma-2 family: the attention output passes through post_attention_layernorm before the
    residual add, so the contribution is (W_o,h x_h) * (1+g) / rms(full attention output)."""
    layer = get_layers(model)[L]
    with torch.no_grad():
        sl = slice(h * head_dim, (h + 1) * head_dim)
        Wo = layer.self_attn.o_proj.weight.float().cpu()
        contrib = x[:, sl] @ Wo[:, sl].T
        if hasattr(layer, "pre_feedforward_layernorm"):
            norm = layer.post_attention_layernorm
            full = x @ Wo.T
            rms = torch.sqrt(full.pow(2).mean(-1, keepdim=True) + norm.eps)
            contrib = contrib * (1.0 + norm.weight.float().cpu()) / rms
        return (contrib @ w_cpu).numpy()


def head_output_norms(model, x, L, n_heads, head_dim):
    """Mean ||W_o,h x_h|| per head for o_proj inputs x [n, H*hd] of layer L."""
    Wo = get_layers(model)[L].self_attn.o_proj.weight.float().cpu()
    out = []
    with torch.no_grad():
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            out.append(float((x[:, sl] @ Wo[:, sl].T).norm(dim=-1).mean()))
    return np.array(out)


# ----------------------------------------------------------------------------- stats
def boot(x, n=2000, seed=0, groups=None):
    """Mean and 95% bootstrap CI. If `groups` is given, resample clusters (e.g. tweets)."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    x = x[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if groups is None:
        m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    else:
        g = np.asarray(groups)[ok]
        uniq, inv = np.unique(g, return_inverse=True)
        sums, cnts = np.bincount(inv, x), np.bincount(inv)
        pick = rng.integers(0, len(uniq), (n, len(uniq)))
        m = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, n=5000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
