from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import torch
LINCE_DIR=Path('/Users/fahad/Documents/Research-Projects/New-Paper/data/raw/lince_msaea/lid_msaea')
def load_lince():
    rows = []
    for split in ['train', 'validation']:
        df = pd.read_parquet(LINCE_DIR / f'{split}.parquet')
        for ws, ls in zip(df.words, df.lid):
            rows.append((list(ws), list(ls)))
    return rows

def type_classes(rows, purity=0.95, min_count=5):
    cnt = defaultdict(Counter)
    for ws, ls in rows:
        for w, l in zip(ws, ls):
            if l in ('lang1', 'lang2'):
                cnt[w][l] += 1
    msa, egy = ([], [])
    for w, c in cnt.items():
        n = c['lang1'] + c['lang2']
        if n < min_count or w.startswith(('#', '@', 'http')) or len(w) < 2:
            continue
        share = c['lang2'] / n
        if share <= 1 - purity:
            msa.append(w)
        elif share >= purity:
            egy.append(w)
    return (sorted(msa), sorted(egy))

def mono_sentences(rows, variety):
    lab, other = ('lang1', 'lang2') if variety == 'MSA' else ('lang2', 'lang1')
    return [(ws, ls) for ws, ls in rows if lab in ls and other not in ls]

def prefix_pool(tok, sents, K):
    """Word-boundary prefixes of monolingual tweets that tokenize to exactly K tokens."""
    pool = []
    for ws, _ in sents:
        ws = [w for w in ws if not w.startswith(('http', '@'))]
        for j in range(1, len(ws) + 1):
            ids = tok(' '.join(ws[:j]), add_special_tokens=False).input_ids
            if len(ids) == K:
                pool.append(ids)
                break
            if len(ids) > K:
                break
    return pool

def word_ids(tok, w):
    """Token ids of ' w' as it appears after a preceding word (works for BPE and SentencePiece)."""
    anchor = 'و'
    a = tok(anchor, add_special_tokens=False).input_ids
    full = tok(anchor + ' ' + w, add_special_tokens=False).input_ids
    if full[:len(a)] == a and len(full) > len(a):
        return full[len(a):]
    return tok(' ' + w, add_special_tokens=False).input_ids

def with_bos(tok, ids):
    bos = tok.bos_token_id
    add_bos = bos is not None and tok('a').input_ids[:1] == [bos]
    return ([bos] if add_bos else []) + ids

def get_layers(model):
    return model.model.layers

def variety_directions(model, tok, rows, device, n_per=400, seed=0):
    """Mean last-subword hidden state of MSA / EGY words in monolingual tweets, per layer."""
    rng = np.random.default_rng(seed)
    sums, counts = ({}, {})
    for var, lab in [('MSA', 'lang1'), ('EGY', 'lang2')]:
        sents = mono_sentences(rows, var)
        pick = rng.choice(len(sents), min(n_per, len(sents)), replace=False)
        acc, n = (None, 0)
        for i in pick:
            ws, ls = sents[i]
            text, spans = ('', [])
            for w in ws:
                if text:
                    text += ' '
                spans.append((len(text), len(text) + len(w)))
                text += w
            enc = tok(text, return_tensors='pt', return_offsets_mapping=True, truncation=True, max_length=128)
            offs = enc.pop('offset_mapping')[0].tolist()
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
            h = torch.stack([x[0, pos].float().sum(0) for x in hs]).cpu()
            acc = h if acc is None else acc + h
            n += len(pos)
        sums[var], counts[var] = (acc, n)
    mu_msa = sums['MSA'] / counts['MSA']
    mu_egy = sums['EGY'] / counts['EGY']
    return (mu_msa, mu_egy)

def make_scorer(mu_msa, mu_egy, device):
    d = mu_egy - mu_msa
    denom = (d * d).sum(-1)
    mu_msa, d, denom = (mu_msa.to(device), d.to(device), denom.to(device))

    def score(h, L):
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
    cache = {'resid': {}, 'attn': {}, 'mlp': {}, 'heads': {}}
    hooks = []
    for L, layer in enumerate(layers):

        def pre(mod, args, kwargs, L=L):
            h = args[0] if args else kwargs['hidden_states']
            cache['resid'][L] = h[0, -1].detach().clone()

        def attn_hook(mod, args, out, L=L):
            o = out[0] if isinstance(out, tuple) else out
            cache['attn'][L] = o[0, -1].detach().clone()

        def mlp_hook(mod, args, out, L=L):
            cache['mlp'][L] = out[0, -1].detach().clone()

        def oproj_pre(mod, args, L=L):
            cache['heads'][L] = args[0][0, -1].detach().clone()
        hooks += [layer.register_forward_pre_hook(pre, with_kwargs=True), layer.self_attn.register_forward_hook(attn_hook), layer.mlp.register_forward_hook(mlp_hook), layer.self_attn.o_proj.register_forward_pre_hook(oproj_pre)]
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
        if component == 'resid':

            def pre(mod, args, kwargs, items=items, L=L):
                if args:
                    h = args[0].clone()
                    for b, _ in items:
                        h[b, -1] = cache['resid'][L]
                    return ((h,) + tuple(args[1:]), kwargs)
                h = kwargs['hidden_states'].clone()
                for b, _ in items:
                    h[b, -1] = cache['resid'][L]
                kwargs['hidden_states'] = h
                return (args, kwargs)
            hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        elif component == 'attn':

            def fwd(mod, args, out, items=items, L=L):
                o = (out[0] if isinstance(out, tuple) else out).clone()
                for b, _ in items:
                    o[b, -1] = cache['attn'][L]
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            hooks.append(layer.self_attn.register_forward_hook(fwd))
        elif component == 'mlp':

            def fwd(mod, args, out, items=items, L=L):
                o = out.clone()
                for b, _ in items:
                    o[b, -1] = cache['mlp'][L]
                return o
            hooks.append(layer.mlp.register_forward_hook(fwd))
        elif component == 'head':

            def pre(mod, args, items=items, L=L):
                x = args[0].clone()
                for b, (_, hd) in items:
                    sl = slice(hd * head_dim, (hd + 1) * head_dim)
                    x[b, -1, sl] = cache['heads'][L][sl]
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
