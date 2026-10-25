def original_direction_selection_no_inference(model, tok, rows, device, n_per=400, seed=0):
    rng = np.random.default_rng(seed)
    recorded = []
    for var, lab in [('MSA', 'lang1'), ('EGY', 'lang2')]:
        sents = mono_sentences(rows, var)
        pick = rng.choice(len(sents), min(n_per, len(sents)), replace=False)
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
            recorded.append({'variety': var, 'mono_index': int(i), 'encoded': {k: v[0].tolist() for k, v in enc.items()}, 'positions': pos, 'text': text, 'n_words': len(pos)})
    return recorded
