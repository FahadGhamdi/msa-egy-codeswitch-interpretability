"""
Experiment 11 (review): is the identity of an embedded word still linearly RECOVERABLE, or does the
monolingual probe simply read the context?

The saved read-out probe was trained on monolingual tweets, where a word's variety and its context's
variety always coincide. Here we train probes on code-switched tweets in which the two are decorrelated:

  balanced probe : trained on ALL labelled words of code-switched training tweets (matrix words and
                   embedded words), with sample weights that give the four cells
                   {MSA, EGY word} x {MSA, EGY matrix} equal total weight, so the matrix variety carries
                   no information about the label. Evaluated on embedded words of held-out tweets.
  matrix-only    : trained only on matrix-variety words of the same tweets (label = context variety,
                   as in monolingual training). Evaluated on the same embedded words.
  control task   : balanced probe trained on labels randomly re-assigned per word TYPE (Hewitt & Liang,
                   2019); selectivity = accuracy - control accuracy (types seen in training).
  saved probe    : the monolingual read-out probe of the paper, on the same test items, with the flip
                   rate at alternative thresholds (sensitivity to the decision threshold).

Folds are formed by tweet (GroupKFold, 5 folds), so no tweet contributes to training and test.
Accuracies are reported for all test items and for test items whose word type never occurs in the
training folds; intervals are tweet-cluster bootstrap intervals.

    python exp11_decodability.py --model google/gemma-2-9b
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from readout import LANG, PROJECT_DIR, boot, get_layers, get_probe, load_rows_with_ids, readout_layer, \
    type_classes_mono, word_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--min_left_tokens", type=int, default=3)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / "exp11_decodability" / slug
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    T0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    R = readout_layer(len(get_layers(model)))
    rows = load_rows_with_ids()
    plain = [(ws, ls) for _, ws, ls in rows]
    probe = get_probe(model, tok, plain, slug, R, device)
    w_saved, b_saved = probe["w"].float().cpu().numpy(), probe["b"]
    m, e = type_classes_mono(plain)
    pure = {"MSA": set(m), "EGY": set(e)}

    # ------------------------------------------------------------------ features of code-switched tweets
    X, y, mat, emb, typ, sid = [], [], [], [], [], []
    for s, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs or langs.count("EGY") == langs.count("MSA"):
            continue
        matrix = "EGY" if langs.count("EGY") > langs.count("MSA") else "MSA"
        ids, _, last = word_positions(tok, ws, ls, "lang1")
        keep = [(wi, last[wi]) for wi, l in enumerate(ls)
                if l in LANG and wi in last and last[wi] >= args.min_left_tokens and ws[wi] in pure[LANG[l]]]
        if not keep:
            continue
        with torch.no_grad():
            h = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states[R][0]
        H = h[[p for _, p in keep]].float().clamp(-6e4, 6e4).cpu().numpy()
        for (wi, _), vec in zip(keep, H):
            v = LANG[ls[wi]]
            X.append(vec); y.append(int(v == "EGY")); mat.append(int(matrix == "EGY"))
            emb.append(int(v != matrix)); typ.append(ws[wi]); sid.append(s)
    X, y, mat, emb = np.array(X), np.array(y), np.array(mat), np.array(emb)
    typ, sid = np.array(typ), np.array(sid)
    print(f"{len(y)} words from {len(set(sid))} code-switched tweets; embedded: {emb.sum()} "
          f"(MSA-in-EGY {((emb == 1) & (y == 0)).sum()}, EGY-in-MSA {((emb == 1) & (y == 1)).sum()})")

    def cell_weights(yy, mm):
        wts = np.zeros(len(yy))
        for a in (0, 1):
            for c in (0, 1):
                sel = (yy == a) & (mm == c)
                if sel.any():
                    wts[sel] = 1.0 / sel.sum()
        return wts * len(yy) / wts.sum()

    def fit(Xtr, ytr, wts):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        clf = LogisticRegression(C=args.C, max_iter=3000, random_state=args.seed)
        clf.fit((Xtr - mu) / sd, ytr, sample_weight=wts)
        return lambda Z: clf.decision_function((Z - mu) / sd)

    rng = np.random.default_rng(args.seed)
    types_u = np.unique(typ)
    ctrl_map = dict(zip(types_u, rng.integers(0, 2, len(types_u))))
    y_ctrl = np.array([ctrl_map[t] for t in typ])

    pred = {k: np.full(len(y), np.nan) for k in ["balanced", "matrix_only", "control"]}
    unseen = np.zeros(len(y), bool)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=sid):
        f_bal = fit(X[tr], y[tr], cell_weights(y[tr], mat[tr]))
        mo = tr[emb[tr] == 0]
        f_mat = fit(X[mo], y[mo], np.ones(len(mo)))
        f_ctl = fit(X[tr], y_ctrl[tr], cell_weights(y_ctrl[tr], mat[tr]))
        pred["balanced"][te], pred["matrix_only"][te], pred["control"][te] = f_bal(X[te]), f_mat(X[te]), f_ctl(X[te])
        unseen[te] = ~np.isin(typ[te], np.unique(typ[tr]))
    pred["saved_mono"] = X @ w_saved + b_saved

    res = {"model": args.model, "readout_layer": R, "n_words": int(len(y)), "n_tweets": int(len(set(sid))),
           "results": {}}
    for g, sel in [("MSA_in_EGY", (emb == 1) & (y == 0)), ("EGY_in_MSA", (emb == 1) & (y == 1)),
                   ("matrix_MSA", (emb == 0) & (y == 0)), ("matrix_EGY", (emb == 0) & (y == 1))]:
        r = {"n": int(sel.sum()), "n_tweets": int(len(set(sid[sel]))), "n_unseen_type": int((sel & unseen).sum())}
        for k in pred:
            lab = y_ctrl if k == "control" else y
            correct = ((pred[k] > 0).astype(int) == lab).astype(float)
            r[f"acc_{k}"] = boot(correct[sel], groups=sid[sel])
            if k != "control" and (sel & unseen).sum() >= 10:
                r[f"acc_{k}_unseen_types"] = boot(correct[sel & unseen], groups=sid[sel & unseen])
        seen = sel & ~unseen
        cb = ((pred["balanced"] > 0).astype(int) == y).astype(float)
        cc = ((pred["control"] > 0).astype(int) == y_ctrl).astype(float)
        r["selectivity_seen_types"] = float(cb[seen].mean() - cc[seen].mean()) if seen.any() else None
        own = np.where(y[sel] == 1, pred["saved_mono"][sel], -pred["saved_mono"][sel])
        r["saved_probe_flip_by_threshold"] = {str(tau): float((own < tau).mean()) for tau in [-2, -1, 0, 1, 2]}
        res["results"][g] = r
    res["minutes"] = round((time.time() - T0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    for g, r in res["results"].items():
        print(f"{g:11s} n={r['n']:5d}  balanced {r['acc_balanced'][0]:.3f}  matrix-only {r['acc_matrix_only'][0]:.3f}"
              f"  saved-mono {r['acc_saved_mono'][0]:.3f}  control {r['acc_control'][0]:.3f}"
              f"  unseen-type balanced {r.get('acc_balanced_unseen_types', [np.nan])[0]:.3f}")
    print(f"\nResults in {out}  ({res['minutes']} min)")


if __name__ == "__main__":
    main()
