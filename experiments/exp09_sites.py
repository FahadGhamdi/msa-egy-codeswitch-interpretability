"""
Experiment 09 - Intervention sites with EQUAL numbers of ablated positions (review, RQ3).

exp06 compared ablation at the embedded word (one position) with ablation at ALL later positions.
That confounds the site with the number of ablated positions, and the first continuation token is
predicted AT the embedded word, so a later-site ablation cannot reach it. This experiment adds:

  sites   : target (position t)  |  next1 (position t+1 only)  |  later (t+1 .. end)
            each with the top-5 heads and with layer- and norm-matched random five-head draws
  measures: as in exp06, plus
            cont_lp_matrix_x1  : following matrix-variety words, EXCLUDING the first continuation token
                                 (the only token that a later-site ablation cannot affect)
            cont_lp_matrix_w{j}: following matrix-variety word j (j = 1..3) separately (distance profile)
  paired  : item-level net contrasts next1 - target and later - target, tweet-cluster bootstrap p,
            Holm across runs is applied in the paper.

Items, probe, heads, mean-ablation reference and random-draw procedure are identical to exp06.

Usage
    python exp09_sites.py --model google/gemma-2-9b
    python exp09_sites.py --model QCRI/Fanar-1-9B --heads_model gemma-2-9b
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from readout import (LANG, PROJECT_DIR, AblationHooks, boot, boot_diff, collect_head_inputs, dla_matrix, get_layers,
                     get_probe, head_geometry, head_output_norms, load_rows_with_ids, probe_split, readout_layer,
                     type_classes, type_classes_mono, with_bos, word_ids, word_positions)
from exp02_causal_patching import style

C_BLUE, C_ORANGE, C_GREY, C_TEXT, C_MUTED = "#2a78d6", "#eb6834", "#b9b8b3", "#0b0b0b", "#52514e"
GROUPS = ["MSA_in_EGY", "EGY_in_MSA", "mono_MSA", "mono_EGY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--heads_model", default=None, help="take the top heads from another model's exp02 (slug)")
    ap.add_argument("--heads_from", default="exp02", choices=["exp02", "exp02_reverse"])
    ap.add_argument("--purity_source", default="mono", choices=["mono", "all"])
    ap.add_argument("--max_items", type=int, default=300, help="per embedded variety")
    ap.add_argument("--mono_items", type=int, default=150, help="control words per variety")
    ap.add_argument("--n_next", type=int, default=3, help="following words scored")
    ap.add_argument("--min_next", type=int, default=1, help="0 = also keep words with no labelled word after them")
    ap.add_argument("--span_draws", type=int, default=0, help="random draws for the span (target..end) ablation")
    ap.add_argument("--later_draws", type=int, default=100, help="random draws for the later-only (target+1..end) ablation")
    ap.add_argument("--n_pref", type=int, default=50, help="max tokens per variety in the next-token preference sets")
    ap.add_argument("--random_draws", type=int, default=100)
    ap.add_argument("--ref_tweets", type=int, default=300)
    ap.add_argument("--probe_words", type=int, default=6000)
    ap.add_argument("--max_K", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    heads_slug = args.heads_model or slug
    name = slug if heads_slug == slug else f"{slug}__heads-{heads_slug}"
    if args.purity_source != "mono":
        name += f"__purity-{args.purity_source}"
    if args.min_next != 1:
        name += f"__minnext{args.min_next}"
    out = PROJECT_DIR / "results" / "exp09_sites" / name
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    n_layers = len(get_layers(model))
    n_heads, head_dim = head_geometry(model)
    R = readout_layer(n_layers)

    s2 = json.loads((PROJECT_DIR / "results" / args.heads_from / heads_slug / "summary.json").read_text())
    top = [(int(h["layer"]), int(h["head"])) for h in s2["top_heads"]][:5]
    head_layers = sorted({L for L, _ in top})
    print(f"top heads ({heads_slug}): {['L%dH%d' % t for t in top]}")

    rows = load_rows_with_ids()
    plain = [(ws, ls) for _, ws, ls in rows]
    m, e = type_classes_mono(plain) if args.purity_source == "mono" else type_classes(plain)
    types = {"MSA": set(m), "EGY": set(e)}
    print(f"unambiguous types ({args.purity_source}): MSA={len(m)} EGY={len(e)}")

    probe = get_probe(model, tok, plain, slug, R, device, args.probe_words, args.seed)
    w, b = probe["w"], probe["b"]
    sd_own = {"MSA": probe["stats"]["msa_own_sd"], "EGY": probe["stats"]["egy_own_sd"]}
    split = probe_split(plain, probe["n_used_tweets"], args.seed)

    # first-token sets for the next-token dialect preference
    # (words whose first token is a bare space piece, e.g. SentencePiece "▁", are skipped: uninformative)
    # Sets are BALANCED: the same number of tokens per variety, the most frequent ones.
    from collections import Counter
    wfreq = Counter(wd for ws, _ in plain for wd in ws)
    first = {v: Counter() for v in types}
    for v in types:
        for t_ in types[v]:
            wi = word_ids(tok, t_)
            if wi and tok.decode([wi[0]]).strip():
                first[v][wi[0]] += wfreq[t_]
    shared = set(first["MSA"]) & set(first["EGY"])
    ranked = {v: [x for x, _ in first[v].most_common() if x not in shared] for v in first}
    n_pref = min(args.n_pref, len(ranked["MSA"]), len(ranked["EGY"]))
    first_ids = {v: torch.tensor(ranked[v][:n_pref], device=device, dtype=torch.long) for v in ranked}
    print(f"next-token preference sets: {n_pref} tokens per variety "
          f"(distinctive available: MSA={len(ranked['MSA'])} EGY={len(ranked['EGY'])}; shared removed: {len(shared)})")
    has_D = len(first_ids["MSA"]) >= 5 and len(first_ids["EGY"]) >= 5
    if not has_D:
        print("  WARNING: too few distinctive first tokens - next-token preference D is not computed (NaN)")

    # ------------------------------------------------------------------ items
    def build(ws, ls, i, own, matrix, sid, group):
        left = tok(" ".join(ws[:i]), add_special_tokens=False).input_ids if i > 0 else []
        wid = word_ids(tok, ws[i])
        if not wid or len(left) > args.max_K:
            return None
        ids = with_bos(tok, left + wid)
        t = len(ids) - 1
        cont = []
        for j in range(i + 1, min(len(ws), i + 1 + args.n_next)):
            cid = word_ids(tok, ws[j])
            if not cid:
                continue
            cont.append((len(ids), len(ids) + len(cid), LANG.get(ls[j])))
            ids = ids + cid
        if sum(1 for c in cont if c[2]) < args.min_next:
            return None
        return {"sid": sid, "group": group, "word": ws[i], "own": own, "matrix": matrix, "ids": ids, "t": t,
                "cont": cont}

    items = {g: [] for g in GROUPS}
    for sid, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs or langs.count("EGY") == langs.count("MSA"):
            continue
        matrix = "EGY" if langs.count("EGY") > langs.count("MSA") else "MSA"
        for i, (wd, l) in enumerate(zip(ws, ls)):
            if l not in LANG or LANG[l] == matrix or i == 0 or wd not in types[LANG[l]]:
                continue
            if [LANG[x] for x in ls[:i] if x in LANG].count(matrix) < 3:
                continue
            it = build(ws, ls, i, LANG[l], matrix, sid, f"{LANG[l]}_in_{matrix}")
            if it:
                items[it["group"]].append(it)
    for v, lab in [("MSA", "lang1"), ("EGY", "lang2")]:
        cands = []
        for k, (ws, ls) in enumerate(split[v][1]):
            for i, (wd, l) in enumerate(zip(ws, ls)):
                if l == lab and i >= 3 and wd in types[v]:
                    it = build(ws, ls, i, v, v, f"mono{v}-{k}", f"mono_{v}")
                    if it:
                        cands.append(it)
                        break                               # one word per tweet
        items[f"mono_{v}"] = cands
    for g in items:
        cap = args.max_items if "_in_" in g else args.mono_items
        if len(items[g]) > cap:
            keep = rng.choice(len(items[g]), cap, replace=False)
            items[g] = [items[g][i] for i in sorted(keep)]
    print("items: " + "  ".join(f"{g}={len(items[g])}" for g in GROUPS))

    # ------------------------------------------------------------------ head means, norms
    print("Collecting head outputs on monolingual words ...")
    ref = {v: {L: [] for L in head_layers} for v in ["MSA", "EGY"]}
    for v, lab in [("MSA", "lang1"), ("EGY", "lang2")]:
        for ws, ls in split[v][0][: args.ref_tweets]:
            ids, pos, _ = word_positions(tok, ws, ls, lab)
            if pos:
                cap = collect_head_inputs(model, ids, pos, head_layers, device)
                for L in head_layers:
                    ref[v][L].append(cap[L])
    ref = {v: {L: torch.cat(ref[v][L]) for L in head_layers} for v in ref}
    ref_mean = {L: 0.5 * (ref["MSA"][L].mean(0) + ref["EGY"][L].mean(0)).to(device) for L in head_layers}
    norms = {L: head_output_norms(model, torch.cat([ref["MSA"][L], ref["EGY"][L]]), L, n_heads, head_dim)
             for L in head_layers}

    def random_draw():
        pick = []
        for L, h in top:
            ratio = norms[L] / max(norms[L][h], 1e-8)
            pool = [g for g in range(n_heads) if (L, g) not in top and (L, g) not in pick and 0.5 <= ratio[g] <= 2.0]
            if len(pool) < 2:
                pool = [g for g in range(n_heads) if (L, g) not in top and (L, g) not in pick]
            pick.append((L, int(rng.choice(pool))))
        return pick

    # (name, heads, where): where = "target" -> the embedded word's last sub-token only (source);
    #                               "later"  -> every position AFTER it only;
    #                               "span"   -> the embedded word and every later position.
    # source vs later vs span separates "a signal written at the switch point and carried forward"
    # from "the heads' local role at the later positions themselves".
    conditions = [("baseline", [], "target"), ("top5", top, "target")]
    conditions += [(f"random5_d{d}", random_draw(), "target") for d in range(args.random_draws)]
    conditions += [("top5_next1", top, "next1")]
    conditions += [(f"random5_next1_d{d}", random_draw(), "next1") for d in range(args.later_draws)]
    conditions += [("top5_later", top, "later")]
    conditions += [(f"random5_later_d{d}", random_draw(), "later") for d in range(args.later_draws)]
    (out / "conditions.json").write_text(json.dumps(
        {c: {"heads": [f"L{L}H{h}" for L, h in hs], "where": wh} for c, hs, wh in conditions}, indent=1))
    WHERE = {"target": lambda t: t, "next1": lambda t: slice(t + 1, t + 2), "later": lambda t: slice(t + 1, None),
             "span": lambda t: slice(t, None)}

    # ------------------------------------------------------------------ run
    def run_item(it):
        ids, t = it["ids"], it["t"]
        k = len(ids) - t
        rec = []
        for c0 in range(0, len(conditions), args.batch):
            chunk = conditions[c0:c0 + args.batch]
            plan = [(WHERE[wh](t), hs) for _, hs, wh in chunk]
            x = torch.tensor([ids] * len(chunk), device=device)
            with AblationHooks(model, plan, head_dim, ref_mean), torch.no_grad():
                o = model(input_ids=x, output_hidden_states=True, logits_to_keep=k)
            z = ((o.hidden_states[R][:, t].float() * w).sum(-1) + b).cpu().numpy()          # EGY-positive
            lp = torch.log_softmax(o.logits.float(), -1)                                      # [B, k, V], pos t..end
            if has_D:
                D = (torch.logsumexp(lp[:, 0, first_ids["EGY"]], -1)
                     - torch.logsumexp(lp[:, 0, first_ids["MSA"]], -1)).cpu().numpy()
            else:
                D = np.full(len(chunk), np.nan)
            tgt = torch.tensor(ids[t + 1:], device=device)
            tok_lp = lp[:, :-1].gather(-1, tgt[None, :, None].expand(len(chunk), -1, 1))[..., 0].cpu().numpy()
            for bi, (cname, _, _) in enumerate(chunk):
                r = {"condition": cname,
                     "own_logit": float(z[bi] if it["own"] == "EGY" else -z[bi]),
                     "D_to_matrix": float(D[bi] if it["matrix"] == "EGY" else -D[bi])}
                for role in ["matrix", "own"]:
                    lab_needed = it["matrix"] if role == "matrix" else it["own"]
                    vals = [tok_lp[bi, s - t - 1:e - t - 1] for s, e, lab in it["cont"] if lab == lab_needed]
                    r[f"cont_lp_{role}"] = float(np.concatenate(vals).mean()) if vals else np.nan
                vals = [tok_lp[bi, s - t - 1:e - t - 1] for s, e, lab in it["cont"] if lab]
                r["cont_lp_all"] = float(np.concatenate(vals).mean()) if vals else np.nan
                # excluding the first continuation token (predicted at the embedded word itself)
                vals = [tok_lp[bi, max(s, t + 2) - t - 1:e - t - 1] for s, e, lab in it["cont"]
                        if lab == it["matrix"] and e > t + 2]
                r["cont_lp_matrix_x1"] = float(np.concatenate(vals).mean()) if vals else np.nan
                for j, (s, e, lab) in enumerate(it["cont"]):
                    r[f"cont_lp_matrix_w{j + 1}"] = (float(tok_lp[bi, s - t - 1:e - t - 1].mean())
                                                     if lab == it["matrix"] else np.nan)
                rec.append(r)
        return rec

    recs, dla_caps = [], {g: {L: [] for L in head_layers} for g in GROUPS}
    all_items = [it for g in GROUPS for it in items[g]]
    for n, it in enumerate(all_items):
        for r in run_item(it):
            recs.append({"item": n, "sid": it["sid"], "group": it["group"], "word": it["word"], **r})
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(all_items)} ({time.time()-T0:.0f}s)")
    df = pd.DataFrame(recs)
    base = df[df.condition == "baseline"].set_index("item")
    for j in range(1, args.n_next + 1):
        if f"cont_lp_matrix_w{j}" not in df:
            df[f"cont_lp_matrix_w{j}"] = np.nan
    for col in ["own_logit", "D_to_matrix", "cont_lp_matrix", "cont_lp_own", "cont_lp_all", "cont_lp_matrix_x1"] + \
            [f"cont_lp_matrix_w{j}" for j in range(1, args.n_next + 1)]:
        df["d_" + col] = df[col] - df.item.map(base[col])
    df["flipped"] = (df.own_logit < 0).astype(float)
    df["cls"] = df.condition.str.replace(r"_d\d+$", "", regex=True)
    df.to_csv(out / "B_long.csv", index=False)

    # ------------------------------------------------------------------ summaries
    def boot_p(x, groups, n=5000, seed=0):
        x = np.asarray(x, float)
        ok = ~np.isnan(x)
        x, g = x[ok], np.asarray(groups)[ok]
        if len(x) < 3:
            return [np.nan] * 4 + [int(len(x))]
        uniq, inv = np.unique(g, return_inverse=True)
        sums, cnts = np.bincount(inv, x), np.bincount(inv)
        r_ = np.random.default_rng(seed)
        pick = r_.integers(0, len(uniq), (n, len(uniq)))
        m = sums[pick].sum(1) / cnts[pick].sum(1)
        p = 2 * min((m <= 0).mean(), (m >= 0).mean())
        return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(max(p, 1 / n)),
                int(len(x))]

    measures = ["d_own_logit", "d_D_to_matrix", "d_cont_lp_matrix", "d_cont_lp_matrix_x1", "d_cont_lp_own",
                "d_cont_lp_all"] + [f"d_cont_lp_matrix_w{j}" for j in range(1, args.n_next + 1)]
    sites = [("target", "top5", "random5"), ("next1", "top5_next1", "random5_next1"),
             ("later", "top5_later", "random5_later")]
    net = {}
    rows_out = []
    for g in ["MSA_in_EGY", "EGY_in_MSA", "mono_MSA", "mono_EGY"]:
        dg = df[df.group == g]
        if dg.empty:
            continue
        sid = dg.groupby("item").sid.first()
        for mtr in measures:
            c = {}
            for site, tc, rc in sites:
                ti = dg[dg.cls == tc].set_index("item")[mtr]
                ri = dg[dg.cls == rc].groupby("item")[mtr].mean()
                c[site] = ti - ri.reindex(ti.index)
                t5 = float(ti.mean())
                draws = dg[dg.cls == rc].groupby("condition")[mtr].mean().values
                res_ = boot_p(c[site].values, sid.reindex(c[site].index).values)
                rows_out.append({"group": g, "measure": mtr, "site": site, "top5": t5,
                                 "p_rank": float((1 + (np.abs(draws) >= abs(t5)).sum()) / (1 + len(draws))),
                                 "net": res_[0], "lo": res_[1], "hi": res_[2], "p": res_[3], "n": res_[4]})
            for a_, b_ in [("next1", "target"), ("later", "target"), ("later", "next1")]:
                dd = c[a_] - c[b_].reindex(c[a_].index)
                net[f"{g}/{mtr}/{a_}_minus_{b_}"] = boot_p(dd.values, sid.reindex(dd.index).values)
    tab = pd.DataFrame(rows_out)
    tab.to_csv(out / "S_sites.csv", index=False)
    summary = {"model": args.model, "heads_from": f"{args.heads_from}/{heads_slug}", "readout_layer": R,
               "top_heads": [f"L{L}H{h}" for L, h in top], "n_items": {g: len(items[g]) for g in GROUPS},
               "n_tweets": {g: int(df[df.group == g].sid.nunique()) for g in GROUPS},
               "random_draws": {"target": args.random_draws, "next1": args.later_draws, "later": args.later_draws},
               "sites": tab.round(5).to_dict("records"), "paired": net,
               "minutes": round((time.time() - T0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    pd.set_option("display.width", 220)
    key = tab[(tab.group == "EGY_in_MSA") & tab.measure.isin(["d_cont_lp_matrix", "d_cont_lp_matrix_x1",
                                                              "d_cont_lp_matrix_w1", "d_cont_lp_matrix_w2"])]
    print(key.round(4).to_string(index=False))
    for k, v in net.items():
        if k.startswith("EGY_in_MSA/d_cont_lp_matrix"):
            print(f"  {k:60s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] p={v[3]:.4f} n={v[4]}")
    print(f"\nResults in {out}  ({summary['minutes']} min)")


if __name__ == "__main__":
    main()
