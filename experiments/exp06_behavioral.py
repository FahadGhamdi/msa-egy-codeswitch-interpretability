"""
Experiment 06 - Does the head mechanism matter for what the MODEL does, not only for
what the probe reads? (revision: reviewer points 1, 2 and 7)

Items
  embedded : natural embedded words from code-switched LinCE tweets (unambiguous type,
             >= 3 matrix-variety words on the left, >= 1 word after it), as in exp05.
  mono     : control words of the same unambiguous types inside MONOLINGUAL tweets that
             were not used to train the probe (no switch; tests general disruption).

Conditions (heads ablated at the target word's last sub-token ONLY; mean ablation)
  baseline, top1, top3, top5 (heads from exp02 patching),
  random5 x N draws, each head replaced by a random head of the SAME layer with a similar
  output norm (layer- and norm-matched control distribution).

Measures (all from one forward pass per condition)
  own_logit : saved read-out probe at layer R, signed toward the word's own variety
  D         : model's next-token dialect preference at the target position, a LOG-RATIO in nats:
              log P(next token in the EGY set) - log P(next token in the MSA set), where each set holds
              the same number of distinctive first tokens of frequent unambiguous types,
              signed toward the MATRIX variety (for mono items: toward the own variety)
  cont_lp   : mean log-probability per token of the ACTUAL following words (up to 3),
              split by the label of those words (matrix-variety vs own-variety words).
              The intervention sits on the target word, so only what comes AFTER it can
              change - its own probability cannot.

Also
  * baseline link: do items the probe calls "flipped" have a model that already expects
    matrix-variety continuations? (probe reading -> model prediction)
  * direct attribution of the top heads with a TYPE-MATCHED monolingual reference
    (mono control words are the same unambiguous types as the embedded words).

Usage
    python exp06_behavioral.py --model google/gemma-2-9b
    python exp06_behavioral.py --model QCRI/Fanar-1-9B --heads_model gemma-2-9b   # same heads as Gemma
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
    ap.add_argument("--span_draws", type=int, default=20, help="random draws for the span (target..end) ablation")
    ap.add_argument("--later_draws", type=int, default=20, help="random draws for the later-only (target+1..end) ablation")
    ap.add_argument("--n_pref", type=int, default=50, help="max tokens per variety in the next-token preference sets")
    ap.add_argument("--random_draws", type=int, default=20)
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
    out = PROJECT_DIR / "results" / ("exp06" if args.heads_from == "exp02" else "exp06_msa_heads") / name
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
    conditions = [("baseline", [], "target"), ("top1", top[:1], "target"), ("top3", top[:3], "target"),
                  ("top5", top, "target")]
    conditions += [(f"random5_d{d}", random_draw(), "target") for d in range(args.random_draws)]
    conditions += [("top5_later", top, "later")]
    conditions += [(f"random5_later_d{d}", random_draw(), "later") for d in range(args.later_draws)]
    conditions += [("top5_span", top, "span")]
    conditions += [(f"random5_span_d{d}", random_draw(), "span") for d in range(args.span_draws)]
    (out / "conditions.json").write_text(json.dumps(
        {c: {"heads": [f"L{L}H{h}" for L, h in hs], "where": wh} for c, hs, wh in conditions}, indent=1))
    WHERE = {"target": lambda t: t, "later": lambda t: slice(t + 1, None), "span": lambda t: slice(t, None)}

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
                rec.append(r)
        return rec

    recs, dla_caps = [], {g: {L: [] for L in head_layers} for g in GROUPS}
    all_items = [it for g in GROUPS for it in items[g]]
    for n, it in enumerate(all_items):
        for r in run_item(it):
            recs.append({"item": n, "sid": it["sid"], "group": it["group"], "word": it["word"], **r})
        cap = collect_head_inputs(model, it["ids"][: it["t"] + 1], [it["t"]], head_layers, device)
        for L in head_layers:
            dla_caps[it["group"]][L].append(cap[L])
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(all_items)} ({time.time()-T0:.0f}s)")
    df = pd.DataFrame(recs)
    base = df[df.condition == "baseline"].set_index("item")
    for col in ["own_logit", "D_to_matrix", "cont_lp_matrix", "cont_lp_own", "cont_lp_all"]:
        df["d_" + col] = df[col] - df.item.map(base[col])
    df["flipped"] = (df.own_logit < 0).astype(float)
    df["cls"] = df.condition.str.replace(r"_d\d+$", "", regex=True)
    df.to_csv(out / "B_long.csv", index=False)

    # ------------------------------------------------------------------ summaries
    measures = ["d_own_logit", "d_D_to_matrix", "d_cont_lp_matrix", "d_cont_lp_own", "d_cont_lp_all"]
    summ = []
    for g in GROUPS:
        dg = df[df.group == g]
        if dg.empty:
            continue
        own_var = g.split("_")[0] if "_in_" in g else g.split("_")[1]
        for cls in ["baseline", "top1", "top3", "top5", "random5", "top5_later", "random5_later", "top5_span",
                    "random5_span"]:
            d = dg[dg.cls == cls]
            row = {"group": g, "condition": cls, "n_items": d.item.nunique(), "flip_rate": float(d.flipped.mean()),
                   "D_to_matrix": float(d.D_to_matrix.mean())}
            for mtr in measures:
                if cls.startswith("random"):  # item-level average over draws first
                    per_item = d.groupby("item")[mtr].mean()
                    mm, lo, hi = boot(per_item.values, groups=d.groupby("item").sid.first().values)
                else:
                    mm, lo, hi = boot(d[mtr].values, groups=d.sid.values)
                row[mtr], row[mtr + "_lo"], row[mtr + "_hi"] = mm, lo, hi
            row["d_own_logit_sd"] = row["d_own_logit"] / sd_own[own_var]
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "B_summary.csv", index=False)

    # top5 vs the random-draw distribution (draw-level means)
    null = []
    for g in GROUPS:
        dg = df[df.group == g]
        if dg.empty:
            continue
        for mtr in measures:
            for kind, tc, rc in [("target", "top5", "random5"), ("later", "top5_later", "random5_later"),
                                 ("span", "top5_span", "random5_span")]:
                t5 = float(dg[dg.cls == tc][mtr].mean())
                draws = dg[dg.cls == rc].groupby("condition")[mtr].mean().values
                if len(draws) == 0:
                    continue
                # item-level contrast: top-5 effect minus the mean effect of the random draws on the SAME item,
                # with a tweet-cluster bootstrap CI
                ti = dg[dg.cls == tc].set_index("item")[mtr]
                ri = dg[dg.cls == rc].groupby("item")[mtr].mean()
                cdf = pd.DataFrame({"c": ti - ri.reindex(ti.index)}).join(dg.groupby("item").sid.first())
                cm, clo, chi = boot(cdf.c.values, groups=cdf.sid.values)
                null.append({"group": g, "ablation": kind, "measure": mtr, "top5": t5, "random_mean": float(draws.mean()),
                             "random_2.5": float(np.percentile(draws, 2.5)), "random_97.5": float(np.percentile(draws, 97.5)),
                             "n_draws": int(len(draws)),
                             "p_rank_abs_ge_top5": float((1 + (np.abs(draws) >= abs(t5)).sum()) / (1 + len(draws))),
                             "contrast_top5_minus_random": cm, "contrast_lo": clo, "contrast_hi": chi})
    null = pd.DataFrame(null)
    null.to_csv(out / "B_top5_vs_random_null.csv", index=False)

    # probe reading -> model prediction (baseline only)
    link = {}
    b0 = df[df.cls == "baseline"]
    for g in ["MSA_in_EGY", "EGY_in_MSA"]:
        d = b0[b0.group == g]
        if len(d) < 10 or not has_D:
            continue
        fl, nf = d[d.flipped == 1].D_to_matrix.values, d[d.flipped == 0].D_to_matrix.values
        from scipy.stats import spearmanr
        rho = spearmanr(-d.own_logit, d.D_to_matrix)
        link[g] = {"n_flipped": int(len(fl)), "n_not": int(len(nf)),
                   "D_to_matrix_flipped_minus_not": boot_diff(fl, nf) if len(fl) > 4 and len(nf) > 4 else None,
                   "spearman(-own_logit, D_to_matrix)": [float(rho.statistic), float(rho.pvalue)]}
        t5 = df[(df.group == g) & (df.cls == "top5")].set_index("item")
        r2 = spearmanr(t5.d_own_logit, t5.d_D_to_matrix)
        link[g]["spearman(d_own_logit, d_D) under top5"] = [float(r2.statistic), float(r2.pvalue)]

    # type-matched direct attribution (positive = writes toward EGY)
    w_cpu = w.float().cpu()
    caps = {g: {L: torch.cat(dla_caps[g][L]) for L in head_layers} for g in GROUPS if dla_caps[g][head_layers[0]]}
    dla = []
    for L, h in top:
        r = {"head": f"L{L}H{h}"}
        for g in caps:
            r[g] = float(dla_matrix(model, L, h, caps[g][L], head_dim, w_cpu).mean())
        if all(k in r for k in GROUPS):
            r["shift_MSA_toward_EGY"] = r["MSA_in_EGY"] - r["mono_MSA"]
            r["shift_EGY_toward_MSA"] = -(r["EGY_in_MSA"] - r["mono_EGY"])
        dla.append(r)
    dla = pd.DataFrame(dla)
    dla.to_csv(out / "B_dla_type_matched.csv", index=False)

    summary = {"model": args.model, "heads_from": f"{args.heads_from}/{heads_slug}", "readout_layer": R,
               "purity_source": args.purity_source, "probe": probe["stats"], "top_heads": [f"L{L}H{h}" for L, h in top],
               "n_items": {g: len(items[g]) for g in GROUPS}, "random_draws": {"target": args.random_draws, "later": args.later_draws, "span": args.span_draws},
               "summary": summ.round(4).to_dict("records"), "top5_vs_random": null.round(4).to_dict("records"),
               "probe_to_prediction_link": link, "dla_type_matched": dla.round(4).to_dict("records"),
               "minutes": round((time.time() - T0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("d_own_logit", "Δ probe own-variety logit"), ("d_D_to_matrix", "Δ next-token preference\ntoward matrix variety"),
              ("d_cont_lp_matrix", "Δ log p of following\nmatrix-variety words")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row, (tc, rc, rlab) in enumerate([("top5", "random5", "embedded word only (source)"),
                                          ("top5_later", "random5_later", "later positions only"),
                                          ("top5_span", "random5_span", "embedded word + later positions")]):
        for ax, (mtr, lab) in zip(axes[row], panels):
            for gi, g in enumerate(GROUPS):
                dg = df[df.group == g]
                sel = summ[(summ.group == g) & (summ.condition == tc)]
                if dg.empty or sel.empty:
                    continue
                draws = dg[dg.cls == rc].groupby("condition")[mtr].mean().values
                ax.scatter(np.full(len(draws), gi) + rng.uniform(-0.12, 0.12, len(draws)), draws, s=10, color=C_GREY,
                           zorder=2)
                col = C_ORANGE if g in ("MSA_in_EGY", "mono_MSA") else C_BLUE
                r = sel.iloc[0]
                ax.errorbar([gi], [r[mtr]], yerr=[[r[mtr] - r[mtr + "_lo"]], [r[mtr + "_hi"] - r[mtr]]], fmt="o",
                            color=col, ms=8, lw=1.5, zorder=3)
            ax.axhline(0, color=C_MUTED, lw=0.8)
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels(["MSA in EGY", "EGY in MSA", "mono MSA", "mono EGY"], fontsize=8)
            ax.set_title(f"{lab}\n[ablation: {rlab}]", color=C_TEXT, fontsize=9.5)
            style(ax)
    fig.suptitle(f"B. Ablating the top-5 heads (colour, 95% CI) vs layer- and norm-matched random heads (grey) — {name}",
                 x=0.01, ha="left", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "B_behavioral.png", dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n", summ[["group", "condition", "n_items", "flip_rate", "d_own_logit", "d_own_logit_sd", "d_D_to_matrix",
                      "d_cont_lp_matrix", "d_cont_lp_own"]].round(3).to_string(index=False))
    print("\nTop-5 vs random-draw distribution:")
    print(null.round(3).to_string(index=False))
    print("\nProbe reading -> model prediction:")
    print(json.dumps(link, indent=1))
    print("\nType-matched direct attribution:")
    print(dla.round(3).to_string(index=False))
    print(f"\nResults in {out}  ({summary['minutes']} min)")


if __name__ == "__main__":
    main()
