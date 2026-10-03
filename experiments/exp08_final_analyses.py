"""
exp08 - Final analyses requested in the third review. Runs on SAVED outputs (exp06 B_long.csv,
exp07 L_items.csv, saved probes); only the tokenizers are loaded (no model forward passes).

A. Source vs later: paired, item-level contrast of the two intervention sites.
     raw     : d(top5_later) - d(top5 at the embedded word)                  per item
     net     : [top5_later - mean(random_later)] - [top5 - mean(random5)]      per item
   tweet-cluster bootstrap CI and two-sided bootstrap p; for every exp06 run.
B. Primary-test family with Holm correction (per run), using tweet-cluster bootstrap p-values of
   the net (top5 minus random) contrasts:
     MSA_in_EGY d_own_logit, EGY_in_MSA d_own_logit, MSA_in_EGY d_D, EGY_in_MSA d_D,
     EGY_in_MSA d_cont_lp_matrix (later).
C. Matched-sample sensitivity: inside the exp07b matched sample (exact on function/content word;
   lexical strength, host mean, matrix words on the left), a linear probability model
     flip ~ is_MSA + z(n_chars) + z(log_freq) + z(n_subtok)
   with pair-clustered SEs - does the gap survive the covariates that stayed imbalanced?
D. Independence audit:
     * probe split: train/held-out cut position (word level, tweets in permuted order);
     * how many monolingual CONTROL tweets of exp06 were also used as exp02 prefix tweets
       (head selection); mono-control contrasts recomputed WITHOUT those tweets.

Usage
    python exp08_final_analyses.py
Writes results/exp08_final_analyses.json and prints the tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from readout import PROJECT_DIR, boot, load_rows_with_ids, mono_sentences, probe_split, type_classes

RUNS = {"gemma-2-9b": "google/gemma-2-9b", "Fanar-1-9B": "QCRI/Fanar-1-9B",
        "Fanar-1-9B__heads-gemma-2-9b": "QCRI/Fanar-1-9B",
        "ALLaM-7B-Instruct-preview": "ALLaM-AI/ALLaM-7B-Instruct-preview", "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base"}


def boot_p(x, groups, n=5000, seed=0):
    """mean, 95% CI and two-sided bootstrap p (tweet-cluster resampling)"""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, g = x[ok], np.asarray(groups)[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    sums, cnts = np.bincount(inv, x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), (n, len(uniq)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(max(p, 1 / n)), int(len(x))


def item_contrast(df, group, measure, top_cls, rand_cls):
    d = df[df.group == group]
    t = d[d.cls == top_cls].set_index("item")[measure]
    r = d[d.cls == rand_cls].groupby("item")[measure].mean()
    sid = d.groupby("item").sid.first()
    return (t - r.reindex(t.index)), t, sid.reindex(t.index)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    res = {}
    # ------------------------------------------------------------------ A + B
    print("A. Source vs later (EGY word in MSA tweet -> following MSA words, d_cont_lp_matrix)")
    for run in RUNS:
        f = PROJECT_DIR / "results" / "exp06" / run / "B_long.csv"
        if not f.exists():
            print(f"  (missing {f})")
            continue
        df = pd.read_csv(f)
        if "top5_later" not in set(df.cls):
            print(f"  {run}: no later-only condition in this B_long (old run) - skipped")
            continue
        r = {}
        for g in ["EGY_in_MSA", "MSA_in_EGY"]:
            for mtr in ["d_cont_lp_matrix", "d_cont_lp_own"]:
                c_src, t_src, sid = item_contrast(df, g, mtr, "top5", "random5")
                c_lat, t_lat, _ = item_contrast(df, g, mtr, "top5_later", "random5_later")
                r[f"{g}/{mtr}"] = {
                    "source_net": boot_p(c_src.values, sid.values),
                    "later_net": boot_p(c_lat.values, sid.values),
                    "later_minus_source_raw": boot_p((t_lat - t_src).values, sid.values),
                    "later_minus_source_net": boot_p((c_lat - c_src).values, sid.values)}
        # B: primary family
        fam = [("MSA_in_EGY", "d_own_logit", "top5", "random5"), ("EGY_in_MSA", "d_own_logit", "top5", "random5"),
               ("MSA_in_EGY", "d_D_to_matrix", "top5", "random5"), ("EGY_in_MSA", "d_D_to_matrix", "top5", "random5"),
               ("EGY_in_MSA", "d_cont_lp_matrix", "top5_later", "random5_later")]
        rows = []
        for g, mtr, tc, rc in fam:
            c, _, sid = item_contrast(df, g, mtr, tc, rc)
            m, lo, hi, p, n = boot_p(c.values, sid.values)
            rows.append({"test": f"{g}/{mtr}/{tc}", "net": m, "lo": lo, "hi": hi, "p": p, "n_items": n})
        adj = holm(np.array([x["p"] for x in rows]))
        for x, a in zip(rows, adj):
            x["p_holm"] = float(a)
        r["primary_family_holm"] = rows
        res.setdefault("A_B", {})[run] = r
        e = r["EGY_in_MSA/d_cont_lp_matrix"]
        fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] p={t[3]:.3f}"
        print(f"  {run:30s} source {fmt(e['source_net'])} | later {fmt(e['later_net'])} | "
              f"later-source (net) {fmt(e['later_minus_source_net'])}")
    print("\nB. Primary family, Holm-adjusted (net contrast vs random):")
    for run, r in res.get("A_B", {}).items():
        print(f"  {run}")
        for x in r["primary_family_holm"]:
            print(f"     {x['test']:45s} {x['net']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]  p={x['p']:.4f}  p_holm={x['p_holm']:.4f}")

    # ------------------------------------------------------------------ C
    print("\nC. Matched-sample sensitivity (exact function/content; lex + host mean + matrix-left):")
    import statsmodels.formula.api as smf
    from exp07b_matching_sensitivity import match
    for f in sorted((PROJECT_DIR / "results" / "exp07").glob("*/L_items.csv")):
        slug = f.parent.name
        d = pd.read_csv(f).dropna(subset=["host_mean"])
        msa, egy = d[d.is_MSA == 1], d[d.is_MSA == 0]
        pairs = match(msa, egy, ["lex_own", "host_mean", "n_left_matrix"], 0.35)
        if len(pairs) < 10:
            continue
        rows = []
        for k, (i, j) in enumerate(pairs):
            rows.append({**d.loc[i].to_dict(), "pair": k})
            rows.append({**d.loc[j].to_dict(), "pair": k})
        m = pd.DataFrame(rows)
        for c in ["n_chars", "log_freq", "n_subtok"]:
            m[c + "_z"] = (m[c] - m[c].mean()) / (m[c].std() + 1e-9)
        out = {}
        for name, rhs in [("unadjusted", "is_MSA"), ("adjusted_len_freq_subtok", "is_MSA + n_chars_z + log_freq_z + n_subtok_z")]:
            fit = smf.ols(f"flip ~ {rhs}", m).fit(cov_type="cluster", cov_kwds={"groups": m.pair})
            ci = fit.conf_int().loc["is_MSA"].tolist()
            out[name] = {"coef_is_MSA": float(fit.params["is_MSA"]), "ci": [float(ci[0]), float(ci[1])],
                         "p": float(fit.pvalues["is_MSA"]), "n_pairs": len(pairs)}
        res.setdefault("C", {})[slug] = out
        a, u = out["adjusted_len_freq_subtok"], out["unadjusted"]
        print(f"  {slug:28s} n={len(pairs)} pairs   unadjusted {u['coef_is_MSA']:+.3f} [{u['ci'][0]:+.3f}, {u['ci'][1]:+.3f}]"
              f"   adjusted {a['coef_is_MSA']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] p={a['p']:.4f}")

    # ------------------------------------------------------------------ D
    print("\nD. Independence audit:")
    from transformers import AutoTokenizer
    rows_all = load_rows_with_ids()
    plain = [(ws, ls) for _, ws, ls in rows_all]
    lince = [(ws, ls) for _, ws, ls in rows_all]                  # same order as exp02.load_lince()
    for run, mid in RUNS.items():
        if "__heads" in run:
            continue
        slug = run
        pfile = sorted((PROJECT_DIR / "results" / "probes" / slug).glob("probe_L*_seed13.pt"))
        blong = PROJECT_DIR / "results" / "exp06" / run / "B_long.csv"
        if not pfile or not blong.exists():
            continue
        probe = torch.load(pfile[0], weights_only=False)
        tok = AutoTokenizer.from_pretrained(mid)
        # --- exp02 prefix tweets (replicates exp02's RNG stream: seed 13, 150 pairs, K=12)
        rng = np.random.default_rng(13)
        msa_types, egy_types = type_classes(lince)
        pools = {}
        for v in ["MSA", "EGY"]:
            pool = []
            for ws, ls in mono_sentences(lince, v):
                ws2 = [w for w in ws if not w.startswith(("http", "@"))]
                for j in range(1, len(ws2) + 1):
                    n_ = len(tok(" ".join(ws2[:j]), add_special_tokens=False).input_ids)
                    if n_ == 12:
                        pool.append(tuple(ws))
                        break
                    if n_ > 12:
                        break
            pools[v] = pool
        used = {"MSA": set(), "EGY": set()}
        for types in [msa_types, egy_types]:
            words = rng.choice(types, min(150, len(types)), replace=len(types) < 150)
            for _ in words:
                used["EGY"].add(pools["EGY"][rng.integers(len(pools["EGY"]))])
                used["MSA"].add(pools["MSA"][rng.integers(len(pools["MSA"]))])
        # --- exp06 monolingual control tweets
        split = probe_split(plain, probe["n_used_tweets"], 13)
        df = pd.read_csv(blong)
        overlap_items = set()
        audit = {}
        for v in ["MSA", "EGY"]:
            ctrl = df[df.group == f"mono_{v}"].drop_duplicates("item")
            ks = ctrl.sid.str.replace(f"mono{v}-", "", regex=False).astype(int)
            tweets = [tuple(split[v][1][k][0]) for k in ks]
            ov = [it for it, tw in zip(ctrl.item, tweets) if tw in used[v]]
            overlap_items |= set(ov)
            audit[v] = {"control_tweets": len(tweets), "also_exp02_prefix_tweets": len(ov),
                        "exp02_prefix_tweets": len(used[v])}
        # contrasts without the overlapping control tweets
        clean = df[~df.item.isin(overlap_items)]
        for v in ["MSA", "EGY"]:
            c, _, sid = item_contrast(clean, f"mono_{v}", "d_own_logit", "top5", "random5")
            audit[v]["mono_own_logit_net_without_overlap"] = boot_p(c.values, sid.values)[:3]
        audit["probe_split"] = {"unit": "words, 80/20, tweets in permuted order (at most one tweet straddles the cut)",
                                "n_train_words": probe["stats"].get("n_train"), "tweets_used": probe["n_used_tweets"]}
        res.setdefault("D", {})[slug] = audit
        print(f"  {slug:28s} control tweets also used as exp02 prefixes: "
              f"MSA {audit['MSA']['also_exp02_prefix_tweets']}/{audit['MSA']['control_tweets']}, "
              f"EGY {audit['EGY']['also_exp02_prefix_tweets']}/{audit['EGY']['control_tweets']}; "
              f"mono net own-logit without them: MSA {audit['MSA']['mono_own_logit_net_without_overlap'][0]:+.2f}, "
              f"EGY {audit['EGY']['mono_own_logit_net_without_overlap'][0]:+.2f}")

    out = PROJECT_DIR / "results" / "exp08_final_analyses.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
