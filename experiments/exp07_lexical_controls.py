"""
Experiment 07 - Does "markedness" survive the obvious alternative explanations?
(revision: reviewer points 3, 4 and 5; plus equivalence tests)

For every natural embedded word (unambiguous type; purity from MONOLINGUAL tweets by
default) we read, with the SAME saved probe as exp06 (layer R):

  nat_own   : own-variety logit in its real context          (flip = nat_own < 0)
  lex_own   : own-variety logit after random OWN-variety prefixes of the same length
              (the word's lexical evidence before any cross-variety context; mean of 2)
  host_last : logit of the last prefix word, signed toward the matrix variety (exp04 def.)
  host_mean : mean over the OTHER labelled prefix words (last word excluded), signed toward
              the matrix variety - context strength not driven by the neighbouring word
plus word covariates: log frequency, characters, sub-tokens, function word (closed-class
list, clitic-stripped), matrix/own words on the left, and a heuristic Egyptian
verb-morphology marker (descriptive only).

Analyses
  R1  logistic regressions of flip on variety, adding covariates step by step
      (cluster-robust SEs by tweet); does the MSA effect survive lexical strength?
  R2  matching MSA items to EGY items on lexical strength (caliper) and comparing flips
  R3  flips by variety x function/content word
  R4  host strength: last word vs mean of the other prefix words
  R5  robustness to the purity definition (mono-only vs all labels)
  R6  equivalence tests (TOST, 90% bootstrap CI vs a pre-set margin of 25% of the pooled
      effect) for the minimal-pair context effect (exp02) and the natural pull (exp04)

Usage
    python exp07_lexical_controls.py --model Qwen/Qwen3-8B-Base
"""
import argparse
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from readout import (LANG, PROJECT_DIR, boot, boot_diff, get_layers, get_probe, load_rows_with_ids, mono_sentences,
                     readout_layer, type_classes, type_classes_mono, with_bos, word_ids)
from exp03_natural_vs_controlled import prefix_pools_all_K

# closed-class words (MSA + Egyptian); matched after stripping a leading conjunction و/ف
FUNCTION_WORDS = set("""
من في على إلى الى الي إلي عن مع عند لدى حتى منذ بين تحت فوق قبل بعد خلال حول ضد نحو دون بدون غير سوى
و ف ثم أو او أم ام بل لكن لكن لكنه ولكن إن ان أن إنه انه أنه إنها انها أنها لأن لان كي لكي حين عندما إذا اذا إذ لو لولا كأن كان
لا لم لن ما ليس ليست قد لقد سوف سـ هل أ أما اما إما إلا الا
هذا هذه ذلك تلك هؤلاء هاؤلاء أولئك هنا هناك هنالك الذي التي الذين اللذين اللتين اللاتي اللواتي
هو هي هم هن هما أنا انا نحن أنت انت أنتم انتم أنتن انتي إياه
كل بعض أي اي أيضا ايضا فقط جدا كما مثل عليه عليها عليهم فيه فيها فيهم منه منها منهم له لها لهم به بها بهم إليه اليه
مش مو ده دي دا دول دوك دولا اللي الى إللي بتاع بتاعت بتوع بتاعي بتاعك عشان علشان علشانك ازاي إزاي ليه ليش فين وين امتى إمتى
ايه إيه إيش ايش مين كده كدا كدة كدهو اوي أوي قوي بس لسه لسا خلاص برضه برضو بردو يعني يعنى بقى بقا عايز عاوز عايزة عاوزة
احنا إحنا انتو إنتو انتوا هما همه اهو أهو اهي أهي اهم أهم ماهو ماهي زي زى ولا وله طب طيب يا ياريت ياللا يلا هو هيا
""".split())
EGY_MORPH = re.compile(r"^(?:و|ف)?(?:ب|ه|ح)(?:ي|ت|ن|ا)\w{2,}|\w{2,}ش$")


def is_function(w):
    return w in FUNCTION_WORDS or (len(w) > 1 and w[0] in "وف" and w[1:] in FUNCTION_WORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--purity_source", default="mono", choices=["mono", "all"])
    ap.add_argument("--max_items", type=int, default=400)
    ap.add_argument("--min_K", type=int, default=3)
    ap.add_argument("--max_K", type=int, default=48)
    ap.add_argument("--probe_words", type=int, default=6000)
    ap.add_argument("--caliper", type=float, default=0.25, help="matching caliper in pooled-SD units of lex_own")
    ap.add_argument("--margin", type=float, default=0.25, help="TOST margin as a fraction of the pooled effect")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    slug = args.model.split("/")[-1]
    out = PROJECT_DIR / "results" / "exp07" / (slug + ("" if args.purity_source == "mono" else f"__purity-{args.purity_source}"))
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    n_layers = len(get_layers(model))
    R = readout_layer(n_layers)

    rows = load_rows_with_ids()
    plain = [(ws, ls) for _, ws, ls in rows]
    tc = {"mono": type_classes_mono(plain), "all": type_classes(plain)}
    types = {k: {"MSA": set(v[0]), "EGY": set(v[1])} for k, v in tc.items()}
    use = types[args.purity_source]
    freq = Counter(w for ws, _ in plain for w in ws)
    print("unambiguous types: " + "  ".join(f"{k}: MSA={len(v['MSA'])} EGY={len(v['EGY'])}" for k, v in types.items()))

    probe = get_probe(model, tok, plain, slug, R, device, args.probe_words, args.seed)
    w, b = probe["w"], probe["b"]
    pools = {v: prefix_pools_all_K(tok, mono_sentences(plain, v), args.max_K) for v in ["MSA", "EGY"]}

    # ------------------------------------------------------------------ items
    items = {"MSA": [], "EGY": []}
    for sid, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs or langs.count("EGY") == langs.count("MSA"):
            continue
        matrix = "EGY" if langs.count("EGY") > langs.count("MSA") else "MSA"
        for i, (wd, l) in enumerate(zip(ws, ls)):
            if l not in LANG or LANG[l] == matrix or i == 0 or wd not in use[LANG[l]]:
                continue
            own = LANG[l]
            left_text = " ".join(ws[:i])
            enc = tok(left_text, add_special_tokens=False, return_offsets_mapping=True)
            K = len(enc.input_ids)
            if K < args.min_K or K > args.max_K or not pools[own].get(K) or len(pools[own][K]) < 2:
                continue
            # last-subword position of every prefix word (within left_ids)
            starts, pos_of_word, cur = [], {}, 0
            for j, x in enumerate(ws[:i]):
                starts.append((cur, cur + len(x)))
                cur += len(x) + 1
            for ti, (a, e_) in enumerate(enc["offset_mapping"]):
                for j, (s0, s1) in enumerate(starts):
                    if e_ > a and s0 < e_ <= s1:
                        pos_of_word[j] = ti
            prefix_words = [(pos_of_word[j], LANG[ls[j]]) for j in range(i) if j in pos_of_word and ls[j] in LANG]
            left_lang = [LANG[x] for x in ls[:i] if x in LANG]
            items[own].append({
                "sid": sid, "word": wd, "own": own, "matrix": matrix, "K": K, "left_ids": enc.input_ids,
                "prefix_words": prefix_words, "n_left_matrix": left_lang.count(matrix), "n_left_own": left_lang.count(own),
                "in_mono_purity": wd in types["mono"][own], "in_all_purity": wd in types["all"][own]})
    for v in items:
        if len(items[v]) > args.max_items:
            keep = rng.choice(len(items[v]), args.max_items, replace=False)
            items[v] = [items[v][i] for i in sorted(keep)]
    print(f"items: MSA-in-EGY={len(items['MSA'])}  EGY-in-MSA={len(items['EGY'])}")

    recs = []
    all_items = items["MSA"] + items["EGY"]
    for n, it in enumerate(all_items):
        wid = word_ids(tok, it["word"])
        K = it["K"]
        pk = pools[it["own"]][K]
        own_prefixes = [pk[j] for j in rng.choice(len(pk), 2, replace=False)]
        seqs = [with_bos(tok, p + wid) for p in [it["left_ids"]] + own_prefixes]
        off = len(seqs[0]) - len(it["left_ids"]) - len(wid)                 # 1 if BOS was added
        with torch.no_grad():
            hs = model(input_ids=torch.tensor(seqs, device=device), output_hidden_states=True).hidden_states[R]
        z = ((hs.float() * w).sum(-1) + b).cpu().numpy()                    # [3, T] EGY-positive
        own_sign = 1.0 if it["own"] == "EGY" else -1.0
        mat_sign = -own_sign
        t = len(seqs[0]) - 1
        pw = it["prefix_words"]
        host_last = mat_sign * z[0, off + it["K"] - 1]
        others = [mat_sign * z[0, off + p] for p, _ in pw if p != K - 1]
        recs.append({
            "item": n, "sid": it["sid"], "word": it["word"], "own": it["own"], "is_MSA": int(it["own"] == "MSA"),
            "nat_own": own_sign * z[0, t], "lex_own": float(np.mean(own_sign * z[1:, t])),
            "host_last": host_last, "host_mean": float(np.mean(others)) if others else np.nan,
            "log_freq": float(np.log(freq[it["word"]])), "n_chars": len(it["word"]), "n_subtok": len(wid),
            "is_function": int(is_function(it["word"])), "egy_morph": int(bool(EGY_MORPH.match(it["word"]))),
            "n_left_matrix": it["n_left_matrix"], "n_left_own": it["n_left_own"], "K": K,
            "in_mono_purity": int(it["in_mono_purity"]), "in_all_purity": int(it["in_all_purity"])})
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(all_items)} ({time.time()-T0:.0f}s)")
    df = pd.DataFrame(recs)
    df["flip"] = (df.nat_own < 0).astype(int)
    df["pull"] = df.lex_own - df.nat_own                                    # how far the context moved the word
    df.to_csv(out / "L_items.csv", index=False)
    res = {"model": args.model, "readout_layer": R, "purity_source": args.purity_source, "probe": probe["stats"],
           "n_items": {"MSA": int(df.is_MSA.sum()), "EGY": int((1 - df.is_MSA).sum())}}

    # ------------------------------------------------------------------ R0 descriptives
    desc = df.groupby("own")[["flip", "nat_own", "lex_own", "pull", "host_last", "host_mean", "log_freq", "n_chars",
                              "n_subtok", "is_function", "egy_morph", "n_left_matrix"]].mean()
    desc.to_csv(out / "L_descriptives.csv")
    res["descriptives"] = desc.round(3).to_dict("index")

    # ------------------------------------------------------------------ R1 regressions
    import statsmodels.formula.api as smf
    d = df.dropna(subset=["host_mean"]).copy()
    for c in ["lex_own", "log_freq", "n_chars", "n_subtok", "n_left_matrix", "host_mean", "host_last"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / (d[c].std() + 1e-9)
    base_cov = "lex_own_z + log_freq_z + n_chars_z + n_subtok_z + is_function + n_left_matrix_z"
    specs = {"M0 variety only": "is_MSA",
             "M1 + lexical strength": "is_MSA + lex_own_z",
             "M2 + word covariates": f"is_MSA + {base_cov}",
             "M3 + host (mean of other words)": f"is_MSA + {base_cov} + host_mean_z",
             "M4 + host (last word)": f"is_MSA + {base_cov} + host_last_z",
             "M5 + both hosts": f"is_MSA + {base_cov} + host_mean_z + host_last_z"}
    reg = []
    groups = pd.factorize(d.sid)[0]
    for name, rhs in specs.items():
        for outcome, fam in [("flip", "logit"), ("nat_own", "ols")]:
            f = f"{outcome} ~ {rhs}"
            try:
                if fam == "logit":
                    fit = smf.logit(f, d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
                    d1, d0 = d.assign(is_MSA=1), d.assign(is_MSA=0)
                    ame = float((fit.predict(d1) - fit.predict(d0)).mean())
                else:
                    fit = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups})
                    ame = float(fit.params["is_MSA"])
                ci = fit.conf_int().loc["is_MSA"].tolist()
                row = {"model": name, "outcome": outcome, "coef_is_MSA": float(fit.params["is_MSA"]),
                       "ci_lo": ci[0], "ci_hi": ci[1], "p": float(fit.pvalues["is_MSA"]),
                       "avg_marginal_effect_is_MSA": ame, "n": int(fit.nobs)}
                for k in ["lex_own_z", "host_mean_z", "host_last_z", "is_function"]:
                    if k in fit.params:
                        row[f"coef_{k}"] = float(fit.params[k])
                reg.append(row)
            except Exception as ex:                                        # separation etc.
                reg.append({"model": name, "outcome": outcome, "error": str(ex)[:120]})
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "L_regressions.csv", index=False)
    res["regressions"] = reg.round(4).to_dict("records")

    # ------------------------------------------------------------------ R2 matching on lexical strength
    sd_pool = df.lex_own.std()
    msa, egy = df[df.is_MSA == 1], df[df.is_MSA == 0]
    cal = args.caliper * sd_pool

    zc = {c: (df[c].mean(), df[c].std() + 1e-9) for c in ["lex_own", "host_mean", "host_last"]}

    def match(replace, cols=("lex_own",)):
        """nearest-neighbour matching of MSA items to EGY items on the given covariates (z-scored,
        Euclidean, within the caliper); greedy, best matches first"""
        used, pairs = set(), []
        E = egy.dropna(subset=list(cols))
        cand = []
        for i, r in msa.dropna(subset=list(cols)).iterrows():
            d2 = sum(((E[c] - r[c]) / zc[c][1]) ** 2 for c in cols) ** 0.5 * zc["lex_own"][1]
            cand.append((d2, i))
        order = sorted(cand, key=lambda x: x[0].min())
        for dist, i in order:
            dist = dist.drop(list(used)) if (not replace and used) else dist
            if dist.empty:
                break
            j = dist.idxmin()
            if dist[j] <= cal:
                pairs.append((i, j))
                if not replace:
                    used.add(j)
        return pairs

    overlap = {"lex_own_range_MSA": [float(msa.lex_own.min()), float(msa.lex_own.max())],
               "lex_own_range_EGY": [float(egy.lex_own.min()), float(egy.lex_own.max())],
               "caliper_logit": float(cal)}
    for key, replace, cols in [("lex_with_replacement", True, ("lex_own",)),
                               ("lex_without_replacement", False, ("lex_own",)),
                               ("lex_and_host_mean_without_replacement", False, ("lex_own", "host_mean")),
                               ("lex_and_both_hosts_without_replacement", False, ("lex_own", "host_mean", "host_last"))]:
        pairs = match(replace, cols)
        rec = {"n_pairs": len(pairs), "n_unique_EGY_controls": len({j for _, j in pairs})}
        if pairs:
            mi_, ej_ = [i for i, _ in pairs], [j for _, j in pairs]
            rec.update({
                "n_unique_MSA_types": int(msa.loc[mi_, "word"].nunique()),
                "n_unique_EGY_types": int(egy.loc[ej_, "word"].nunique()),
                "n_unique_tweets": int(pd.concat([msa.loc[mi_, "sid"], egy.loc[ej_, "sid"]]).nunique()),
                "max_share_single_type": float(max(msa.loc[mi_, "word"].value_counts(normalize=True).max(),
                                                   egy.loc[ej_, "word"].value_counts(normalize=True).max())),
                # standardised mean differences after matching (|SMD| < 0.1 = good balance)
                "SMD_after": {c: float((msa.loc[mi_, c].mean() - egy.loc[ej_, c].mean()) /
                                       (np.sqrt(0.5 * (msa[c].var() + egy[c].var())) + 1e-9))
                              for c in ["lex_own", "host_mean", "host_last", "log_freq", "n_chars", "n_subtok",
                                        "is_function", "n_left_matrix"]}})
        if len(pairs) >= 10:
            mi, ej = [i for i, _ in pairs], [j for _, j in pairs]
            fm, fe = msa.loc[mi, "flip"].values, egy.loc[ej, "flip"].values
            pm, pe = msa.loc[mi, "pull"].values, egy.loc[ej, "pull"].values
            rec.update({"lex_own_MSA": float(msa.loc[mi, "lex_own"].mean()), "lex_own_EGY": float(egy.loc[ej, "lex_own"].mean()),
                        "flip_MSA": float(fm.mean()), "flip_EGY": float(fe.mean()), "flip_diff": boot_diff(fm, fe),
                        "pull_MSA": float(pm.mean()), "pull_EGY": float(pe.mean()), "pull_diff": boot_diff(pm, pe),
                        "host_mean_MSA": float(np.nanmean(msa.loc[mi, "host_mean"])),
                        "host_mean_EGY": float(np.nanmean(egy.loc[ej, "host_mean"])),
                        "host_last_MSA": float(msa.loc[mi, "host_last"].mean()),
                        "host_last_EGY": float(egy.loc[ej, "host_last"].mean())})
        overlap[key] = rec
    overlap["SMD_before"] = {c: float((msa[c].mean() - egy[c].mean()) / (np.sqrt(0.5 * (msa[c].var() + egy[c].var())) + 1e-9))
                             for c in ["lex_own", "host_mean", "host_last", "log_freq", "n_chars", "n_subtok",
                                       "is_function", "n_left_matrix"]}

    # sensitivity: at most ONE item per word type (random pick, 20 repeats) - is the matched gap driven
    # by a few frequent types?
    sens = []
    msa_all, egy_all = msa, egy
    for rep_ in range(20):
        rs = np.random.default_rng(100 + rep_)
        pick = lambda d: d.loc[[rs.choice(ix) for ix in d.groupby("word").groups.values()]]
        msa, egy = pick(msa_all), pick(egy_all)
        pr = match(False, ("lex_own", "host_mean"))
        if len(pr) >= 5:
            sens.append({"n_pairs": len(pr), "flip_diff": float(msa.loc[[i for i, _ in pr], "flip"].mean()
                                                               - egy.loc[[j for _, j in pr], "flip"].mean())})
    msa, egy = msa_all, egy_all
    if sens:
        sd_ = pd.DataFrame(sens)
        overlap["one_item_per_type_lex_and_host_mean"] = {
            "repeats": int(len(sd_)), "mean_n_pairs": float(sd_.n_pairs.mean()),
            "flip_diff_mean": float(sd_.flip_diff.mean()),
            "flip_diff_range": [float(sd_.flip_diff.min()), float(sd_.flip_diff.max())]}

    bins = pd.qcut(df.lex_own, 5, duplicates="drop")
    strat = df.groupby([bins, "own"], observed=True).agg(n=("flip", "size"), flip=("flip", "mean"),
                                                          pull=("pull", "mean")).reset_index()
    strat["lex_own"] = strat["lex_own"].astype(str)
    strat.to_csv(out / "L_strata_by_lexical_strength.csv", index=False)
    res["matching"] = overlap
    res["strata"] = strat.round(3).to_dict("records")

    # ------------------------------------------------------------------ R3 function vs content
    fc = df.groupby(["own", "is_function"]).agg(n=("flip", "size"), flip=("flip", "mean"), lex_own=("lex_own", "mean"),
                                                pull=("pull", "mean")).reset_index()
    fc.to_csv(out / "L_function_vs_content.csv", index=False)
    res["function_vs_content"] = fc.round(3).to_dict("records")
    mo = df[df.own == "EGY"].groupby("egy_morph").agg(n=("flip", "size"), flip=("flip", "mean"),
                                                      lex_own=("lex_own", "mean")).reset_index()
    res["egy_morph_marker (heuristic, EGY items)"] = mo.round(3).to_dict("records")

    # ------------------------------------------------------------------ R4 host definitions
    host = {}
    for v in ["MSA", "EGY"]:
        dv = df[df.own == v]
        host[v] = {"host_last": boot(dv.host_last.values, groups=dv.sid.values),
                   "host_mean_other_words": boot(dv.host_mean.values, groups=dv.sid.values),
                   "corr(pull, host_last)": float(dv[["pull", "host_last"]].corr().iloc[0, 1]),
                   "corr(pull, host_mean)": float(dv[["pull", "host_mean"]].corr().iloc[0, 1])}
    res["host"] = host

    # ------------------------------------------------------------------ R5 purity robustness
    rob = {}
    for flag in ["in_mono_purity", "in_all_purity"]:
        sub = df[df[flag] == 1]
        rob[flag] = {v: {"n": int((sub.own == v).sum()), "flip": float(sub[sub.own == v].flip.mean())}
                     for v in ["MSA", "EGY"]}
    res["purity_robustness"] = rob

    # ------------------------------------------------------------------ R6 equivalence (TOST)
    def tost(a, bb, margin_frac, n=5000):
        a, bb = np.asarray(a, float), np.asarray(bb, float)
        a, bb = a[~np.isnan(a)], bb[~np.isnan(bb)]
        pooled = abs(np.r_[a, bb].mean())
        delta = margin_frac * pooled
        r_ = np.random.default_rng(2)
        diffs = a[r_.integers(0, len(a), (n, len(a)))].mean(1) - bb[r_.integers(0, len(bb), (n, len(bb)))].mean(1)
        lo, hi = np.percentile(diffs, [5, 95])
        return {"diff": float(a.mean() - bb.mean()), "ci90": [float(lo), float(hi)], "margin": float(delta),
                "equivalent": bool(lo > -delta and hi < delta), "n": [int(len(a)), int(len(bb))]}
    eq = {}
    p1 = PROJECT_DIR / "results" / "exp02" / slug / "P1_scores_long.csv"
    if p1.exists():
        s1 = pd.read_csv(p1)
        s1 = s1[s1.layer == R].pivot_table(index=["pair", "target"], columns="prefix", values="s").reset_index()
        s1["ctx"] = s1.EGY - s1.MSA
        eq["exp02 minimal-pair context effect (MSA vs EGY targets)"] = tost(
            s1[s1.target == "MSA"].ctx, s1[s1.target == "EGY"].ctx, args.margin)
    y = PROJECT_DIR / "results" / "exp04" / slug / "Y_long.csv"
    if y.exists():
        yl = pd.read_csv(y)
        yl = yl[(yl.layer == R) & (yl.n_left_matrix >= 3)]
        for mtr in ["natural_pull", "unintegrated_pull"]:
            eq[f"exp04 {mtr} (MSA-in-EGY vs EGY-in-MSA)"] = tost(yl[yl.own == "MSA"][mtr], yl[yl.own == "EGY"][mtr],
                                                                 args.margin)
    res["equivalence_TOST"] = eq
    res["minutes"] = round((time.time() - T0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))

    pd.set_option("display.width", 200)
    print("\nDescriptives:\n", desc.round(2).to_string())
    print("\nRegressions (coefficient of 'embedded word is MSA'):")
    cols = [c for c in ["model", "outcome", "coef_is_MSA", "ci_lo", "ci_hi", "p", "avg_marginal_effect_is_MSA",
                        "coef_lex_own_z", "coef_host_mean_z", "coef_host_last_z", "error"] if c in reg.columns]
    print(reg[cols].round(3).to_string(index=False))
    print("\nMatching on lexical strength:\n", json.dumps(overlap, indent=1, default=float))
    print("\nFunction vs content:\n", fc.round(3).to_string(index=False))
    print("\nHost definitions:\n", json.dumps(host, indent=1, default=float))
    print("\nPurity robustness:\n", json.dumps(rob, indent=1))
    print("\nEquivalence (TOST):\n", json.dumps(eq, indent=1))
    print(f"\nResults in {out}  ({res['minutes']} min)")


if __name__ == "__main__":
    main()
