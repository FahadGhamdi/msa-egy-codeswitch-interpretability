"""
exp07b - Matching sensitivity on the saved exp07 item tables (no model needed, seconds to run).

exp07's matching balanced lexical strength and host strength, but left residual imbalance in two
covariates (standardised mean differences around 0.5-0.7):
  * is_function   - EGY items are more often function words (which flip LESS) -> favours the finding
  * n_left_matrix - EGY items have more matrix words on their left (which flip MORE) -> works against it
This script re-matches with those covariates included:
  exact match on is_function, nearest neighbour on z(lex_own), z(host_mean), z(n_left_matrix)
  [and optionally z(host_last)], without replacement, within a caliper; reports balance (SMD),
  the flip-rate difference with a bootstrap CI, and a one-item-per-word-type sensitivity (50 repeats).

Usage
    python exp07b_matching_sensitivity.py                       # all models with results/exp07/*/L_items.csv
    python exp07b_matching_sensitivity.py --caliper 0.3
"""
import argparse
import json

import numpy as np
import pandas as pd

from readout import PROJECT_DIR, boot_diff

COVS = ["lex_own", "host_mean", "host_last", "log_freq", "n_chars", "n_subtok", "is_function", "n_left_matrix"]


def smd(a, b, c, ref):
    return float((a[c].mean() - b[c].mean()) / (np.sqrt(0.5 * (ref[0][c].var() + ref[1][c].var())) + 1e-9))


def match(msa, egy, cols, caliper, exact="is_function"):
    """greedy nearest neighbour without replacement, exact on `exact`, Euclidean on z-scored `cols`"""
    allx = pd.concat([msa, egy])
    z = {c: (allx[c].mean(), allx[c].std() + 1e-9) for c in cols}
    pairs, used = [], set()
    cands = []
    for i, r in msa.iterrows():
        E = egy[(egy[exact] == r[exact]) & ~egy.index.isin(used)] if exact else egy
        if E.empty:
            continue
        d = sum(((E[c] - r[c]) / z[c][1]) ** 2 for c in cols) ** 0.5
        cands.append((float(d.min()), i))
    for _, i in sorted(cands):                                   # best matches first
        r = msa.loc[i]
        E = egy[~egy.index.isin(used)]
        if exact:
            E = E[E[exact] == r[exact]]
        if E.empty:
            continue
        d = sum(((E[c] - r[c]) / z[c][1]) ** 2 for c in cols) ** 0.5
        j = d.idxmin()
        if d[j] <= caliper * np.sqrt(len(cols)):
            pairs.append((i, j))
            used.add(j)
    return pairs


def summarise(msa, egy, pairs, ref):
    mi, ej = [i for i, _ in pairs], [j for _, j in pairs]
    a, b = msa.loc[mi], egy.loc[ej]
    out = {"n_pairs": len(pairs), "n_MSA_types": int(a.word.nunique()), "n_EGY_types": int(b.word.nunique()),
           "n_tweets": int(pd.concat([a.sid, b.sid]).nunique()),
           "SMD_after": {c: round(smd(a, b, c, ref), 3) for c in COVS}}
    if len(pairs) >= 8:
        out.update({"flip_MSA": float(a.flip.mean()), "flip_EGY": float(b.flip.mean()),
                    "flip_diff": [round(x, 3) for x in boot_diff(a.flip.values, b.flip.values)],
                    "pull_diff": [round(x, 3) for x in boot_diff(a.pull.values, b.pull.values)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliper", type=float, default=0.35, help="per-covariate caliper in SD units")
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    res = {}
    for f in sorted((PROJECT_DIR / "results" / "exp07").glob("*/L_items.csv")):
        slug = f.parent.name
        df = pd.read_csv(f).dropna(subset=["host_mean"])
        msa, egy = df[df.is_MSA == 1], df[df.is_MSA == 0]
        ref = (msa, egy)
        r = {"n_items": {"MSA": len(msa), "EGY": len(egy)},
             "raw_flip": {"MSA": float(msa.flip.mean()), "EGY": float(egy.flip.mean())},
             "SMD_before": {c: round(smd(msa, egy, c, ref), 3) for c in COVS}}
        for key, cols in [("exact_func__lex_host_mean_nleft", ["lex_own", "host_mean", "n_left_matrix"]),
                          ("exact_func__lex_both_hosts_nleft", ["lex_own", "host_mean", "host_last", "n_left_matrix"])]:
            r[key] = summarise(msa, egy, match(msa, egy, cols, args.caliper), ref)
            diffs, ns = [], []
            for k in range(args.repeats):
                rs = np.random.default_rng(1000 + k)
                one = lambda d: d.loc[[rs.choice(ix) for ix in d.groupby("word").groups.values()]]
                m1, e1 = one(msa), one(egy)
                pr = match(m1, e1, cols, args.caliper)
                if len(pr) >= 5:
                    diffs.append(m1.loc[[i for i, _ in pr], "flip"].mean() - e1.loc[[j for _, j in pr], "flip"].mean())
                    ns.append(len(pr))
            if diffs:
                r[key]["one_item_per_type"] = {"repeats": len(diffs), "mean_n_pairs": float(np.mean(ns)),
                                               "flip_diff_mean": float(np.mean(diffs)),
                                               "flip_diff_2.5_97.5": [float(np.percentile(diffs, 2.5)),
                                                                      float(np.percentile(diffs, 97.5))],
                                               "share_of_repeats_with_diff>0": float(np.mean(np.array(diffs) > 0))}
        res[slug] = r
        print(f"\n#### {slug}  raw flip MSA {r['raw_flip']['MSA']:.2f} / EGY {r['raw_flip']['EGY']:.2f}")
        for key in ["exact_func__lex_host_mean_nleft", "exact_func__lex_both_hosts_nleft"]:
            v = r[key]
            s = v["SMD_after"]
            print(f"  {key}: n={v['n_pairs']} types {v['n_MSA_types']}/{v['n_EGY_types']} "
                  f"flip {v.get('flip_MSA', float('nan')):.2f}/{v.get('flip_EGY', float('nan')):.2f} "
                  f"diff {v.get('flip_diff')}  max|SMD| {max(abs(x) for x in s.values()):.2f} "
                  f"(lex {s['lex_own']:+.2f} host_mean {s['host_mean']:+.2f} host_last {s['host_last']:+.2f} "
                  f"func {s['is_function']:+.2f} nleft {s['n_left_matrix']:+.2f})")
            if "one_item_per_type" in v:
                o = v["one_item_per_type"]
                print(f"      one item per type ({o['repeats']}x, ~{o['mean_n_pairs']:.0f} pairs): diff {o['flip_diff_mean']:+.2f} "
                      f"[{o['flip_diff_2.5_97.5'][0]:+.2f}, {o['flip_diff_2.5_97.5'][1]:+.2f}], >0 in "
                      f"{100 * o['share_of_repeats_with_diff>0']:.0f}% of repeats")
    out = PROJECT_DIR / "results" / "exp07b_matching_sensitivity.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
