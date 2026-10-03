"""
Review analyses computed from the saved result files only (no model runs):
  S1  local insertions: flip rates for embedded words with no word of their own variety on their left
      (the whole left context available to the model is of the matrix variety), with a tweet-cluster
      bootstrap interval for the MSA-in-EGY minus EGY-in-MSA difference;
  S2  number of independent tweets behind the flip-rate / regression samples;
  S3  Holm correction across the five paired later-minus-embedded site comparisons.

    python code/review_analyses.py --results ../results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Qwen3-8B-Base", "ALLaM-7B-Instruct-preview", "gemma-2-9b", "Fanar-1-9B"]
RUNS = MODELS + ["Fanar-1-9B__heads-gemma-2-9b"]


def cluster_boot_diff(d, n=5000, seed=0):
    """MSA flip rate minus EGY flip rate, resampling tweets (sid) within each group."""
    rng = np.random.default_rng(seed)
    out = []
    grp = {v: [g.flip.values for _, g in d[d.own == v].groupby("sid")] for v in ["MSA", "EGY"]}
    for _ in range(n):
        m = []
        for v in ["MSA", "EGY"]:
            cl = grp[v]
            pick = rng.integers(0, len(cl), len(cl))
            m.append(np.concatenate([cl[i] for i in pick]).mean())
        out.append(m[0] - m[1])
    out = np.array(out)
    est = d[d.own == "MSA"].flip.mean() - d[d.own == "EGY"].flip.mean()
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return [float(est), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(max(p, 1 / n))]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    R = Path(ap.parse_args().results)
    res = {"S1_local_insertions": {}, "S2_tweets": {}}
    for m in MODELS:
        d = pd.read_csv(R / "exp07" / m / "L_items.csv")
        loc = d[d.n_left_own == 0]
        g = loc.groupby("own").agg(n=("flip", "size"), tweets=("sid", "nunique"), flip=("flip", "mean"))
        res["S1_local_insertions"][m] = {
            "flip": {v: float(g.loc[v, "flip"]) for v in ["MSA", "EGY"]},
            "n": {v: int(g.loc[v, "n"]) for v in ["MSA", "EGY"]},
            "tweets": {v: int(g.loc[v, "tweets"]) for v in ["MSA", "EGY"]},
            "diff": cluster_boot_diff(loc),
            "all_items_diff": cluster_boot_diff(d, seed=1)}
        h = d.dropna(subset=["host_mean"])
        res["S2_tweets"][m] = {
            "flip_sample": {v: [int((d.own == v).sum()), int(d[d.own == v].sid.nunique())] for v in ["MSA", "EGY"]},
            "regression_sample": {v: [int((h.own == v).sum()), int(h[h.own == v].sid.nunique())] for v in ["MSA", "EGY"]}}
    e8 = json.loads((R / "exp08_final_analyses.json").read_text())["A_B"]
    ps = np.array([e8[r]["EGY_in_MSA/d_cont_lp_matrix"]["later_minus_source_net"][3] for r in RUNS])
    res["S3_paired_sites_holm"] = {r: {"p": float(p), "p_holm": float(a)} for r, p, a in zip(RUNS, ps, holm(ps))}
    out = R / "review" / "review_saved.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
