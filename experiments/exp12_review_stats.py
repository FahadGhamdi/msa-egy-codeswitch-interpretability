"""
Experiment 12 (review): statistics recomputed from the saved LONG result files (no model runs).

  R1  Spearman correlations of the prediction link (exp06) with TWEET-CLUSTER bootstrap intervals and p-values
      (the paper previously reported scipy p-values that treat words as independent).
  R2  TOST sensitivity: equivalence verdicts with margins of 15%, 25% (paper) and 35% of the pooled effect,
      reported with three decimals and with a separate test of the difference against zero.
  R3  Number of independent tweets behind each sample (exp04 flip sample, exp06 ablation sample,
      site analysis sample).

    python exp12_review_stats.py            # writes results/review/review_long.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from readout import PROJECT_DIR, readout_layer

MODELS = {"Qwen3-8B-Base": 36, "ALLaM-7B-Instruct-preview": 32, "gemma-2-9b": 42, "Fanar-1-9B": 42}
RUNS = list(MODELS) + ["Fanar-1-9B__heads-gemma-2-9b"]
RES = PROJECT_DIR / "results"


def cluster_rho(x, y, g, n=2000, seed=0):
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[ok], y[ok], g[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.integers(0, len(uniq), len(uniq))])
        bs.append(spearmanr(x[pick], y[pick]).statistic)
    bs = np.array(bs)
    est = spearmanr(x, y).statistic
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(max(p, 1 / n)),
            int(len(x)), int(len(uniq))]


def tost_multi(a, b, fracs=(0.15, 0.25, 0.35), n=5000, seed=2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    pooled = abs(np.r_[a, b].mean())
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (n, len(a)))].mean(1) - b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    lo90, hi90 = np.percentile(d, [5, 95])
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    out = {"diff": float(a.mean() - b.mean()), "ci90": [float(lo90), float(hi90)], "ci95": [float(lo95), float(hi95)],
           "differs_from_zero_95": bool(lo95 > 0 or hi95 < 0), "n": [int(len(a)), int(len(b))]}
    for f in fracs:
        m = f * pooled
        out[f"margin_{int(f * 100)}"] = {"margin": float(m), "equivalent": bool(lo90 > -m and hi90 < m),
                                         "beyond_margin": bool(lo90 > m or hi90 < -m)}
    return out


def main():
    res = {"R1_link_cluster": {}, "R2_tost": {}, "R3_tweets": {}}
    for run in RUNS:
        f = RES / "exp06" / run / "B_long.csv"
        if not f.exists():
            print("missing", f)
            continue
        df = pd.read_csv(f)
        r = {}
        b0 = df[df.cls == "baseline"]
        t5 = df[df.cls == "top5"]
        for g in ["MSA_in_EGY", "EGY_in_MSA"]:
            d = b0[b0.group == g]
            r[f"{g}/baseline(-own_logit,D)"] = cluster_rho(-d.own_logit, d.D_to_matrix, d.sid)
            d = t5[t5.group == g]
            r[f"{g}/top5(d_own_logit,d_D)"] = cluster_rho(d.d_own_logit, d.d_D_to_matrix, d.sid)
        res["R1_link_cluster"][run] = r
        items = df.drop_duplicates("item")
        res["R3_tweets"].setdefault(run, {})["exp06"] = {
            g: [int((items.group == g).sum()), int(items[items.group == g].sid.nunique())] for g in items.group.unique()}
        lat = df[(df.cls == "top5_later") & (df.group == "EGY_in_MSA")].dropna(subset=["d_cont_lp_matrix"])
        res["R3_tweets"][run]["sites_EGY_in_MSA"] = [int(len(lat)), int(lat.sid.nunique())]
    for m, L in MODELS.items():
        R = readout_layer(L)
        t = {}
        p1 = RES / "exp02" / m / "P1_scores_long.csv"
        if p1.exists():
            s1 = pd.read_csv(p1)
            s1 = s1[s1.layer == R].pivot_table(index=["pair", "target"], columns="prefix", values="s").reset_index()
            s1["ctx"] = s1.EGY - s1.MSA
            t["minimal_pairs"] = tost_multi(s1[s1.target == "MSA"].ctx, s1[s1.target == "EGY"].ctx)
        y = RES / "exp04" / m / "Y_long.csv"
        if y.exists():
            yl = pd.read_csv(y)
            yl = yl[(yl.layer == R) & (yl.n_left_matrix >= 3)]
            for mtr in ["natural_pull", "unintegrated_pull"]:
                t[mtr] = tost_multi(yl[yl.own == "MSA"][mtr], yl[yl.own == "EGY"][mtr])
        res["R2_tost"][m] = t
    out = RES / "review" / "review_long.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:6000])
    print("saved", out)


if __name__ == "__main__":
    main()
