"""
Build one cross-model table from the results of exp01 / exp01b / exp02 / exp04.

Usage:  python compare_models.py
Output: results/comparison/models_table.csv  and  models_table.md
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]  # repository root
RES = PROJECT_DIR / "results"


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    slugs = sorted({p.name for exp in ["exp01", "exp02", "exp04"] if (RES / exp).exists()
                    for p in (RES / exp).iterdir() if p.is_dir()})
    rows = []
    for slug in slugs:
        r = {"model": slug}
        s1 = load_json(RES / "exp01" / slug / "summary.json")
        if s1:
            r["layers"] = s1["n_layers_incl_emb"] - 1
            for k in ["mono_test", "unseen_types", "ambiguous_forms", "cs_words"]:
                r[f"A_{k}"] = round(s1["best_score"].get(k, float("nan")), 3)
                r[f"A_{k}_lexical"] = round(s1["lexical_baseline"].get(k, float("nan")), 3)
        f = RES / "exp01" / slug / "F_context_effect.csv"
        if f.exists():
            F = pd.read_csv(f)
            last = F.layer.max()
            for cond, key in [("MSA embedded in EGY", "F_MSA_in_EGY"), ("EGY embedded in MSA", "F_EGY_in_MSA")]:
                d = F[(F.condition == cond) & (F.types == "unambiguous") & (F.layer == last)]
                if len(d):
                    r[key] = f"{d.effect.iloc[0]:+.2f} [{d.ci_lo.iloc[0]:+.2f},{d.ci_hi.iloc[0]:+.2f}]"
        s2 = load_json(RES / "exp02" / slug / "summary.json")
        if s2:
            k = [x for x in s2 if x.startswith("context_sensitivity")][0]
            r["P1_ctx_MSA"] = round(s2[k]["MSA"]["mean"], 2)
            r["P1_ctx_EGY"] = round(s2[k]["EGY"]["mean"], 2)
            r["P2_top_attn_layers"] = ",".join(map(str, s2["top_attn_layers_by_patching"]))
            r["P2_peak_mlp_layer"] = s2["peak_restoration"]["MSA_mlp"]["layer"]
            r["P3_top_heads"] = ", ".join(f"L{h['layer']}H{h['head']}" for h in s2["top_heads"][:3])
        s4 = load_json(RES / "exp04" / slug / "summary.json")
        if s4:
            R = s4["readout_layer"]
            b = s4.get(f"matrix_left>=3@L{R}", {})
            for m in ["natural_pull", "unintegrated_pull", "target_own_logit"]:
                if b.get(m):
                    r[f"Y_{m}_MSA"] = round(b[m]["MSA_in_EGY"], 2)
                    r[f"Y_{m}_EGY"] = round(b[m]["EGY_in_MSA"], 2)
                    r[f"Y_{m}_diff"] = f"{b[m]['difference']:+.2f} [{b[m]['ci_lo']:+.2f},{b[m]['ci_hi']:+.2f}]"
            rob = s4.get("robustness", {}).get(f"L{R}")
            if rob:
                r["Y_own_sd_MSA"] = round(rob["target_own_in_sd_units"]["MSA"], 2)
                r["Y_own_sd_EGY"] = round(rob["target_own_in_sd_units"]["EGY"], 2)
                r["Y_flip_nat_MSA"] = f"{100 * rob['flip_rate_natural']['MSA']:.0f}%"
                r["Y_flip_nat_EGY"] = f"{100 * rob['flip_rate_natural']['EGY']:.0f}%"
                r["Y_flip_rand_MSA"] = f"{100 * rob['flip_rate_random_matrix_prefix']['MSA']:.0f}%"
                r["Y_flip_rand_EGY"] = f"{100 * rob['flip_rate_random_matrix_prefix']['EGY']:.0f}%"
        rows.append(r)

    df = pd.DataFrame(rows).set_index("model").T
    out = RES / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "models_table.csv")
    try:
        (out / "models_table.md").write_text(df.to_markdown())
    except ImportError:
        pass
    print(df.to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
