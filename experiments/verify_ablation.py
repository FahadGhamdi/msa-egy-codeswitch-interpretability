"""
Audit of the head-ablation hooks used by exp06 (readout.AblationHooks) on the REAL model.

Checks, for the top-5 heads of exp02 (which span several layers), on a few code-switched items
and for each intervention site (target / later / span):
  1. every ablated head slice, at every ablated position, equals the mean of ITS OWN layer
     (the exp05 late-binding bug would have used another layer's mean);
  2. the per-layer means really differ between layers (so check 1 is informative);
  3. nothing before the intervention point changes (all layers, positions < first ablated position);
  4. in the lowest ablated layer, non-ablated heads at the ablated positions are untouched;
  5. "later" ablation leaves the representation of the embedded word itself unchanged.

Usage
    python verify_ablation.py --model google/gemma-2-9b
Writes results/verify_ablation/<model>.json and prints PASS/FAIL per check.
"""
import argparse
import json

import numpy as np
import torch

from readout import (LANG, PROJECT_DIR, AblationHooks, collect_head_inputs, get_layers, head_geometry,
                     load_rows_with_ids, mono_sentences, readout_layer, type_classes_mono, with_bos, word_ids,
                     word_positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--n_items", type=int, default=5)
    ap.add_argument("--ref_tweets", type=int, default=40)
    args = ap.parse_args()
    slug = args.model.split("/")[-1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    layers = get_layers(model)
    n_heads, hd = head_geometry(model)
    R = readout_layer(len(layers))
    s2 = json.loads((PROJECT_DIR / "results" / "exp02" / slug / "summary.json").read_text())
    top = [(int(h["layer"]), int(h["head"])) for h in s2["top_heads"]][:5]
    Ls = sorted({L for L, _ in top})
    print(f"top heads {['L%dH%d' % t for t in top]}  (layers {Ls})")

    rows = load_rows_with_ids()
    plain = [(ws, ls) for _, ws, ls in rows]
    ref = {L: [] for L in Ls}
    for v, lab in [("MSA", "lang1"), ("EGY", "lang2")]:
        for ws, ls in mono_sentences(plain, v)[: args.ref_tweets]:
            ids, pos, _ = word_positions(tok, ws, ls, lab)
            if pos:
                cap = collect_head_inputs(model, ids, pos, Ls, device)
                for L in Ls:
                    ref[L].append(cap[L])
    ref_mean = {L: torch.cat(ref[L]).mean(0).to(device) for L in Ls}

    m, e = type_classes_mono(plain)
    types = {"MSA": set(m), "EGY": set(e)}
    items = []
    for sid, ws, ls in rows:
        langs = [LANG[l] for l in ls if l in LANG]
        if "MSA" not in langs or "EGY" not in langs:
            continue
        for i in range(1, len(ws) - 2):
            if ls[i] in LANG and ws[i] in types[LANG[ls[i]]]:
                left = tok(" ".join(ws[:i]), add_special_tokens=False).input_ids
                ids = with_bos(tok, left + word_ids(tok, ws[i]))
                t = len(ids) - 1
                for j in range(i + 1, i + 3):
                    ids = ids + word_ids(tok, ws[j])
                items.append((ids, t))
                break
        if len(items) >= args.n_items:
            break

    def capture(ids, plan=None):
        """o_proj inputs (after any ablation hook) for all positions, all layers in Ls, + residual at R."""
        cap = {}
        ctx = AblationHooks(model, [plan], hd, ref_mean) if plan else None
        if ctx:
            ctx.__enter__()                              # ablation hooks registered FIRST ...
        hooks = [layers[L].self_attn.o_proj.register_forward_pre_hook(      # ... so these see their output
            lambda mod, a, L=L: cap.__setitem__(L, a[0][0].float().cpu())) for L in Ls]
        try:
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
        finally:
            for h in hooks:
                h.remove()
            if ctx:
                ctx.__exit__(None, None, None)
        return cap, hs[R][0].float().cpu()

    res = {"model": args.model, "top_heads": [f"L{L}H{h}" for L, h in top], "checks": {}}
    means = {L: ref_mean[L].float().cpu() for L in Ls}
    res["checks"]["2_layer_means_differ"] = bool(all(
        not torch.allclose(means[a], means[b], atol=1e-2) for a in Ls for b in Ls if a < b))
    ok1, ok3, ok4, ok5 = True, True, True, True
    worst = {"1": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}
    for ids, t in items:
        base, base_R = capture(ids)
        for where in ["target", "later", "span"]:
            pos = {"target": [t], "later": list(range(t + 1, len(ids))), "span": list(range(t, len(ids)))}[where]
            plan = ({"target": t, "later": slice(t + 1, None), "span": slice(t, None)}[where], top)
            abl, abl_R = capture(ids, plan)
            first = pos[0]
            for L, h in top:                                               # check 1
                sl = slice(h * hd, (h + 1) * hd)
                exp = means[L][sl].to(torch.bfloat16).float()
                d = float((abl[L][pos][:, sl] - exp).abs().max())
                worst["1"] = max(worst["1"], d)
                ok1 &= d < 1e-2
            for L in Ls:                                                   # check 3
                d = float((abl[L][:first] - base[L][:first]).abs().max()) if first > 0 else 0.0
                worst["3"] = max(worst["3"], d)
                ok3 &= d < 1e-3
            L0 = Ls[0]                                                     # check 4
            keep = [g for g in range(n_heads) if (L0, g) not in top]
            for g in keep:
                sl = slice(g * hd, (g + 1) * hd)
                d = float((abl[L0][pos][:, sl] - base[L0][pos][:, sl]).abs().max())
                worst["4"] = max(worst["4"], d)
                ok4 &= d < 1e-3
            if where == "later":                                           # check 5
                d = float((abl_R[t] - base_R[t]).abs().max())
                worst["5"] = max(worst["5"], d)
                ok5 &= d < 1e-3
    res["checks"].update({"1_ablated_slices_equal_own_layer_mean": bool(ok1),
                          "3_positions_before_intervention_unchanged": bool(ok3),
                          "4_non_ablated_heads_untouched_in_first_layer": bool(ok4),
                          "5_later_ablation_leaves_embedded_word_unchanged": bool(ok5),
                          "max_abs_deviation": worst, "n_items": len(items)})
    out = PROJECT_DIR / "results" / "verify_ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(res, indent=2))
    for k, v in res["checks"].items():
        if isinstance(v, bool):
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("  max abs deviation:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"saved {out / (slug + '.json')}")


if __name__ == "__main__":
    main()
