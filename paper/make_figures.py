"""
Regenerates every data figure of the manuscript from the saved result files.

    python code/make_figures.py --results ../results --out figures

All numbers are read from the result files written by the experiment scripts
(exp01-exp08); nothing is typed in by hand. Figure 1 (examples) uses no data.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

SHORT = {"Qwen3-8B": "Qwen3", "ALLaM-7B": "ALLaM", "Gemma-2-9B": "Gemma-2", "Fanar-1-9B": "Fanar"}
MODELS = [("Qwen3-8B-Base", "Qwen3-8B", 36), ("ALLaM-7B-Instruct-preview", "ALLaM-7B", 32),
          ("gemma-2-9b", "Gemma-2-9B", 42), ("Fanar-1-9B", "Fanar-1-9B", 42)]
RUNS = [("Qwen3-8B-Base", "Qwen3-8B"), ("ALLaM-7B-Instruct-preview", "ALLaM-7B"), ("gemma-2-9b", "Gemma-2-9B"),
        ("Fanar-1-9B", "Fanar-1-9B"), ("Fanar-1-9B__heads-gemma-2-9b", "Fanar (Gemma heads)")]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
C_MSA, C_EGY = ORANGE, BLUE                                    # fixed identity: MSA word = orange, EGY word = blue
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 1
def fig_examples(out):
    import arabic_reshaper
    from bidi.algorithm import get_display
    ar = font_manager.FontProperties(family="Noto Naskh Arabic", size=15)

    def shape(t):
        return get_display(arabic_reshaper.reshape(t))

    ex = [("(a) MSA word embedded in an Egyptian matrix sentence", ["أنا", "مش", "عارف", "هو", "سوف", "يسافر", "إمتى"], 4,
           "ana miš ʕārif huwwa  sawfa  yisāfir imta",
           "'I don't know when he will travel.'",
           "MSA future particle sawfa (Egyptian: ḥa-/ha-) inside an Egyptian frame (miš, imta)",
           "MSA words in Egyptian frames are read as Egyptian by the probe in about half of cases", C_MSA),
          ("(b) Egyptian word embedded in an MSA matrix sentence", ["أعلنت", "الوزارة", "أن", "الامتحانات", "هتبدأ", "الأسبوع", "القادم"], 4,
           "aʕlanat al-wizāra anna l-imtiḥānāt  hatibdaʔ  al-usbūʕ al-qādim",
           "'The ministry announced that the exams will start next week.'",
           "Egyptian future prefix ha- (MSA: sa-tabdaʔ) inside an MSA frame",
           "Egyptian words in MSA frames keep their Egyptian reading in more than 90% of cases", C_EGY)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0))
    for ax, (title, words, k, translit, gloss, note, reading, col) in zip(axes, ex):
        ax.axis("off")
        for tt in [ax.text(0.0, 0.95, title, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes),
                   ax.text(0.0, 0.36, translit, fontsize=8, style="italic", color=INK2, transform=ax.transAxes),
                   ax.text(0.0, 0.2, gloss, fontsize=8, transform=ax.transAxes),
                   ax.text(0.0, 0.1, f"{note};\n{reading}.", fontsize=7.5, color=INK2, va="top", transform=ax.transAxes)]:
            tt.set_in_layout(False)
    fig.text(0.0, -0.06, "Constructed examples for illustration (not taken from the corpus).", fontsize=7, color=INK2)
    fig.tight_layout(h_pad=3.0)          # fix the axes geometry BEFORE measuring the Arabic words
    for ax, (title, words, k, translit, gloss, note, reading, col) in zip(axes, ex):
        # right-to-left layout of the Arabic words, the embedded word highlighted
        x = 0.98
        for i, w in enumerate(words):
            t = ax.text(x, 0.62, shape(w), fontproperties=ar, ha="right", va="center", transform=ax.transAxes,
                        color=col if i == k else INK,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2) if i == k else None)
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x0 - 0.018
    save(fig, out, "fig1_examples")


# --------------------------------------------------------------------------- Figure 3
def fig_probe_depth(R, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
    series = [("mono_test", "held-out monolingual", BLUE), ("unseen_types", "unseen word types", ORANGE),
              ("ambiguous_forms", "ambiguous forms", AQUA)]
    for ax, (slug, name, nl) in zip(axes, MODELS):
        d = pd.read_csv(R / "exp01" / slug / "A_probe_by_layer.csv")
        x = d.layer / nl
        for col, lab, c in series:
            ax.plot(x, d[col], color=c, lw=1.6, label=lab)
            ax.axhline(d[f"baseline_{col}"].iloc[0], color=c, lw=0.9, ls=(0, (3, 2)))
        ax.set_title(name)
        ax.set_xlabel("relative depth")
        ax.set_ylim(0.45, 1.0)
        style(ax)
    axes[0].set_ylabel("probe accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), handlelength=1.6)
    fig.text(0.5, -0.09, "Dashed lines: lexical (majority-label) baseline for the same evaluation set.", ha="center",
             fontsize=7, color=INK2)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig3_probe_depth")


# --------------------------------------------------------------------------- Figure 4
def fig_context_effect(R, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
    for ax, (slug, name, nl) in zip(axes, MODELS):
        d = pd.read_csv(R / "exp01" / slug / "F_context_effect.csv")
        d = d[d.types == "unambiguous"]
        for cond, lab, c in [("MSA embedded in EGY", "MSA word in EGY sentence", C_MSA),
                             ("EGY embedded in MSA", "EGY word in MSA sentence", C_EGY)]:
            s = d[d.condition == cond].sort_values("layer")
            x = s.layer / nl
            ax.fill_between(x, s.ci_lo, s.ci_hi, color=c, alpha=0.18, lw=0)
            ax.plot(x, s.effect, color=c, lw=1.6, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(name)
        ax.set_xlabel("relative depth")
        style(ax)
    axes[0].set_ylabel("Δ P(own variety)\nvs. same type, monolingual")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig4_context_effect")


# --------------------------------------------------------------------------- Figure 5
def fig_flip_rates(R, out):
    rows = []
    for slug, name, nl in MODELS:
        s = json.loads((R / "exp04" / slug / "summary.json").read_text())
        rb = s["robustness"][f"L{s['readout_layer']}"]
        rows.append((name, rb["flip_rate_natural"], rb["flip_rate_random_matrix_prefix"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    w = 0.36
    for ax, key, title in [(axes[0], 1, "(a) natural left context"), (axes[1], 2, "(b) random matrix-variety prefix")]:
        x = np.arange(len(rows))
        for j, (v, lab, c) in enumerate([("MSA", "MSA word in EGY sentence", C_MSA), ("EGY", "EGY word in MSA sentence", C_EGY)]):
            vals = [100 * r[key][v] for r in rows]
            bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=c, label=lab)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=6.8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_title(title)
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("flip rate (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig5_flip_rates")


# --------------------------------------------------------------------------- Figure 6
def fig_matching(R, out):
    m7b = json.loads((R / "exp07b_matching_sensitivity.json").read_text())
    e8 = json.loads((R / "exp08_final_analyses.json").read_text())["C"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    specs = [("raw", "raw difference (all items)", INK2, "o"),
             ("m1", "matched: function/content, lexical strength, host mean, matrix words left", BLUE, "s"),
             ("m1adj", "same matched sample, adjusted for length, frequency, sub-tokens", ORANGE, "D"),
             ("m2", "matched: additionally last host word", AQUA, "^"),
             ("one", "one item per word type (50 re-draws; 2.5-97.5% sensitivity range)", NEUTRAL, "v")]
    yl, ys = [], []
    for i, (slug, name, nl) in enumerate(MODELS):
        v = m7b[slug]
        vals = {"raw": (100 * (v["raw_flip"]["MSA"] - v["raw_flip"]["EGY"]), None, None),
                "m1": tuple(100 * np.array(v["exact_func__lex_host_mean_nleft"]["flip_diff"])),
                "m1adj": (100 * e8[slug]["adjusted_len_freq_subtok"]["coef_is_MSA"],
                          100 * e8[slug]["adjusted_len_freq_subtok"]["ci"][0], 100 * e8[slug]["adjusted_len_freq_subtok"]["ci"][1]),
                "m2": tuple(100 * np.array(v["exact_func__lex_both_hosts_nleft"]["flip_diff"])),
                "one": (100 * v["exact_func__lex_host_mean_nleft"]["one_item_per_type"]["flip_diff_mean"],
                        *[100 * q for q in v["exact_func__lex_host_mean_nleft"]["one_item_per_type"]["flip_diff_2.5_97.5"]])}
        for j, (k, lab, c, mk) in enumerate(specs):
            y = i * 6 + j
            mean, lo, hi = vals[k]
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=c, lw=1.6)
            ax.plot(mean, y, mk, color=c, ms=5, mec="white", mew=0.6, label=lab if i == 0 else None)
        yl.append(name)
        ys.append(i * 6 + 2)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlabel("flip-rate difference, MSA-in-EGY minus EGY-in-MSA (percentage points)")
    style(ax, "x")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, out, "fig7_matching")


# --------------------------------------------------------------------------- Figure 7
def fig_host(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    w = 0.36
    for ax, key, title in [(axes[0], "host_last", "(a) word immediately preceding the embedded word"),
                           (axes[1], "host_mean_other_words", "(b) mean over the other host words")]:
        x = np.arange(len(MODELS))
        for j, (v, lab, c) in enumerate([("MSA", "EGY host of an MSA word", C_MSA), ("EGY", "MSA host of an EGY word", C_EGY)]):
            ms, lo, hi = [], [], []
            for slug, name, nl in MODELS:
                h = json.loads((R / "exp07" / slug / "summary.json").read_text())["host"][v][key]
                ms.append(h[0]); lo.append(h[0] - h[1]); hi.append(h[2] - h[0])
            ax.bar(x + (j - 0.5) * (w + 0.02), ms, w, color=c, label=lab, yerr=[lo, hi],
                   error_kw=dict(ecolor=INK, lw=0.8, capsize=0))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in MODELS])
        ax.set_title(title)
        style(ax)
    axes[0].set_ylabel("probe logit toward\nthe matrix variety")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig6_host")


# --------------------------------------------------------------------------- Figure 8
def fig_patching(R, out):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), sharey="row")
    for col, (slug, name, nl) in enumerate(MODELS):
        p1 = pd.read_csv(R / "exp02" / slug / "P1_context_effect_by_layer.csv")
        ax = axes[0, col]
        for tgt, lab, c in [("MSA", "MSA target", C_MSA), ("EGY", "EGY target", C_EGY)]:
            a = p1[(p1.target == tgt) & (p1.prefix == "EGY")].sort_values("layer").set_index("layer")
            b = p1[(p1.target == tgt) & (p1.prefix == "MSA")].sort_values("layer").set_index("layer")
            eff = (a.s - b.s)
            ax.plot(eff.index / nl, eff.values, color=c, lw=1.6, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(name)
        style(ax)
        p2 = pd.read_csv(R / "exp02" / slug / "P2_patching_by_layer.csv")
        ax = axes[1, col]
        for comp, lab, c in [("resid", "residual stream", BLUE), ("attn", "attention output", ORANGE), ("mlp", "MLP output", AQUA)]:
            s = p2[(p2.target == "MSA") & (p2.component == comp)].sort_values("layer")
            ax.plot(s.layer / nl, s.restoration, color=c, lw=1.4, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        style(ax)
    axes[0, 0].set_ylabel("context effect\ns(EGY prefix) − s(MSA prefix)")
    axes[1, 0].set_ylabel("restoration fraction\n(MSA targets)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[1, 0].legend(frameon=False, loc="upper left")
    fig.supxlabel("relative depth (top: layer read; bottom: patched layer)", fontsize=8.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, out, "fig8_patching")


def fig_heads(R, out):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#8fb7e8", BLUE, "#123a6b"])
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 4.6))
    for ax, (slug, name, nl) in zip(axes, MODELS):
        h = pd.read_csv(R / "exp02" / slug / "P3_heads.csv")
        piv = h.pivot(index="layer", columns="head", values="mean")
        vmax = float(np.nanmax(piv.values))
        im = ax.imshow(np.clip(piv.values, 0, None), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([f"L{L}" for L in piv.index])
        ax.set_xticks(range(0, piv.shape[1], 4))
        ax.set_title(name, loc="left", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
    axes[-1].set_xlabel("attention head")
    fig.text(0.0, 1.0, "Δ variety score at the read-out layer when a single head's output is patched (MSA targets)",
             fontsize=7.5, color=INK2)
    fig.tight_layout(h_pad=0.6)
    save(fig, out, "fig9_heads")


# --------------------------------------------------------------------------- Figure 10
def fig_link(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    names, d, lo, hi, r0m, r0e, rim, rie = [], [], [], [], [], [], [], []
    for slug, name, nl in MODELS:
        L = json.loads((R / "exp06" / slug / "summary.json").read_text())["probe_to_prediction_link"]
        v = L["MSA_in_EGY"]["D_to_matrix_flipped_minus_not"]
        names.append(name); d.append(v[0]); lo.append(v[0] - v[1]); hi.append(v[2] - v[0])
        r0m.append(L["MSA_in_EGY"]["spearman(-own_logit, D_to_matrix)"][0])
        r0e.append(L["EGY_in_MSA"]["spearman(-own_logit, D_to_matrix)"][0])
        rim.append(-L["MSA_in_EGY"]["spearman(d_own_logit, d_D) under top5"][0])
        rie.append(-L["EGY_in_MSA"]["spearman(d_own_logit, d_D) under top5"][0])
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, d, 0.5, color=C_MSA, yerr=[lo, hi], error_kw=dict(ecolor=INK, lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names])
    ax.set_ylabel("D: flipped − not flipped (nats)")
    ax.set_title("(a) MSA words in EGY sentences: next-token\npreference of items the probe reads as flipped")
    ax.axhline(0, color=INK2, lw=0.6)
    style(ax)
    ax = axes[1]
    w = 0.2
    for j, (vals, lab, c, hatch) in enumerate([(r0m, "baseline ρ, MSA-in-EGY", C_MSA, None), (r0e, "baseline ρ, EGY-in-MSA", C_EGY, None),
                                               (rim, "−ρ under ablation, MSA-in-EGY", C_MSA, "////"),
                                               (rie, "−ρ under ablation, EGY-in-MSA", C_EGY, "////")]):
        ax.bar(x + (j - 1.5) * (w + 0.01), vals, w, color=c if hatch is None else "white", edgecolor=c, hatch=hatch,
               lw=0.8, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Spearman correlation")
    ax.set_title("(b) probe reading vs. next-token preference")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    style(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig10_link")


# --------------------------------------------------------------------------- Figure 11
def fig_ablation(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    groups = [("MSA_in_EGY", "MSA in EGY", C_MSA, "o"), ("EGY_in_MSA", "EGY in MSA", C_EGY, "o"),
              ("mono_MSA", "monolingual MSA", C_MSA, "s"), ("mono_EGY", "monolingual EGY", C_EGY, "s")]
    for ax, mtr, title in [(axes[0], "d_own_logit", "(a) Δ own-variety probe logit"),
                           (axes[1], "d_D_to_matrix", "(b) Δ next-token preference D")]:
        yt, yl = [], []
        for i, (run, name) in enumerate(RUNS):
            nd = pd.read_csv(R / "exp06" / run / "B_top5_vs_random_null.csv")
            nd = nd[(nd.ablation == "target") & (nd.measure == mtr)].set_index("group")
            for j, (g, lab, c, mk) in enumerate(groups):
                y = i * 5 + j
                r = nd.loc[g]
                ax.plot([r["random_2.5"], r["random_97.5"]], [y, y], color=NEUTRAL, lw=4, solid_capstyle="round",
                        label="random heads, 95% range (100 draws)" if (i == 0 and j == 0) else None)
                ax.plot(r["top5"], y, mk, color=c if mk == "o" else "white", mec=c, mew=1.2, ms=5,
                        label=f"top-5 heads: {lab}" if i == 0 else None)
            yt.append(i * 5 + 1.5); yl.append(name)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(yt); ax.set_yticklabels(yl)
        ax.set_title(title)
        style(ax, "x")
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.08, 1, 1))
    save(fig, out, "fig11_ablation")


# --------------------------------------------------------------------------- Figure 12
def fig_sites(R, out):
    """exp09: equal-count sites (a) and distance profile of the embedded-word vs later-site effects (b)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    sites = [("target", "embedded word (1 position)", NEUTRAL), ("next1", "next position only (1 position)", AQUA),
             ("later", "all later positions", BLUE)]
    tabs = {}
    for run, _ in RUNS:
        t = pd.DataFrame(json.loads((R / "exp09_sites" / run / "summary.json").read_text())["sites"])
        tabs[run] = t[t.group == "EGY_in_MSA"].set_index(["measure", "site"])
    ax = axes[0]
    for i, (run, name) in enumerate(RUNS):
        for j, (st, lab, c) in enumerate(sites):
            r = tabs[run].loc[("d_cont_lp_matrix", st)]
            y = i * 4 + j
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.6)
            ax.plot(r.net, y, "o", color=c, ms=5, mec="white", mew=0.6, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks([i * 4 + 1 for i in range(len(RUNS))]); ax.set_yticklabels([r[1] for r in RUNS])
    ax.invert_yaxis()
    ax.set_xlabel("Δ log p per token of following MSA words\n(top-5 heads minus random heads)")
    ax.set_title("(a) effect by intervention site")
    style(ax, "x")
    ax = axes[1]
    words = ["w1", "w2", "w3"]
    for k, (st, c, mk, dx) in enumerate([("target", NEUTRAL, "o", -0.12), ("later", BLUE, "s", 0.12)]):
        for i, (run, name) in enumerate(RUNS):
            xs, ys = [], []
            for wi, w in enumerate(words):
                r = tabs[run].loc[(f"d_cont_lp_matrix_{w}", st)]
                xs.append(wi + 1 + dx + (i - 2) * 0.035); ys.append(r.net)
            ax.plot(xs, ys, mk, color=c, ms=3.5, mec="white", mew=0.4, alpha=0.9,
                    label=("embedded word" if st == "target" else "all later positions") if i == 0 else None)
        means = [np.mean([tabs[r].loc[(f"d_cont_lp_matrix_{w}", st)].net for r, _ in RUNS]) for w in words]
        ax.plot([wi + 1 + dx for wi in range(len(words))], means, "-", color=c, lw=1.4)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["word 1", "word 2", "word 3"])
    ax.set_ylabel("Δ log p per token (net)")
    ax.set_title("(b) which following word is affected")
    style(ax, "y")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h + h2[1:], l + ["(b) all later positions; dots: runs, lines: mean"], frameon=False, fontsize=6.8,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    save(fig, out, "fig12_sites")


def fig_table1_arabic(out):
    """Arabic-script versions of the constructed examples in Table 1 (embedded word coloured)."""
    import arabic_reshaper
    from bidi.algorithm import get_display
    ar = font_manager.FontProperties(family="Noto Naskh Arabic", size=11)
    rows = [(["احنا", "رحنا", "المطعم", "ثم", "روّحنا", "على", "طول"], 3, C_MSA),
            (["الامتحان", "كان", "صعب", "جدًا", "لذلك", "محدش", "خلص"], 4, C_MSA),
            (["لم", "يتمكن", "الفريق", "من", "الفوز", "عشان", "الإصابات"], 5, C_EGY),
            (["أعلنت", "الوزارة", "أن", "الامتحانات", "هتبدأ", "الأسبوع", "القادم"], 4, C_EGY)]
    for n, (words, k, col) in enumerate(rows, 1):
        fig = plt.figure(figsize=(3.2, 0.2))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        fig.canvas.draw()
        x = 0.99
        exts = []
        for i, w in enumerate(words):
            t = ax.text(x, 0.5, get_display(arabic_reshaper.reshape(w)), fontproperties=ar, ha="right", va="center",
                        color=col if i == k else INK, fontweight="bold" if i == k else "normal", transform=ax.transAxes)
            fig.canvas.draw()
            e = t.get_window_extent()
            exts.append(e)
            x = e.transformed(ax.transAxes.inverted()).x0 - 0.025
        from matplotlib.transforms import Bbox
        bb = Bbox.union(exts).transformed(fig.dpi_scale_trans.inverted())
        bb = Bbox.from_extents(bb.x0 - 0.02, 0, bb.x1 + 0.02, 0.2)
        fig.savefig(out / f"t1_ex{n}.pdf", bbox_inches=bb)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_examples(out)
    fig_table1_arabic(out)
    fig_probe_depth(R, out)
    fig_context_effect(R, out)
    fig_flip_rates(R, out)
    fig_matching(R, out)
    fig_host(R, out)
    fig_patching(R, out)
    fig_heads(R, out)
    fig_link(R, out)
    fig_ablation(R, out)
    fig_sites(R, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
