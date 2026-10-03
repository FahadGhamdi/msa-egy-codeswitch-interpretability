"""
Regenerates every results table of the manuscript (LaTeX fragments in tables/) from the saved
result files, so that no number in the tables is transcribed by hand.

    python code/make_tables.py --results ../results --out tables
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = [("Qwen3-8B-Base", "Qwen3-8B"), ("ALLaM-7B-Instruct-preview", "ALLaM-7B"),
          ("gemma-2-9b", "Gemma-2-9B"), ("Fanar-1-9B", "Fanar-1-9B")]
RUNS = MODELS + [("Fanar-1-9B__heads-gemma-2-9b", "Fanar (Gemma heads)")]


def J(p):
    return json.loads(Path(p).read_text())


def f(x, d=2, sign=False):
    if round(x, d) == 0:
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def ci(m, lo, hi, d=2, sign=True):
    return f"{f(m, d, sign)} [{f(lo, d)}, {f(hi, d)}]"


def pc(x, d=0):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(100 * x)).quantize(q, rounding=ROUND_HALF_UP))


APPENDIX_TABLES = {"tab_wordclass", "tab_exploratory"}


def write(out, name, body):
    if name in APPENDIX_TABLES:                      # keep appendix tables under their headings
        body = body.replace("\\begin{table}[!htbp]", "\\begin{table}[H]")
    (out / f"{name}.tex").write_text(body)


def table_representation(R, out):
    rows = []
    for slug, name in MODELS:
        s1 = J(R / "exp01" / slug / "summary.json")
        F = pd.read_csv(R / "exp01" / slug / "F_context_effect.csv")
        F = F[F.types == "unambiguous"]
        Lmax = F.layer.max()
        fm = F[(F.condition == "MSA embedded in EGY") & (F.layer == Lmax)].iloc[0]
        fe = F[(F.condition == "EGY embedded in MSA") & (F.layer == Lmax)].iloc[0]
        s4 = J(R / "exp04" / slug / "summary.json")
        Rl = s4["readout_layer"]
        rb = s4["robustness"][f"L{Rl}"]
        own = s4[f"matrix_left>=3@L{Rl}"]["target_own_logit"]
        b = s1["best_score"]
        rows.append(f"{name} & {b['mono_test']:.2f} / {b['unseen_types']:.2f} / {b['ambiguous_forms']:.2f} & "
                    f"{f(fm.effect, 2)} / {f(fe.effect, 2)} & {pc(rb['flip_rate_natural']['MSA'])} / {pc(rb['flip_rate_natural']['EGY'])} & "
                    f"{pc(rb['flip_rate_random_matrix_prefix']['MSA'])} / {pc(rb['flip_rate_random_matrix_prefix']['EGY'])} & "
                    f"{own['MSA_in_EGY']:.1f} / {own['EGY_in_MSA']:.1f} \\\\")
    lex = J(R / "exp01" / MODELS[0][0] / "summary.json")["lexical_baseline"]
    body = (r"""\begin{table}[!htbp]
\caption{Representation of embedded words in the four models. Probe accuracy is the best layer on held-out monolingual words, unseen word types and ambiguous forms (lexical majority-label baselines: """
            + f"{lex['mono_test']:.2f} / {lex['unseen_types']:.2f} / {lex['ambiguous_forms']:.2f}"
            + r"""). Type-matched effect: change in $P(\text{own variety})$ at the last layer relative to the same unambiguous word type in monolingual text. Flip rates and lexical strength (own-variety logit of the word after a random prefix of its own variety) are read at layer $R$ for embedded words with at least three matrix-variety words on their left. Each cell gives MSA-in-EGY / EGY-in-MSA except the first column.\label{tab:representation}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCCCC}
\toprule
\textbf{Model} & \textbf{Probe acc.} & \textbf{Type-matched effect} & \textbf{Flip, natural (\%)} & \textbf{Flip, random prefix (\%)} & \textbf{Lexical strength} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_representation", body)


def table_regression(R, out):
    order = ["M0 variety only", "M1 + lexical strength", "M2 + word covariates", "M3 + host (mean of other words)",
             "M4 + host (last word)", "M5 + both hosts"]
    labels = ["variety only", "+ lexical strength", "+ word covariates", "+ host (mean of other words)",
              "+ host (last word)", "+ both host measures"]
    cols = {}
    for slug, name in MODELS:
        reg = J(R / "exp07" / slug / "summary.json")["regressions"]
        d = {r["model"]: r for r in reg if r.get("outcome") == "flip"}
        cols[name] = d
    lines = []
    for o, lab in zip(order, labels):
        cells = []
        for _, name in MODELS:
            r = cols[name][o]
            star = "" if r["p"] < 0.05 else "$^{\\dagger}$"
            cells.append(f"{100 * r['avg_marginal_effect_is_MSA']:.1f} ({'$<$0.001' if r['p'] < 0.001 else f'{r[chr(112)]:.3f}'}){star}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    n = J(R / "exp07" / MODELS[0][0] / "summary.json")["regressions"][0]["n"]
    body = (r"""\begin{table}[!htbp]
\caption{Logistic regression of the flip indicator on the variety of the embedded word. Cells give the average marginal effect of the word being MSA (percentage points of flip probability) and, in parentheses, the $p$-value of its coefficient with tweet-clustered standard errors. Word covariates: lexical strength, log frequency, characters, sub-tokens, function word, matrix words on the left (all $z$-scored except the function-word indicator). Word covariates include lexical strength; the last three rows add the host measures to the word-covariate model separately and jointly. The decline across rows is a reduction of the adjusted gap, not a causal decomposition.\label{tab:regression}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCC}
\toprule
\textbf{Specification} & """ + " & ".join(f"\\textbf{{{m[1]}}}" for m in MODELS) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write(out, "tab_regression", body)


def table_matching(R, out):
    m = J(R / "exp07b_matching_sensitivity.json")
    e8 = J(R / "exp08_final_analyses.json")["C"]
    rows = []
    for slug, name in MODELS:
        a = m[slug]["exact_func__lex_host_mean_nleft"]
        b = m[slug]["exact_func__lex_both_hosts_nleft"]
        adj = e8[slug]["adjusted_len_freq_subtok"]
        o = a["one_item_per_type"]
        rows.append(f"{name} & {a['n_pairs']} ({a['n_MSA_types']}/{a['n_EGY_types']}) & {pc(a['flip_MSA'])} / {pc(a['flip_EGY'])} & "
                    f"{pc(a['flip_diff'][0])} [{pc(a['flip_diff'][1])}, {pc(a['flip_diff'][2])}] & "
                    f"{pc(adj['coef_is_MSA'])} [{pc(adj['ci'][0])}, {pc(adj['ci'][1])}] & "
                    f"{b['n_pairs']}: {pc(b['flip_diff'][0])} [{pc(b['flip_diff'][1])}, {pc(b['flip_diff'][2])}] & "
                    f"{pc(o['flip_diff_mean'])} [{pc(o['flip_diff_2.5_97.5'][0])}, {pc(o['flip_diff_2.5_97.5'][1])}] \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Matched comparisons (flip rates in \%). Matching: exact on function vs.\ content word; nearest neighbour without replacement on lexical strength, mean host strength and number of matrix words on the left (caliper 0.35 SD per covariate). Differences are MSA-in-EGY minus EGY-in-MSA with bootstrap 95\% intervals. ``Adjusted'': linear probability model within the matched sample with length, log frequency and sub-token count, pair-clustered SEs. ``+ last host word'': the last host word added to the matching set. ``One item per type'': mean over 50 re-draws with one item per word type; the bracket is the 2.5--97.5\% range across re-draws (a sensitivity range, not a confidence interval).\label{tab:matching}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCCCCC}
\toprule
\textbf{Model} & \textbf{Pairs (types)} & \textbf{Flip MSA / EGY} & \textbf{Difference} & \textbf{Adjusted} & \textbf{+ last host word} & \textbf{One item per type} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_matching", body)


def table_tost(R, out):
    keys = [("exp02 minimal-pair context effect (MSA vs EGY targets)", "Minimal pairs: context effect (MSA vs.\\ EGY targets)"),
            ("exp04 unintegrated_pull (MSA-in-EGY vs EGY-in-MSA)", "Random matrix prefix: pull on the embedded words"),
            ("exp04 natural_pull (MSA-in-EGY vs EGY-in-MSA)", "Natural context: pull on the embedded words")]
    lines = []
    for k, lab in keys:
        cells = []
        for slug, name in MODELS:
            t = J(R / "exp07" / slug / "summary.json")["equivalence_TOST"][k]
            lo, hi, mg = t["ci90"][0], t["ci90"][1], t["margin"]
            rl = J(R / "review" / "review_long.json")["R2_tost"][slug]
            rk = {"exp02 minimal-pair context effect (MSA vs EGY targets)": "minimal_pairs",
                  "exp04 unintegrated_pull (MSA-in-EGY vs EGY-in-MSA)": "unintegrated_pull",
                  "exp04 natural_pull (MSA-in-EGY vs EGY-in-MSA)": "natural_pull"}[k]
            dz = rl[rk]["differs_from_zero_95"]
            side = "MSA" if t["diff"] > 0 else "EGY"
            if t["equivalent"]:
                verdict = "equivalent"
            elif lo > mg or hi < -mg:
                verdict = side + " larger, beyond margin"
            elif dz:
                verdict = side + " larger (95\\%); not beyond margin"
            else:
                verdict = "inconclusive"
            c95 = rl[rk]["ci95"]
            cells.append(f"{f(t['diff'], 3, True)} [{f(lo, 3)}, {f(hi, 3)}] \\{{{f(c95[0], 3)}, {f(c95[1], 3)}\\}} $\\pm${mg:.3f}: {verdict}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Equivalence tests (two one-sided tests). Each cell: difference (MSA minus EGY), bootstrap 90\% interval in brackets, two-sided 95\% interval in braces, pre-set margin ($\pm$25\% of the pooled mean effect on the same scale) and verdict: \emph{equivalent} if the 90\% interval lies within the margin; \emph{larger, beyond margin} if it lies entirely outside it; \emph{larger (95\%); not beyond margin} if the two-sided 95\% interval excludes zero (a difference at the 5\% level) but the 90\% interval overlaps the margin; otherwise \emph{inconclusive}. The margin rule (25\% of the pooled effect) was fixed before the tests were run; sensitivity to other margins is discussed in the text. Minimal pairs use the difference-of-means read-out on random unambiguous word types; the other rows use probe logits on the naturally embedded words.\label{tab:tost}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\raggedright\arraybackslash}X}
\begin{tabularx}{\fulllength}{p{3.6cm}CCCC}
\toprule
\textbf{Test} & """ + " & ".join(f"\\textbf{{{m[1]}}}" for m in MODELS) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_tost", body)


def table_patching(R, out):
    rows = []
    for slug, name in MODELS:
        s = J(R / "exp02" / slug / "summary.json")
        cs = s["context_sensitivity_at_R (s_EGYprefix - s_MSAprefix)"]
        heads = ", ".join(f"L{h['layer']}H{h['head']}" for h in s["top_heads"][:5])
        pr = s["peak_restoration"]
        rev = R / "exp02_reverse" / slug / "summary.json"
        if rev.exists():
            a = {(h["layer"], h["head"]) for h in s["top_heads"][:5]}
            b = {(h["layer"], h["head"]) for h in J(rev)["top_heads"][:5]}
            ov = f"{len(a & b)}/5"
        else:
            ov = "--"
        eg = R / "exp10_heads_EGY" / slug / "summary.json"
        if eg.exists():
            a = [(h["layer"], h["head"]) for h in s["top_heads"][:5]]
            b = [(h["layer"], h["head"]) for h in J(eg)["top_heads"][:5]]
            ov += f" & {len(set(a) & set(b))}/5" + ("$^{*}$" if a[0] == b[0] else "")
        else:
            ov += " & --"
        rows.append(f"{name} & {s['readout_layer']} & {ci(cs['MSA']['mean'], cs['MSA']['ci_lo'], cs['MSA']['ci_hi'], 2, False)} & "
                    f"{ci(cs['EGY']['mean'], cs['EGY']['ci_lo'], cs['EGY']['ci_hi'], 2, False)} & "
                    f"{', '.join(str(x) for x in s['top_attn_layers_by_patching'])} & {pr['MSA_mlp']['layer']} & {heads} & {ov} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Minimal pairs and activation patching (150 pairs per target variety, 12-token prefixes). Context effect: $s(\text{EGY prefix})-s(\text{MSA prefix})$ at the read-out layer $R$ (difference-of-means score; 0 = MSA mean, 1 = EGY mean), with 95\% bootstrap intervals. Attention layers are ranked by the effect of patching their output at the target position; the MLP column gives the layer with maximal MLP restoration for MSA targets. Reverse: overlap of the top-5 heads when the MSA-prefix run is used as the clean run. EGY targets: overlap of the top-5 heads when layers and heads are ranked on Egyptian instead of MSA target words ($^{*}$: same strongest head).\label{tab:patching}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lcCCCcCcc}
\toprule
\textbf{Model} & $R$ & \textbf{Context effect, MSA targets} & \textbf{Context effect, EGY targets} & \textbf{Top attention layers} & \textbf{MLP} & \textbf{Top-5 heads (rank order)} & \textbf{Reverse} & \textbf{EGY targets} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_patching", body)


def table_ablation(R, out):
    e8 = J(R / "exp08_final_analyses.json")["A_B"]
    rows = []
    for run, name in RUNS:
        fam = {x["test"]: x for x in e8[run]["primary_family_holm"]}
        s6 = J(R / "exp06" / run / "summary.json")
        sm = {(r["group"], r["condition"]): r for r in s6["summary"]}
        fb, fa = sm[("MSA_in_EGY", "baseline")]["flip_rate"], sm[("MSA_in_EGY", "top5")]["flip_rate"]

        def cell(k):
            x = fam[k]
            p = "$<$0.001" if x["p_holm"] < 0.001 else f"{x['p_holm']:.3f}"
            return f"{f(x['net'], 2, True)} [{f(x['lo'], 2)}, {f(x['hi'], 2)}] ({p})"
        rows.append(f"{name} & {cell('MSA_in_EGY/d_own_logit/top5')} & {cell('EGY_in_MSA/d_own_logit/top5')} & "
                    f"{cell('MSA_in_EGY/d_D_to_matrix/top5')} & {cell('EGY_in_MSA/d_D_to_matrix/top5')} & "
                    f"{pc(fb)} $\\rightarrow$ {pc(fa)} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Mean ablation of the top-5 patching heads at the embedded word, relative to 100 layer- and norm-matched random five-head sets. Each cell: item-level net contrast (top-5 effect minus the mean random effect on the same item), tweet-cluster bootstrap 95\% interval and Holm-adjusted $p$ within the run's five primary tests (the fifth test, the all-later-site effect on the following MSA words in the initial site design, is reported in Section~\ref{sec:res-sites}). $D$: next-token dialect preference toward the matrix variety (nats). Last column: flip rate of MSA-in-EGY words before $\rightarrow$ after ablation (\%).\label{tab:ablation}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCCCc}
\toprule
\textbf{Run} & \textbf{Own logit, MSA in EGY} & \textbf{Own logit, EGY in MSA} & \textbf{$D$, MSA in EGY} & \textbf{$D$, EGY in MSA} & \textbf{Flip MSA} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_ablation", body)


def table_sites(R, out):
    """exp09: equal-count intervention sites (embedded word / next position / all later positions)."""
    def pv(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"

    def c(t):
        return f"{f(t[0], 3, True)} [{f(t[1], 3)}, {f(t[2], 3)}] ({pv(t[3])})"
    rows, rows2 = [], []
    for run, name in RUNS:
        sj = J(R / "exp09_sites" / run / "summary.json")
        t = pd.DataFrame(sj["sites"])
        t = t[t.group == "EGY_in_MSA"].set_index(["measure", "site"])
        g = lambda m, st: [t.loc[(m, st), k] for k in ["net", "lo", "hi", "p", "n"]]
        pr = sj["paired"]
        m = "d_cont_lp_matrix"
        rows.append(f"{name} & {int(g(m, 'target')[4])} & {c(g(m, 'target'))} & {c(g(m, 'next1'))} & {c(g(m, 'later'))} & "
                    f"{c(pr['EGY_in_MSA/' + m + '/next1_minus_target'])} & {c(pr['EGY_in_MSA/' + m + '/later_minus_next1'])} \\\\")
        for wi, w in enumerate(["w1", "w2", "w3"]):
            a = g(f"d_cont_lp_matrix_{w}", "target")
            b = g(f"d_cont_lp_matrix_{w}", "later")
            d = pr[f"EGY_in_MSA/d_cont_lp_matrix_{w}/later_minus_target"]
            lab = name if wi == 0 else ""
            rows2.append(f"{lab} & {wi + 1} & {int(a[4])} & {c(a)} & {c(b)} & {c(d)} \\\\")
        rows2[-1] += "[2pt]"
    body = (r"""\begin{table}[!htbp]
\caption{Intervention sites with equal numbers of ablated positions. EGY words embedded in MSA sentences; outcome: change in the mean log-probability per token of the following MSA words (up to three words), net of 100 layer- and norm-matched random draws per site. Embedded word and next position each ablate one position; later positions ablate every position after the embedded word. Brackets: tweet-cluster bootstrap 95\% intervals; parentheses: two-sided bootstrap $p$. Paired columns: item-level differences between the net effects of two sites. Only items whose next three words include an MSA-labelled word enter (column $n$; 23--24 tweets per model). This is a separate run from the initial site comparison, with new random draws.\label{tab:sites}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lcCCCCC}
\toprule
\textbf{Run} & $n$ & \textbf{Embedded word} & \textbf{Next position} & \textbf{All later positions} & \textbf{Next $-$ embedded} & \textbf{All later $-$ next} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_sites", body)
    body2 = (r"""\begin{table}[!htbp]
\caption{Distance profile of the site effects (same items and design as Table~\ref{tab:sites}). Net change in the log-probability per token of the first, second and third following word when that word is MSA, for ablation at the embedded word and at all later positions, and the item-level paired difference between the two sites on the same word. Brackets: tweet-cluster bootstrap 95\% intervals; parentheses: two-sided bootstrap $p$; $n$: items. Only the first sub-token of word 1 is predicted from the state at the embedded word; later-site ablation can reach word 1 only through its remaining sub-tokens. The numbers of items differ by word (13--21), and the $p$-values are not corrected for the number of comparisons; this profile is exploratory.\label{tab:distance}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lccCCC}
\toprule
\textbf{Run} & \textbf{Word} & $n$ & \textbf{Ablation at the embedded word} & \textbf{Ablation at all later positions} & \textbf{Later $-$ embedded (paired)} \\
\midrule
""" + "\n".join(rows2) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_distance", body2)


def table_decode(R, out):
    rows = []
    for slug, name in MODELS:
        r = J(R / "exp11_decodability" / slug / "summary.json")["results"]
        a, b = r["MSA_in_EGY"], r["EGY_in_MSA"]
        th = a["saved_probe_flip_by_threshold"]
        rows.append(f"{name} & {ci(*a['acc_saved_mono'], d=2, sign=False)} & {ci(*a['acc_balanced'], d=2, sign=False)} & "
                    f"{ci(*a['acc_balanced_unseen_types'], d=2, sign=False)} ({a['n_unseen_type']}) & "
                    f"{a['selectivity_seen_types']:.2f} & {ci(*b['acc_balanced'], d=2, sign=False)} & "
                    f"{ci(*b['acc_balanced_unseen_types'], d=2, sign=False)} ({b['n_unseen_type']}) & "
                    f"{r['matrix_MSA']['acc_saved_mono'][0]:.2f} & {pc(th['-2'])}--{pc(th['2'])} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Is the embedded word's own variety still linearly recoverable? Accuracy of recovering the embedded word's own variety at layer $R$ (tweet-cluster bootstrap 95\% intervals). Saved probe: the monolingual read-out probe of the paper. Balanced probe: logistic probe trained on all labelled words of pure types in code-switched tweets with the four cells \{MSA, EGY word\}~$\times$~\{MSA, EGY matrix\} weighted equally (which decorrelates the matrix-variety label from the word label but does not remove other contextual information), evaluated on held-out tweets (5-fold split by tweet); unseen types: test words whose type does not occur in the training folds (number of items in parentheses). Selectivity: balanced-probe accuracy minus the accuracy of the same probe on a control task with labels assigned at random per word type (types seen in training). Matrix MSA: accuracy of the saved probe on MSA words in MSA-matrix code-switched tweets. Last column: flip rate of MSA-in-EGY words under the saved probe when the decision threshold on the own-variety logit is moved from $-2$ to $+2$.\label{tab:decode}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCCcCCcc}
\toprule
 & \multicolumn{4}{c}{\textbf{MSA in EGY}} & \multicolumn{2}{c}{\textbf{EGY in MSA}} & & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
\textbf{Model} & \textbf{Saved probe} & \textbf{Balanced probe} & \textbf{Balanced, unseen types} & \textbf{Select.} & \textbf{Balanced probe} & \textbf{Balanced, unseen types} & \textbf{Matrix MSA} & \textbf{Flip (\%), $\tau=-2..2$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_decode", body)


def table_appendix(R, out):
    # verification of the ablation hooks + independence audit
    rows = []
    d8 = J(R / "exp08_final_analyses.json")["D"]
    for slug, name in MODELS:
        v = J(R / "verify_ablation" / f"{slug}.json")["checks"]
        ok = all(v[k] for k in v if isinstance(v[k], bool))
        dev = max(v["max_abs_deviation"].values())
        a = d8[slug]
        rows.append(f"{name} & {'pass (5/5)' if ok else 'FAIL'} & {dev:.0e} & {a['MSA']['also_exp02_prefix_tweets']}/{a['MSA']['control_tweets']} & "
                    f"{a['EGY']['also_exp02_prefix_tweets']}/{a['EGY']['control_tweets']} & "
                    f"{f(a['MSA']['mono_own_logit_net_without_overlap'][0], 2, True)} / {f(a['EGY']['mono_own_logit_net_without_overlap'][0], 2, True)} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Audit of the intervention code and of sample independence. Hook audit: five automated checks on the real models, run on five items per model (each ablated head slice equals the mean of its own layer; layer means differ; no change before the intervention point; non-ablated heads untouched in the first ablated layer; later-only ablation leaves the embedded word unchanged); maximal absolute deviation over checks. Overlap: monolingual control tweets that were also used as prefix tweets during head selection. Last column: net top-5 effect on monolingual MSA / EGY words after removing the overlapping tweets.\label{tab:audit}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
\textbf{Model} & \textbf{Hook audit} & \textbf{Max dev.} & \textbf{Overlap MSA} & \textbf{Overlap EGY} & \textbf{Mono effect w/o overlap} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write(out, "tab_audit", body)

    rows = []
    for slug, name in MODELS:
        s = J(R / "exp07" / slug / "summary.json")
        fc = {(r["own"], r["is_function"]): r for r in s["function_vs_content"]}
        pr = s["purity_robustness"]
        rows.append(f"{name} & {pc(fc[('MSA', 0)]['flip'])} ({fc[('MSA', 0)]['n']}) & {pc(fc[('EGY', 0)]['flip'])} ({fc[('EGY', 0)]['n']}) & "
                    f"{pc(fc[('MSA', 1)]['flip'])} ({fc[('MSA', 1)]['n']}) & {pc(fc[('EGY', 1)]['flip'])} ({fc[('EGY', 1)]['n']}) & "
                    f"{pc(pr['in_all_purity']['MSA']['flip'])} / {pc(pr['in_all_purity']['EGY']['flip'])} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Flip rates (\%) of embedded words by word class (item counts in parentheses) and under the alternative purity definition computed from all labels (MSA / EGY). Main analyses compute type purity from monolingual tweets only.\label{tab:wordclass}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
 & \multicolumn{2}{c}{\textbf{Content words}} & \multicolumn{2}{c}{\textbf{Function words}} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Model} & \textbf{MSA in EGY} & \textbf{EGY in MSA} & \textbf{MSA in EGY} & \textbf{EGY in MSA} & \textbf{All-label purity} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write(out, "tab_wordclass", body)

    rows = []
    for run, name in RUNS:
        s = J(R / "exp06" / run / "summary.json")
        L = s["probe_to_prediction_link"]
        m = L["MSA_in_EGY"]
        dd = m["D_to_matrix_flipped_minus_not"]
        rl = J(R / "review" / "review_long.json")["R1_link_cluster"][run]
        cr = lambda k: f"{f(rl[k][0], 2)} [{f(rl[k][1], 2)}, {f(rl[k][2], 2)}]"
        rows.append(f"{name} & {m['n_flipped']}/{m['n_flipped'] + m['n_not']} & {ci(dd[0], dd[1], dd[2], 2, True)} & "
                    f"{cr('MSA_in_EGY/baseline(-own_logit,D)')} & {cr('EGY_in_MSA/baseline(-own_logit,D)')} & "
                    f"{cr('MSA_in_EGY/top5(d_own_logit,d_D)')} & {cr('EGY_in_MSA/top5(d_own_logit,d_D)')} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Probe reading and model prediction. Flipped: MSA-in-EGY items read as Egyptian at layer $R$. $D$ difference: next-token preference toward the matrix variety, flipped minus non-flipped items (nats, bootstrap 95\% interval). Baseline $\rho$: Spearman correlation between the flip-oriented probe reading ($-$own-variety logit) and $D$. Under ablation: Spearman correlation between the item-level change in own-variety logit and the change in $D$ under top-5 mean ablation. Brackets: tweet-cluster bootstrap 95\% intervals (2000 resamples); all cluster-bootstrap $p\le0.005$.\label{tab:link}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{lcCCCCC}
\toprule
 & & & \multicolumn{2}{c}{\textbf{Baseline $\rho$}} & \multicolumn{2}{c}{\textbf{$\rho$ under ablation}} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Run} & \textbf{Flipped} & \textbf{$D$ difference} & \textbf{MSA in EGY} & \textbf{EGY in MSA} & \textbf{MSA in EGY} & \textbf{EGY in MSA} \\
\midrule
""" + "\n".join(rows[:4]) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_link", body)

    rows = []
    for run, name in RUNS:
        s = J(R / "exp06" / run / "summary.json")
        d = s["dla_type_matched"][0]
        rows.append(f"{name} & {d['head']} & {f(d['mono_MSA'], 2, True)} & {f(d['mono_EGY'], 2, True)} & {f(d['MSA_in_EGY'], 2, True)} & "
                    f"{f(d['EGY_in_MSA'], 2, True)} & {f(d['shift_MSA_toward_EGY'], 2, True)} & {f(d['shift_EGY_toward_MSA'], 2, True)} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Direct attribution of the top head's output at the target position onto the read-out probe (positive = toward EGY; Gemma-2 family includes the post-attention normalisation). Monolingual reference words are of the same unambiguous types as the embedded words. Shift: context-induced change of the head's contribution for embedded words relative to the monolingual reference of the same types (toward the matrix variety). Direct attribution ignores downstream processing and is reported as supporting evidence only.\label{tab:dla}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{llCCCCCC}
\toprule
\textbf{Run} & \textbf{Head} & \textbf{Mono MSA} & \textbf{Mono EGY} & \textbf{MSA in EGY} & \textbf{EGY in MSA} & \textbf{Shift, MSA words} & \textbf{Shift, EGY words} \\
\midrule
""" + "\n".join(rows[:4]) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write(out, "tab_dla", body)


def table_samples(R, out):
    thou = lambda x: f"{x:,}".replace(",", "{,}")
    """Per-experiment samples, units and measures (explains why rates differ between analyses)."""
    n4, n6, n7 = [], [], []
    for slug, name in MODELS:
        s4 = J(R / "exp04" / slug / "summary.json")
        Rl = s4["readout_layer"]
        a = s4[f"matrix_left>=3@L{Rl}"]["natural_pull"]["n"]
        n4.append(f"{a[0]}/{a[1]}")
        b = J(R / "exp06" / slug / "summary.json")["n_items"]
        n6.append(f"{b['MSA_in_EGY']}/{b['EGY_in_MSA']}")
        c = J(R / "exp07" / slug / "summary.json")["n_items"]
        n7.append(f"{c['MSA']}/{c['EGY']}")
    F = pd.read_csv(R / "exp01" / MODELS[0][0] / "F_context_effect.csv")
    F = F[(F.layer == F.layer.max()) & (F.types == "unambiguous")].set_index("condition")["n"]
    nF = f"{F['MSA embedded in EGY']}/{F['EGY embedded in MSA']} (all models)"
    ndec = []
    for slug, _ in MODELS:
        dj = J(R / "exp11_decodability" / slug / "summary.json")
        ndec.append(f"{dj['n_words']} words / {dj['n_tweets']} tweets")
    m7b = J(R / "exp07b_matching_sensitivity.json")
    n7b = [f"{m7b[slug]['n_items']['MSA']}/{m7b[slug]['n_items']['EGY']}" for slug, _ in MODELS]
    npairs = [str(m7b[slug]["exact_func__lex_host_mean_nleft"]["n_pairs"]) for slug, _ in MODELS]
    e8 = J(R / "exp08_final_analyses.json")["A_B"]
    n8 = [str(e8[slug]["EGY_in_MSA/d_cont_lp_matrix"]["source_net"][4]) for slug, _ in MODELS]
    s1 = J(R / "exp01" / MODELS[0][0] / "summary.json")
    ev = s1["eval_set_sizes"]
    pr = J(R / "exp07" / MODELS[0][0] / "summary.json")["probe"]["n_train"]
    rows = [
        (r"Probe training and evaluation (\S\ref{sec:probe})", "words from monolingual tweets",
         ("layer sweep: " + "; ".join([f"{thou(s1['n_train_words'])} train", f"{thou(ev['mono_test'])} held-out", f"{thou(ev['unseen_types'])} unseen-type", f"{thou(ev['ambiguous_forms'])} ambiguous"]) + f". Saved read-out probe: {thou(pr['MSA'])} MSA + {thou(pr['EGY'])} EGY words (80/20 split)"),
         "accuracy", "all layers"),
        (r"Type-matched context effect (\S\ref{sec:res-phenomenon})", "embedded words of unambiguous types vs.\\ the same types in monolingual text", nF, r"$\Delta P(\text{own})$", "all layers"),
        (r"Flip rates, natural vs.\ random prefix (\S\ref{sec:res-phenomenon})", r"embedded words with $\geq$3 matrix words on the left", "; ".join(n4), "flip, own-variety logit", "$R$"),
        (r"Flip rates by word class and purity (\S\ref{sec:res-controls}, App.~\ref{app:robust})", r"embedded words of pure types with $\geq$3 preceding tokens", "; ".join(n7), "flip", "$R$"),
        (r"Host strength, regression, matching (\S\ref{sec:res-phenomenon}--\ref{sec:res-controls})", "the same items, restricted to those with defined host measures", "; ".join(n7b) + "; matched pairs " + "; ".join(npairs), "probe logit of host words; flip", "$R$"),
        (r"Minimal pairs, patching (\S\ref{sec:res-patching})", "random unambiguous types after 12-token prefixes", "150 pairs per variety", "difference-of-means score", "all / $R$"),
        (r"Ablation and prediction link (\S\ref{sec:res-link}--\ref{sec:res-ablation})", "embedded words with a following word; 150 monolingual controls per variety", "; ".join(n6), r"own-variety logit, $D$, continuation log-prob.", "$R$ / output"),
        (r"Recoverability of own variety (\S\ref{sec:decodemethod})", "all labelled words of pure types in code-switched tweets, 5 folds split by tweet",
         "; ".join(ndec), "accuracy, selectivity", "$R$"),
        (r"Intervention sites (\S\ref{sec:res-sites})", "EGY-in-MSA items followed by an MSA word within three words", "; ".join(n8), "continuation log-prob.", "output"),
    ]
    lines = "\n".join(" & ".join(r) + r" \\" for r in rows)
    body = (r"""\begin{table}[!htbp]
\caption{Samples and measures of each analysis. Where two counts are separated by a slash they are MSA-in-EGY / EGY-in-MSA items; where four entries are separated by semicolons they refer to Qwen3-8B / ALLaM-7B / Gemma-2-9B / Fanar-1-9B (item sets differ slightly between models because of tokenisation and the sub-token filters). The numbers of flipped and non-flipped items used in the prediction analyses are given in Table~\ref{tab:link}. All analyses use the same LinCE tweets, so results for different models are not independent replications.\label{tab:samples}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\begin{tabularx}{\fulllength}{p{3.4cm}LLp{3.0cm}p{1.4cm}}
\toprule
\textbf{Analysis} & \textbf{Unit} & \textbf{$n$} & \textbf{Measure} & \textbf{Layer} \\
\midrule
""" + lines + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_samples", body)


def table_exploratory(R, out):
    e8 = J(R / "exp08_final_analyses.json")["A_B"]
    rows = []
    for run, name in RUNS:
        cells = []
        for k in ["EGY_in_MSA/d_cont_lp_own", "MSA_in_EGY/d_cont_lp_own", "MSA_in_EGY/d_cont_lp_matrix"]:
            v = e8[run][k]["later_net"]
            cells.append(f"{f(v[0], 3, True)} [{f(v[1], 3)}, {f(v[2], 3)}]")
        n = [e8[run][k]["later_net"][4] for k in ["EGY_in_MSA/d_cont_lp_own", "MSA_in_EGY/d_cont_lp_own", "MSA_in_EGY/d_cont_lp_matrix"]]
        rows.append(f"{name} & " + " & ".join(cells) + f" & {n[0]} / {n[1]} / {n[2]} \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Exploratory (not pre-specified) outcomes of ablating the top-5 heads at the positions after the embedded word, net of 100 matched random draws: change in mean log-probability per token of the following words of a given variety, with tweet-cluster bootstrap 95\% intervals. Columns 2--3: following words of the embedded word's own variety; column 4: following matrix-variety words after an MSA insertion.\label{tab:exploratory}}
\begin{adjustwidth}{-\extralength}{0cm}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\fulllength}{lCCCc}
\toprule
\textbf{Run} & \textbf{EGY in MSA $\rightarrow$ following EGY words} & \textbf{MSA in EGY $\rightarrow$ following MSA words} & \textbf{MSA in EGY $\rightarrow$ following EGY words} & $n$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")
    write(out, "tab_exploratory", body)


def table_local(R, out):
    rv = R / "review" / "review_saved.json"
    if not rv.exists():
        return
    r = J(rv)
    rows = []
    for slug, name in MODELS:
        a = r["S1_local_insertions"][slug]
        d, dl = a["diff"], a["all_items_diff"]
        rows.append(f"{name} & {pc(dl[0])} [{pc(dl[1])}, {pc(dl[2])}] & {a['n']['MSA']} / {a['n']['EGY']} & "
                    f"{pc(a['flip']['MSA'])} / {pc(a['flip']['EGY'])} & {pc(d[0])} [{pc(d[1])}, {pc(d[2])}] \\\\")
    body = (r"""\begin{table}[!htbp]
\caption{Local insertions. Flip rates (\%) restricted to embedded words with no MSA- or EGY-labelled word of their own variety on their left, so that all labelled words in the left context available to the model are of the matrix variety (drawn from the item set of the word-class analyses, Table~\ref{tab:samples}). Differences are MSA-in-EGY minus EGY-in-MSA in percentage points with tweet-cluster bootstrap 95\% intervals; each local item comes from a different tweet.\label{tab:local}}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCC}
\toprule
\textbf{Model} & \textbf{Difference, all items} & \textbf{Local items (MSA / EGY)} & \textbf{Flip, local (MSA / EGY)} & \textbf{Difference, local} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write(out, "tab_local", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table_representation(R, out)
    table_regression(R, out)
    table_matching(R, out)
    table_tost(R, out)
    table_patching(R, out)
    table_ablation(R, out)
    table_sites(R, out)
    table_appendix(R, out)
    table_samples(R, out)
    table_exploratory(R, out)
    table_local(R, out)
    table_decode(R, out)
    print("tables written to", out)


if __name__ == "__main__":
    main()
