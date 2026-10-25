"""Generate review tables from existing saved values and new frozen-protocol outputs."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revision/manuscript/tables'
MODELS=[('Qwen3-8B-Base','Qwen3'),('ALLaM-7B-Instruct-preview','ALLaM'),('gemma-2-9b','Gemma-2'),('Fanar-1-9B','Fanar')]
def J(p):return json.loads(p.read_text())
def table(name,caption,cols,header,rows):
    text='\\begin{table}[!htbp]\n\\caption{'+caption+'\\label{'+name+'}}\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\small\n\\begin{tabularx}{\\fulllength}{'+cols+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabularx}\n\\end{adjustwidth}\n\\end{table}\n'
    (OUT/(name.replace(':','_')+'.tex')).write_text(text)

def main():
    R=ROOT/'results'; rows=[]
    ns4=[];ns6=[];ns7=[];nd=[];npairs=[];nmatch=[];nsites=[]
    match=J(R/'exp07b_matching_sensitivity.json')
    for slug,_ in MODELS:
        s=J(R/'exp04'/slug/'summary.json');n=s[f"matrix_left>=3@L{s['readout_layer']}"]['natural_pull']['n'];ns4.append(f'{n[0]}/{n[1]}')
        n=J(R/'exp06'/slug/'summary.json')['n_items'];ns6.append(f"{n['MSA_in_EGY']}/{n['EGY_in_MSA']}")
        n=J(R/'exp07'/slug/'summary.json')['n_items'];ns7.append(f"{n['MSA']}/{n['EGY']}")
        n=J(R/'exp11_decodability'/slug/'summary.json');nd.append(f"{n['n_words']}/{n['n_tweets']}")
        n=match[slug]['n_items'];nmatch.append(f"{n['MSA']}/{n['EGY']}")
        npairs.append(str(match[slug]['exact_func__lex_host_mean_nleft']['n_pairs']))
        nsites.append(str(int(J(R/'exp08_final_analyses.json')['A_B'][slug]['EGY_in_MSA/d_cont_lp_matrix']['source_net'][4])))
    for label,counts in [('Type-matched words',['332/366']*4),('Natural/random prefix items',ns4),('Word-class/purity items',ns7),('Regression/matching eligible',nmatch),('Matched pairs',npairs),('Patching pairs per variety',['150']*4),('Ablation/prediction items',ns6),('Recoverability: words/tweets',nd),('Intervention-site items',nsites)]:
        rows.append(' & '.join([label,*counts])+r' \\')
    table('tab:samples',r'Sample sizes by model. Paired counts give MSA-in-EGY/EGY-in-MSA items, except recoverability (words/tweets). Tokenisation and eligibility filters differ by model. The layer sweep uses 15,000 training words, 5960 held-out words, 2236 unseen-type words and 11,435 ambiguous words; the saved readout uses 4811 MSA and 4800 EGY training words. Ablation includes 150 monolingual controls per variety. Probe accuracy and type-matched effects are evaluated over layers; flip, lexical and recoverability analyses use layer $R$; prediction outcomes use the output distribution. All models use the same corpus, so their results are not independent replications.', 'Xrrrr',r'\textbf{Analysis/unit} & \textbf{Qwen3} & \textbf{ALLaM} & \textbf{Gemma-2} & \textbf{Fanar}',rows)
    diff=pd.read_csv(ROOT/'revision/results/saved_asymmetry.csv')
    metric=pd.read_csv(ROOT/'revision/results/saved_sensitivity.csv')
    rows=[]
    for slug,label in MODELS:
        for cutoff in [0,1,2]:
            d=diff[(diff.model==slug)&(diff['sample']=='exp07_all')&(diff.exclude_margin_le==cutoff)].iloc[0]
            n=metric[(metric.model==slug)&(metric['sample']=='exp07_all')&(metric.exclude_margin_le==cutoff)&(metric.metric=='flip')].set_index('own')
            rows.append(f"{label} & {cutoff} & {int(n.loc['MSA','n_items'])}/{int(n.loc['EGY','n_items'])} & {100*n.loc['MSA','estimate']:.1f}/{100*n.loc['EGY','estimate']:.1f} & {100*d.difference:.1f} [{100*d.ci_low:.1f}, {100*d.ci_high:.1f}]"+r' \\')
    table('tab:revision_ties',r'Review-time sensitivity to near-tied matrix labels in the saved lexical-control cohort (experiment 07), not the experiment-04 cohort of Table~\ref{tab:representation}. Exclude tweets with $|n_{\mathrm{MSA}}-n_{\mathrm{EGY}}|\leq c$. Counts and flip rates give MSA-in-EGY/EGY-in-MSA; the last column is their difference in percentage points with 95\% tweet-cluster percentile intervals (5000 resamples; seed 20261021). This is internal sensitivity analysis, not independent confirmation.', 'lcrrX',r'Model & $c$ & Items & Flip (\%) & Difference [95\% CI]',rows)
    rows=[]
    for slug,label in MODELS:
        d=metric[(metric.model==slug)&(metric['sample']=='exp07_all')&(metric.exclude_margin_le==0)&(metric.metric=='nat_own')].set_index('own')
        cells=[f"{d.loc[v,'estimate']:.2f} [{d.loc[v,'ci_low']:.2f}, {d.loc[v,'ci_high']:.2f}]" for v in ['MSA','EGY']]
        rows.append(' & '.join([label,*cells])+r' \\')
    table('tab:revision_logits',r'Threshold-free mean own-variety logits with 95\% tweet-cluster intervals in the saved experiment-07 cohort. Positive values favour the word\textquotesingle s own variety. Values complement the flip-rate analyses but do not replace Table~\ref{tab:representation}: its experiment-04 cohort and probe construction must not be silently conflated with these saved items. Logit scales are model/probe-specific.', 'lXX',r'Model & MSA in EGY & EGY in MSA',rows)
    print('Generated sample columns and two new sensitivity tables')

if __name__=='__main__':main()
