from pathlib import Path
import os,json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ['ELECTRONICS_REPRO_OUT'])/'exp04';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'revision/results/exp04_linked_20261021'
CFG=json.loads((ROOT/'revision/protocols/exp04_linked_analysis_20261021.json').read_text())
MODELS=['Qwen3-8B-Base','ALLaM-7B-Instruct-preview','gemma-2-9b','Fanar-1-9B']
summaries=[];gaps=[]
def bootstrap(d,seed):
    g=d.groupby('sid',sort=True).agg(n=('flip','size'),flip_sum=('flip','sum'),logit_sum=('natural_own_logit','sum'))
    n=g.n.to_numpy();f=g.flip_sum.to_numpy();l=g.logit_sum.to_numpy()
    rng=np.random.default_rng(np.random.SeedSequence(seed));fs=[];ls=[]
    for offset in range(0,CFG['bootstrap_repetitions'],500):
        k=min(500,CFG['bootstrap_repetitions']-offset)
        idx=rng.integers(0,len(g),size=(k,len(g)));den=n[idx].sum(axis=1)
        fs.append(f[idx].sum(axis=1)/den);ls.append(l[idx].sum(axis=1)/den)
    return np.concatenate(fs),np.concatenate(ls)
for mi,slug in enumerate(MODELS):
    full=pd.read_csv(SOURCE/f'linked_table5_{slug}.csv')
    for si,subset in enumerate(['all_reconstructed','unique_signature_only']):
        base=full if si==0 else full[full.signature_candidate_count==1]
        for cutoff in (0,1,2):
            d=base[base.margin>cutoff];bflip={}
            for vi,v in enumerate(('MSA','EGY')):
                a=d[d.own==v];seed=[CFG['seed'],mi,si,cutoff,vi]
                bf,bl=bootstrap(a,seed);bflip[v]=bf
                summaries.append({'model':slug,'subset':subset,'exclude_margin_le':cutoff,'own':v,'n_items':len(a),'n_tweets':a.sid.nunique(),'flip_count':int(a.flip.sum()),'flip_rate':a.flip.mean(),'flip_ci_low':np.quantile(bf,.025),'flip_ci_high':np.quantile(bf,.975),'natural_own_logit_mean':a.natural_own_logit.mean(),'logit_ci_low':np.quantile(bl,.025),'logit_ci_high':np.quantile(bl,.975)})
            bg=bflip['MSA']-bflip['EGY']
            gaps.append({'model':slug,'subset':subset,'exclude_margin_le':cutoff,'n_items':len(d),'n_tweets':d.sid.nunique(),'flip_gap':d[d.own=='MSA'].flip.mean()-d[d.own=='EGY'].flip.mean(),'gap_ci_low':np.quantile(bg,.025),'gap_ci_high':np.quantile(bg,.975)})
pd.DataFrame(summaries).to_csv(OUT/'stratum_estimates.csv',index=False)
pd.DataFrame(gaps).to_csv(OUT/'asymmetry_estimates.csv',index=False)
