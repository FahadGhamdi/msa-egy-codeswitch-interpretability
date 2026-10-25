"""Review-time sensitivity analyses. No model inference; see frozen protocol JSON.
Never substitutes the exp07 cohort for the original exp04 Table 5 cohort.
"""
from pathlib import Path
import hashlib, json, urllib.request, datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'revision/results'
RAW = ROOT / 'data/raw/lince_msaea/lid_msaea'
MODELS = ['Qwen3-8B-Base','ALLaM-7B-Instruct-preview','gemma-2-9b','Fanar-1-9B']
SEED, B = 20261021, 5000
EXPECTED_DATA_SHA256 = {
    'train':'8044dbdbbb1fab6f50f8791a1d55a4d558e82374fbecfc8ad6b6d87cd828946b',
    'validation':'9f0c6fa4df1d4f58cc4af583b54a6cd0a05005c83e4a16b5f502fcb0ce252c37',
}

def boot_mean(d, column):
    g = d.groupby('sid')[column].agg(['sum','count'])
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0,len(g),(B,len(g)))
    means = g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
    return float(d[column].mean()), *map(float,np.quantile(means,[.025,.975]))

def boot_difference(d):
    # Strata are disjoint by matrix variety; item-weighted means within tweet clusters.
    rng=np.random.default_rng(SEED)
    reps=[]
    for own in ['MSA','EGY']:
        g=d[d.own==own].groupby('sid').flip.agg(['sum','count'])
        ix=rng.integers(0,len(g),(B,len(g)))
        reps.append(g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1))
    point=d[d.own=='MSA'].flip.mean()-d[d.own=='EGY'].flip.mean()
    return float(point), *map(float,np.quantile(reps[0]-reps[1],[.025,.975]))

def main():
    OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    rows={}; sources=[]
    for split in ['train','validation']:
        p=RAW/f'{split}.parquet'
        url=f'https://huggingface.co/datasets/lince-benchmark/lince/resolve/refs%2Fconvert%2Fparquet/lid_msaea/{split}/0000.parquet'
        if not p.exists():
            with urllib.request.urlopen(url,timeout=45) as r: p.write_bytes(r.read())
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==EXPECTED_DATA_SHA256[split], 'Data bytes changed; review provenance before running'
        frame=pd.read_parquet(p)
        sources.append({'split':split,'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(frame)})
        for i,r in frame.iterrows():
            ws,ls=list(r.words),list(r.lid)
            assert len(ws)==len(ls)
            m,e=ls.count('lang1'),ls.count('lang2')
            rows[f'{split}-{i}']={'words':ws,'labels':ls,'msa':m,'egy':e,'margin':abs(m-e),'cs':m>0 and e>0}
    cs=[r for r in rows.values() if r['cs']]
    counts={'total_rows':len(rows),'code_switched':len(cs),'tied':sum(r['margin']==0 for r in cs),
            'margin_1':sum(r['margin']==1 for r in cs),'margin_2':sum(r['margin']==2 for r in cs),
            'margin_gt_2':sum(r['margin']>2 for r in cs)}
    assert len(cs)==1054, 'Dataset version/count mismatch: do not continue'
    records=[]; differences=[]; checks=[]
    for model in MODELS:
        d=pd.read_csv(ROOT/'results/exp07'/model/'L_items.csv')
        assert d.item.is_unique and np.isfinite(d.nat_own).all()
        assert np.array_equal(d.flip.to_numpy(),(d.nat_own<0).astype(int).to_numpy())
        assert set(d.sid)<=set(rows)
        for r in d.itertuples():
            src=rows[r.sid]; lab='lang1' if r.own=='MSA' else 'lang2'
            assert (r.word,lab) in set(zip(src['words'],src['labels']))
            assert src['margin']>0
            assert (src['egy']>src['msa'])==(r.own=='MSA')
        d['matrix_margin']=[rows[s]['margin'] for s in d.sid]
        for sample,base in [('exp07_all',d),('exp07_left_matrix_ge3',d[d.n_left_matrix>=3])]:
            for exclude in [0,1,2]:
                sub=base[base.matrix_margin>exclude]
                diff=boot_difference(sub)
                differences.append(dict(model=model,sample=sample,exclude_margin_le=exclude,
                                        difference=diff[0],ci_low=diff[1],ci_high=diff[2]))
                for own,g in sub.groupby('own'):
                    for metric in ['flip','nat_own']:
                        mean,lo,hi=boot_mean(g,metric)
                        records.append(dict(model=model,sample=sample,exclude_margin_le=exclude,own=own,
                                            metric=metric,n_items=len(g),n_tweets=g.sid.nunique(),estimate=mean,ci_low=lo,ci_high=hi))
        checks.append({'model':model,'n_items':len(d),'n_tweets':d.sid.nunique(),'item_alignment_pass':True})
    pd.DataFrame(records).to_csv(OUT/'saved_sensitivity.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'saved_asymmetry.csv',index=False)
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,
            'data_sources':sources,'checks':checks,'seed':SEED,'bootstrap_draws':B,
            'scope':'exp07 saved items only. This is NOT independent replication and NOT the exact exp04 Table5 cohort.'}
    (OUT/'saved_analysis_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print(pd.DataFrame(differences).to_string(index=False))

if __name__=='__main__': main()
