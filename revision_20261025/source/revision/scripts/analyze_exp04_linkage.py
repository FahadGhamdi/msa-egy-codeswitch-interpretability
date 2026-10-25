"""Validate conditional linkage and calculate pre-specified saved-output checks.

No original files are changed; no tokenizer, model, or probe is executed.
"""
from pathlib import Path
import ast, collections, datetime, hashlib, json, sys, zipfile
import numpy as np
import pandas as pd

ROOT=Path('/workspace/scratch/e3ad691229b6')
REPO=ROOT/'electronics_revision_20261021'
REV=REPO/'revision'
LINK=ROOT/'local_evidence_received/exp04_linkage/ELECTRONICS_EXP04_LINKAGE_20261021T200851Z'
EVID=ROOT/'local_evidence_received/verified_outer/ELECTRONICS_REVISION_AUDIT_20261021T192824Z'
SRC=EVID/'candidate_inventory/source_and_dataset'
RESULTS=ROOT/'local_evidence_received/verified_payload/results'
OUT=REV/'results/exp04_linked_20261021'
OUT.mkdir(parents=True,exist_ok=True)
PROTOCOL=REV/'protocols/exp04_linked_analysis_20261021.json'
CFG=json.loads(PROTOCOL.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dig(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
initial={str(PROTOCOL):sha(PROTOCOL),str(Path(__file__)):sha(Path(__file__))}
archive=ROOT/'upload/ELECTRONICS_EXP04_LINKAGE_RETURN_20261021T201511Z.zip'
initial[str(archive)]=sha(archive)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    manifest=json.loads(z.read('MANIFEST.json'))['files']
    assert len(z.namelist())==len(set(z.namelist()))
    for r in manifest:
        b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        p=LINK.parent/r['path'];assert sha(p)==r['sha256']
historical_inputs=json.loads((LINK/'manifests/inputs_before.json').read_text())
def check_input(p,relative):
    found=[v for k,v in historical_inputs.items() if k.endswith('/'+relative)]
    assert len(found)==1 and sha(p)==found[0]['sha256'],relative
    initial[str(p)]=sha(p)
helper_origins=json.loads((LINK/'reports/helper_origins.json').read_text())
extracted=ast.parse((LINK/'code/extracted_selection.py').read_text())
for r in helper_origins:
    p=SRC/r['file'];check_input(p,r['file'])
    a=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==r['function'])
    b=next(n for n in extracted.body if isinstance(n,ast.FunctionDef) and n.name==r['function'])
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
dataset={};all_tweets=[]
for split in ('train','validation'):
    p=SRC/f'data/raw/lince_msaea/lid_msaea/{split}.parquet';check_input(p,f'data/raw/lince_msaea/lid_msaea/{split}.parquet')
    df=pd.read_parquet(p)
    for i,r in df.iterrows():
        words=list(r['words']);labels=list(r['lid']);sid=f'{split}-{i}'
        n1=labels.count('lang1');n2=labels.count('lang2')
        dataset[sid]=(words,labels,n1,n2)
        all_tweets.append({'sid':sid,'n_msa':n1,'n_egy':n2,'margin':abs(n1-n2),'code_switched':n1>0 and n2>0})
counts=collections.Counter(t['margin'] for t in all_tweets if t['code_switched'])
write('corpus_margin_counts.json',{'total_rows':len(all_tweets),'code_switched':sum(counts.values()),'ties':counts[0],'margin1':counts[1],'margin2':counts[2],'margin_gt2':sum(v for k,v in counts.items() if k>2)})
MODELS=[('Qwen3-8B-Base',29,36),('ALLaM-7B-Instruct-preview',26,32),('gemma-2-9b',34,42),('Fanar-1-9B',34,42)]
summaries=[];gaps=[];audits=[]
def bootstrap(d,seed):
    g=d.groupby('sid',sort=True).agg(n=('flip','size'),flip_sum=('flip','sum'),logit_sum=('natural_own_logit','sum'))
    n=g.n.to_numpy();f=g.flip_sum.to_numpy();l=g.logit_sum.to_numpy()
    rng=np.random.default_rng(np.random.SeedSequence(seed));fs=[];ls=[]
    for offset in range(0,CFG['bootstrap_repetitions'],500):
        k=min(500,CFG['bootstrap_repetitions']-offset)
        idx=rng.integers(0,len(g),size=(k,len(g)));den=n[idx].sum(axis=1)
        fs.append(f[idx].sum(axis=1)/den);ls.append(l[idx].sum(axis=1)/den)
    return np.concatenate(fs),np.concatenate(ls)
for mi,(slug,R,L) in enumerate(MODELS):
    mp=LINK/f'item_mapping_{slug}.csv';initial[str(mp)]=sha(mp)
    m=pd.read_csv(mp,keep_default_na=False)
    yp=RESULTS/'exp04'/slug/'Y_long.csv';check_input(yp,f'results/exp04/{slug}/Y_long.csv')
    y=pd.read_csv(yp,keep_default_na=False)
    assert m.item.tolist()==list(range(len(m))) and len(y)==len(m)*(L+1)
    assert not y.duplicated(['item','layer']).any()
    assert y.groupby('item').layer.apply(list).tolist()==[list(range(L+1))]*len(m)
    const=['own','word','K','n_left_matrix','n_left_own']
    assert (y.groupby('item')[const].nunique()==1).all().all()
    yr=y[y.layer==R].reset_index(drop=True)
    assert yr.item.tolist()==m.item.tolist()
    for c in const:assert yr[c].tolist()==m[c].tolist(),(slug,c)
    assert (m.classification=='conditional_reconstruction').all() and m.full_sequence_match.all()
    assert not m.duplicated(['sid','word_index']).any()
    margins=[]
    for r in m.to_dict('records'):
        ws,ls,n1,n2=dataset[r['sid']];j=int(r['word_index']);own=r['own'];lab='lang1' if own=='MSA' else 'lang2';other='lang2' if own=='MSA' else 'lang1'
        assert ws[j]==r['word'] and ls[j]==lab
        assert ls[:j].count(lab)==r['n_left_own'] and ls[:j].count(other)==r['n_left_matrix']
        assert n1>0 and n2>0 and n1!=n2 and ((n1<n2)==(own=='MSA'))
        assert dig(ws)==r['tweet_words_sha256'] and dig(ls)==r['tweet_labels_sha256'] and dig({'words':ws,'labels':ls})==r['tweet_words_labels_sha256']
        opts=json.loads(r['candidate_locations_json'])
        assert len(opts)==r['signature_candidate_count'] and {'sid':r['sid'],'word_index':j} in opts
        for op in opts:
            ow,ol,_,_=dataset[op['sid']];oj=op['word_index']
            assert ow[oj]==r['word'] and ol[oj]==lab
            assert ol[:oj].count(lab)==r['n_left_own'] and ol[:oj].count(other)==r['n_left_matrix']
        margins.append(abs(n1-n2))
    m['margin']=margins
    m['natural_own_logit']=yr.target_own_logit.to_numpy()-yr.natural_pull.to_numpy()
    assert np.isfinite(m.natural_own_logit).all()
    m['flip']=(m.natural_own_logit<0).astype(int)
    full=m[m.n_left_matrix>=3].copy()
    summaryp=RESULTS/'exp04'/slug/'summary.json';check_input(summaryp,f'results/exp04/{slug}/summary.json')
    original=json.loads(summaryp.read_text())['robustness'][f'L{R}']['flip_rate_natural']
    for v in ('MSA','EGY'):assert abs(full[full.own==v].flip.mean()-original[v])<1e-12
    full[['item','sid','word_index','own','margin','signature_candidate_count','natural_own_logit','flip']].to_csv(OUT/f'linked_table5_{slug}.csv',index=False)
    audits.append({'model':slug,'all_linked_items':len(m),'table5_items':len(full),'table5_tweets':int(full.sid.nunique()),'table5_duplicate_signature_items':int((full.signature_candidate_count>1).sum()),'raw_tweet_word_label_hash_and_context_verified':True,'saved_item_metadata_all_layers_verified':True,'conditional_linkage':True,'tokenizer_rerun_in_this_validation':False})
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
assert all(sha(Path(p))==s for p,s in initial.items())
write('validation.json',{'paper':'electronics-4637664','archive_members_verified':len(manifest),'models':audits,'total_conditional_links_verified':sum(a['all_linked_items'] for a in audits),'protocol_sha256':sha(PROTOCOL),'inputs_and_code_sha256':initial,'inputs_unchanged':True,'historical_identity_proven':False,'inference_executed':False,'bootstrap_repetitions':CFG['bootstrap_repetitions'],'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}})
print(pd.DataFrame(gaps).to_string(index=False))
