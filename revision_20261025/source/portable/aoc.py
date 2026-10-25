from pathlib import Path
import json,hashlib,collections,numpy as np
import os
B=Path(os.environ['ELECTRONICS_REPRO_OUT']);p=B/'aoc/AOC_EXTERNAL_READOUT_RETURN_20261024_A';pkg=Path(__file__).resolve().parents[1]/'reproduction_inputs/AOC_ISSUED'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
man=json.load(open(p/'RETURN_MANIFEST.json'))
assert set(man)=={str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.name!='RETURN_MANIFEST.json'}
for rel,h in man.items():assert sha(p/rel)==h,rel
assert (p/'FROZEN_PROTOCOL.txt').read_bytes()==(pkg/'PROTOCOL.txt').read_bytes()
spec=json.load(open(pkg/'EXPECTED_INPUTS.json'));checks={x['path']:x['sha256'] for x in json.load(open(p/'INPUT_CHECKS.json'))};expected={r.get('relative_to_New_Paper',r.get('path')):r['sha256'] for r in spec['common_inputs']}
for m in spec['models']:
 expected[m['probe_path']]=m['probe_sha256'];expected.update({r['path']:r['sha256'] for r in m['snapshot_files']})
assert checks==expected
results={};total=0
for m in spec['models']:
 slug=m['slug'];items=json.load(open(pkg/(slug+'_ITEMS.json')));rr=[]
 for it in items:
  r=json.load(open(p/slug/(str(it['item'])+'.json')))
  assert all(r[k]==v for k,v in it.items())
  assert np.isfinite(r['egy_logit']) and r['own_logit']==(r['egy_logit'] if r['own']=='EGY' else -r['egy_logit'])
  assert r['flipped']==(r['own_logit']<0) and len(r['feature_sha256'])==64
  rr.append(r)
 total+=len(rr);comments=sorted({r['comment_id'] for r in rr});idx={c:i for i,c in enumerate(comments)};N=len(comments)
 # Columns are linear comment-level sums. Ratios recomputed for each bootstrap draw.
 columns=[];meta={};matched={}
 for cohort in ['primary','host_margin_gt2']:
  ss=[r for r in rr if cohort=='primary' or r['host_margin_gt2']];meta[cohort]={}
  for g in ['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']:
   ii=[r for r in ss if r['group']==g];start=len(columns)
   for key in ['count','flip','logit']:
    v=np.zeros(N)
    for r in ii:v[idx[r['comment_id']]]+=1 if key=='count' else (r['flipped'] if key=='flip' else r['own_logit'])
    columns.append(v)
   meta[cohort][g]={'n':len(ii),'comments':len({r['comment_id'] for r in ii}),'types':len({r['word'] for r in ii}),'flip_count':sum(r['flipped'] for r in ii),'zero_ties':sum(r['own_logit']==0 for r in ii),'flip_rate':float(np.mean([r['flipped'] for r in ii])),'mean_own_logit':float(np.mean([r['own_logit'] for r in ii])),'col':start}
  matched[cohort]={}
  for own,host in [('MSA','EGY'),('EGY','MSA')]:
   mono=[r for r in ss if r['group']=='mono_'+own];sw=[r for r in ss if r['group']==own+'_in_'+host]
   types=sorted({r['word'] for r in mono}&{r['word'] for r in sw});v=np.zeros(N)
   for rows,sign in [(mono,1),(sw,-1)]:
    counts=collections.Counter(r['word'] for r in rows)
    for r in rows:
     if r['word'] in types:v[idx[r['comment_id']]]+=sign*r['own_logit']/(len(types)*counts[r['word']])
   # Direct type-wise recomputation verifies the linear contribution estimator.
   direct=np.mean([np.mean([r['own_logit'] for r in mono if r['word']==t])-np.mean([r['own_logit'] for r in sw if r['word']==t]) for t in types])
   assert abs(v.sum()-direct)<1e-10
   matched[cohort][own]={'types':len(types),'estimate':float(direct),'col':len(columns)};columns.append(v)
 matrix=np.stack(columns,axis=1);rng=np.random.default_rng(20261025);boots=[]
 for start in range(0,10000,100):
  weights=rng.multinomial(N,np.full(N,1/N),size=100);boots.append(weights@matrix)
 boot=np.concatenate(boots)
 def ci(x,level=.95):
  valid=x[np.isfinite(x)];a=(1-level)/2
  return {'interval':np.quantile(valid,[a,1-a]).tolist(),'valid_resamples':len(valid),'invalid_resamples':10000-len(valid)}
 model={}
 for cohort in meta:
  groups=meta[cohort];rates={}
  for g,d in groups.items():
   c=d.pop('col');den=boot[:,c];rates[g]=np.divide(boot[:,c+1],den,out=np.full(10000,np.nan),where=den>0)
   means=np.divide(boot[:,c+2],den,out=np.full(10000,np.nan),where=den>0)
   d['flip_rate_ci95']=ci(rates[g]);d['mean_own_logit_ci95']=ci(means)
  delta=rates['MSA_in_EGY']-rates['EGY_in_MSA'];estimate=groups['MSA_in_EGY']['flip_rate']-groups['EGY_in_MSA']['flip_rate']
  mm=matched[cohort];mc={k:mm[k].pop('col') for k in mm}
  for k in mm:mm[k]['ci95_fixed_weights']=ci(boot[:,mc[k]])
  model[cohort]={'groups':groups,'directional_flip_difference':{'estimate':estimate,'ci95':ci(delta),'ci98_75':ci(delta,.9875) if cohort=='primary' else None},'matched_shift':mm,'matched_directional_difference':{'estimate':mm['MSA']['estimate']-mm['EGY']['estimate'],'ci95_fixed_weights':ci(boot[:,mc['MSA']]-boot[:,mc['EGY']])}}
 results[slug]=model
 print(slug,json.dumps({'primary':model['primary']['directional_flip_difference'],'matched':model['primary']['matched_directional_difference'],'sensitivity':model['host_margin_gt2']['directional_flip_difference']}),flush=True)
out={'audit':{'payload_hashes':len(man),'input_hash_records_match_expected':len(checks),'items_verified':total,'protocol_exact_match':True,'raw_features_not_received':True,'projection_not_recomputed_from_raw':True},'bootstrap':{'seed':20261025,'resamples':10000,'unit':'complete comment','unknown_article_author_dependence':True,'matched_weights':'fixed original type-condition counts; conditional lexical composition'},'results':results}
(B/'aoc/ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
