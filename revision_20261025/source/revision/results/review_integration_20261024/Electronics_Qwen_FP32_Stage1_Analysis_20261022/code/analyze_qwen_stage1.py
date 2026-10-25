"""Independent offline Stage1 audit and tweet-cluster analysis; no model imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse, csv, hashlib, io, json, re, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import logsumexp

ROOT=Path(os.environ.get('ELECTRONICS_AUDIT_WORKSPACE','/workspace/scratch/e3ad691229b6'))
SRC=ROOT/'work/session03_received/ELECTRONICS_QWEN_FP32_STAGE1_20261022T113131Z'
OUT=ROOT/'output/Electronics_Qwen_FP32_Stage1_Analysis_20261022'
OLD=ROOT/'work/Electronics_Four_Model_Head_Stability'
PILOT=ROOT/'upload/qwen_pilot_core/ELECTRONICS_PREDICTION_PILOT_QWEN_20261022T082339Z'
MODES=['balanced','MSA','EGY']; GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 # Analysis choices recorded before any numerical summary is calculated.
 save('ANALYSIS_SPEC.json',{
  'scope':'Post-review Qwen-only FP32 robustness; not historical replay; not independent confirmation',
  'input_sha256':'7e59e13c8aa100102dd86a9ad376b912bd4f632fe896c3faa8409c505a57615d',
  'bootstrap_seed':20261024,'resamples':10000,'interval':'pointwise percentile 95%; no multiplicity-controlled significance claims',
  'groups':GROUPS,'estimand':'item-weighted mean; resample tweets with multiplicity; preserve all eligible items; pair all conditions',
  'contrasts':['top5-baseline','mean(100 fixed historical random5)-baseline','top5-mean(random5)'],
  'replacement_comparisons':'paired MSA-balanced, EGY-balanced, EGY-MSA; full D set and all non-D metrics',
  'D':'logsumexp EGY - logsumexp MSA; sign reversed when matrix=MSA',
  'subsets':'frozen 200 nested permutations at sizes6/9; one full set12; ranked prefixes6/9 also descriptive',
  'random_reference_bands':'2.5/97.5 percentiles across 100 condition means; not tweet confidence intervals',
  'missing':'continuation role without eligible tokens is missing, not zero; bootstrap denominator uses eligible item multiplicities',
  'aliases':'use unique canonical source where available; retain ambiguous candidate tuple without choosing identity; check candidate overlap and report ambiguous exclusions',
  'historical':'aggregate comparison only where full historical item rows are unavailable; D membership not historically identified',
  'integration':'analysis outputs only; do not change original tables or close four-model reviewer items'})
 items=read(SRC/'ITEMS.json');design=read(SRC/'DESIGN.json');state=read(SRC/'STATE.json')
 assert len(items)==624 and [i['item'] for i in items]==list(range(624))
 archive=ROOT/'upload/ELECTRONICS_QWEN_FP32_STAGE1_20261022T113131Z_REVIEW_CORE_SESSION_03.zip'
 assert sha(archive.read_bytes())==read(OUT/'ANALYSIS_SPEC.json')['input_sha256']
 manifest=read(SRC/'RETURN_CONTENTS.json')
 for x in manifest['core']:assert sha((SRC/x['path']).read_bytes())==x['sha256'],x['path']
 prior_zip=ROOT/'upload/ELECTRONICS_QWEN_FP32_STAGE1_20261022T113131Z_REVIEW_CORE_SESSION_02.zip'
 assert sha(prior_zip.read_bytes())=='dcc1669f6ded24f39f7291e5d364c071b9ee7368e469746e58c436dece6fbbbe'
 with zipfile.ZipFile(prior_zip) as z:
  prefix=z.namelist()[0].split('/')[0]+'/'
  prev=json.loads(z.read(prefix+'STATE.json'))
  for rel,h in prev['outputs'].items():assert state['outputs'][rel]==h
  old_events=z.read(prefix+'actions.jsonl');assert (SRC/'actions.jsonl').read_bytes().startswith(old_events)
  previous_items=[n for n in z.namelist() if '/items/' in n]
  for n in previous_items:assert z.read(n)==(SRC/n[len(prefix):]).read_bytes()
 for sess in [2,3]:
  for p in (SRC/f'continuation_sessions/session_{sess:02d}/code').glob('*.py'):
   assert p.read_bytes()==(ROOT/'output/Electronics_Qwen_FP32_Bounded_Continuation_20261022/code'/p.name).read_bytes(),p
 for p in (SRC/'executed_code').glob('*.py'):
  assert p.read_bytes()==(ROOT/'output/Electronics_Qwen_FP32_Robustness_Stage1_20261022/code'/p.name).read_bytes(),p
 legacy=read(PILOT/'LEGACY_CONDITIONS.json')
 assert sha((PILOT/'LEGACY_CONDITIONS.json').read_bytes()) in read(SRC/'SOURCE_INPUTS.json').values()
 mapping={x['label']:x['uid'] for x in design['logical']}
 assert len(design['unique'])==len(mapping)==304 and mapping['baseline']==0
 for mode in MODES:
  for name in ['top5']+[f'random5_d{i}' for i in range(100)]:
   u=design['unique'][mapping[mode+'__'+name]]
   assert u['replacement']==mode
   assert u['heads']==sorted([list(map(int,re.fullmatch(r'L(\d+)H(\d+)',h).groups())) for h in legacy[name]['heads']])
 rng=np.random.default_rng(20261022)
 perms=[{v:rng.permutation(12).tolist() for v in ['MSA','EGY']} for _ in range(200)]
 assert perms==design['permutations']
 assert design['pools']=={v:read(SRC/'TOKEN_LISTS.json')['ranked'][v][:12] for v in ['MSA','EGY']}
 met=[];pool=[];raw=[];token_error=0.;normalizer_spread=0.
 for i,it in enumerate(items):
  with np.load(SRC/f'items/item_{i:04d}.npz',allow_pickle=False) as a:
   m=a['metrics'];p=a['pool_logprobs'];r=a['target_pool_raw_logits'];lp=a['continuation_token_logprobs']
   assert np.array_equal(a['uid'],np.arange(304)) and int(a['item'])==i
   assert m.shape==(304,4) and p.shape==r.shape==(304,24)
   assert np.isfinite(m[:,0]).all() and np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(lp).all()
   assert (p<=0).all() and (lp<=0).all()
   for j,role in enumerate(['all','own','matrix'],1):
    positions=[k-it['t']-1 for start,end,lab in it['cont'] if lab and (role=='all' or lab==it[role]) for k in range(start,end)]
    if positions:
     calc=np.ascontiguousarray(lp[:,positions]).mean(axis=1);assert np.array_equal(calc,m[:,j]);token_error=max(token_error,float(np.max(abs(calc-m[:,j]))))
    else:assert np.isnan(m[:,j]).all()
   normalizer_spread=max(normalizer_spread,float(np.ptp(r.astype(float)-p,axis=1).max()))
   assert all(x['passed'] for x in read(SRC/f'items/item_{i:04d}_checks.json'))
   met.append(m.astype(float));pool.append(p.astype(float));raw.append(r.astype(float))
 met=np.array(met);pool=np.array(pool);raw=np.array(raw)
 sign=np.array([1 if it['matrix']=='EGY' else -1 for it in items])[:,None]
 D=(logsumexp(pool[:,:,12:],axis=2)-logsumexp(pool[:,:,:12],axis=2))*sign
 Dr=(logsumexp(raw[:,:,12:],axis=2)-logsumexp(raw[:,:,:12],axis=2))*sign
 d_raw_error=float(abs(D-Dr).max())
 checks=read(SRC/'LEGACY_RANDOM_FP32_NORMS.json');ratios=[]
 head=lambda h:tuple(map(int,re.fullmatch(r'L(\d+)H(\d+)',h).groups()))
 with np.load(SRC/'reference_means_norms.npz',allow_pickle=False) as n:
  for k in range(100):
   for a,b in zip(legacy['top5']['heads'],legacy[f'random5_d{k}']['heads']):
    l,h=head(a);ll,hh=head(b);assert l==ll;ratios.append(float(n[f'norm_{ll}'][hh]/n[f'norm_{l}'][h]))
 assert np.array_equal(ratios,[x['ratio'] for x in checks]) and all(.5<=r<=2 for r in ratios)
 events=[json.loads(s) for s in (SRC/'actions.jsonl').read_text().splitlines()]
 complete=[e for e in events if e['event']=='item_complete'];assert Counter(e['item'] for e in complete)==Counter(range(624))
 assert [e['item'] for e in complete]==read(SRC/'EXECUTION_ORDER.json')
 assert not any(e['event']=='fp32_references' and e['session']!=1 for e in events)
 aliases=[{k:x[k] for k in ['item','sid','canonical_sid_candidates']} for x in items if len(x['canonical_sid_candidates'])!=1]
 group_info=[];weights={};idxs={}
 for gi,g in enumerate(GROUPS):
  ix=np.array([i for i,x in enumerate(items) if x['group']==g]);idxs[g]=ix
  # No candidate source crosses two alias IDs in any group: no invented resolution.
  owner={}
  for i in ix:
   for sid in items[i]['canonical_sid_candidates']:
    if sid in owner:assert items[owner[sid]]['sid']==items[i]['sid']
    owner[sid]=i
  keys=[tuple(items[i]['canonical_sid_candidates']) for i in ix]
  unique=list(dict.fromkeys(keys));cluster=np.array([unique.index(k) for k in keys]);C=len(unique)
  rng=np.random.default_rng(20261024+gi)
  # Per-group streams explicitly derived from the frozen seed; paired weights reused everywhere.
  draws=rng.integers(0,C,size=(10000,C));counts=np.zeros((10000,C),dtype=np.int16)
  np.add.at(counts,(np.arange(10000)[:,None],draws),1)
  weights[g]=counts[:,cluster].astype(float)
  group_info.append({'group':g,'items':len(ix),'clusters':C,'ambiguous_items':sum(len(items[i]['canonical_sid_candidates'])!=1 for i in ix),'seed':20261024+gi})
 audit={'verified':True,'payloads':len(manifest['core']),'items':624,'conditions_per_item':304,'item_condition_rows':624*304,'baseline_checks':624*8,'previous_item_files_unchanged':len(previous_items),'previous_checkpoint_hashes_preserved':len(prev['outputs']),'continuation_max_error':token_error,'pool_normalizer_spread_max':normalizer_spread,'D_normalized_vs_raw_max_error':d_raw_error,'norm_checks':500,'norm_violations':0,'reference_units_deferred':600,'reference_means_not_reconstructed_here':True,'groups':group_info,'ambiguous_aliases':aliases,'session3':read(SRC/'SESSION_3.json'),'max_recorded_driver_GiB':max(e.get('driver_bytes',0) for e in events)/2**30,'full_vocabulary_logits_available':False,'historical_replay_gate':'unresolved'}
 save('INDEPENDENT_AUDIT.json',audit)
 print('AUDIT PASSED',json.dumps(group_info),flush=True)

 def summarise(g,x):
  if x.ndim==1:x=x[:,None]
  valid=np.isfinite(x);assert np.all(valid==valid[:,:1]),'within-block missingness differs'
  denom=weights[g]@valid[:,0].astype(float)
  boot=(weights[g]@np.nan_to_num(x,nan=0.))/np.where(denom>0,denom,np.nan)[:,None]
  return np.nanmean(x,axis=0),np.nanpercentile(boot,[2.5,97.5],axis=0),int(valid[:,0].sum()),int((denom==0).sum())

 values={'own_logit':met[:,:,0],'flip_pp':100*(met[:,:,0]<0),'cont_lp_all':met[:,:,1],'cont_lp_own':met[:,:,2],'cont_lp_matrix':met[:,:,3],'D_to_matrix_n12':D}
 rows=[];replace=[];historical=[];condition_rows=[];full_contrasts={}
 for g in GROUPS:
  ix=idxs[g]
  for metric,v in values.items():
   if not np.isfinite(v[ix,0]).any():continue
   per_mode={}
   for mode in MODES:
    top=mapping[mode+'__top5'];rr=[mapping[mode+f'__random5_d{k}'] for k in range(100)]
    base=v[ix,0];tv=v[ix,top];rv=v[ix][:,rr].mean(1)
    contrasts=np.column_stack([tv-base,rv-base,tv-rv]);per_mode[mode]=contrasts
    point,ci,n,empty=summarise(g,contrasts)
    refmeans=np.nanmean(v[ix][:,rr]-base[:,None],axis=0)
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     rows.append(dict(group=g,metric=metric,replacement=mode,contrast=label,n_items=n,estimate=point[j],ci_low=ci[0,j],ci_high=ci[1,j],empty_bootstraps=empty,baseline_mean=float(np.nanmean(base)),top5_mean=float(np.nanmean(tv)),random_effect_p025=float(np.percentile(refmeans,2.5)),random_effect_p975=float(np.percentile(refmeans,97.5))))
    for k,u in [('baseline',0),('top5',top)]+[(f'random5_d{k}',u) for k,u in enumerate(rr)]:
     condition_rows.append(dict(group=g,metric=metric,replacement=mode,condition=k,n_items=n,mean=float(np.nanmean(v[ix,u])),delta=float(np.nanmean(v[ix,u]-base))))
   for a,b in [('MSA','balanced'),('EGY','balanced'),('EGY','MSA')]:
    x=per_mode[a]-per_mode[b];point,ci,n,empty=summarise(g,x)
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     replace.append(dict(group=g,metric=metric,comparison=a+' minus '+b,contrast=label,n_items=n,estimate=point[j],ci_low=ci[0,j],ci_high=ci[1,j]))
   full_contrasts[(g,metric)]=per_mode
  print('METRICS',g,flush=True)
 csvsave('METRIC_CONTRASTS.csv',rows);csvsave('REPLACEMENT_PAIRED_DIFFERENCES.csv',replace);csvsave('CONDITION_MEANS.csv',condition_rows)

 # Every frozen subset is assessed, without repeating any model evaluation.
 subsets=[]
 for n in [6,9]:
  subsets.append((n,'ranked',-1,list(range(n)),list(range(12,12+n))))
  for k,p in enumerate(perms):subsets.append((n,'permutation',k,p['MSA'][:n],[12+j for j in p['EGY'][:n]]))
 subsets.append((12,'full',-1,list(range(12)),list(range(12,24))))
 S=len(subsets);effects={m:np.empty((624,S,3)) for m in MODES};baselines=np.empty((624,S));nullbands={}
 for s,(n,kind,k,a,b) in enumerate(subsets):
  ds=(logsumexp(pool[:,:,b],axis=2)-logsumexp(pool[:,:,a],axis=2))*sign
  baselines[:,s]=ds[:,0]
  for mode in MODES:
   t=mapping[mode+'__top5'];rr=[mapping[mode+f'__random5_d{k}'] for k in range(100)]
   r=ds[:,rr].mean(1);effects[mode][:,s,:]=np.column_stack([ds[:,t]-ds[:,0],r-ds[:,0],ds[:,t]-r])
   for g in GROUPS:
    ix=idxs[g];nullbands[(g,mode,s)]=np.percentile((ds[ix][:,rr]-ds[ix,0,None]).mean(0),[2.5,97.5])
 drows=[];composition=[]
 for g in GROUPS:
  ix=idxs[g]
  for mode in MODES:
   x=effects[mode][ix].reshape(len(ix),-1);point,ci,n,empty=summarise(g,x)
   point=point.reshape(S,3);ci=ci.reshape(2,S,3)
   for s,(size,kind,k,a,b) in enumerate(subsets):
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     band=nullbands[(g,mode,s)]
     drows.append(dict(group=g,replacement=mode,size=size,subset_kind=kind,permutation=k,contrast=label,n_items=n,estimate=point[s,j],ci_low=ci[0,s,j],ci_high=ci[1,s,j],baseline_D=float(baselines[ix,s].mean()),random_effect_p025=band[0],random_effect_p975=band[1]))
   for size in [6,9,12]:
    ss=[s for s,q in enumerate(subsets) if q[0]==size and q[1]==('full' if size==12 else 'permutation')]
    uniq=len({(tuple(sorted(subsets[s][3])),tuple(sorted(subsets[s][4]))) for s in ss})
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     z=point[ss,j];composition.append(dict(group=g,replacement=mode,size=size,contrast=label,draws=len(ss),unique_set_pairs=uniq,min=float(z.min()),p025=float(np.percentile(z,2.5)),median=float(np.median(z)),p975=float(np.percentile(z,97.5)),max=float(z.max()),positive=int((z>0).sum()),negative=int((z<0).sum()),zero=int((z==0).sum()),opposite_sign_to_full=int((np.sign(z)!=np.sign(point[-1,j])).sum())))
   print('D',g,mode,flush=True)
 csvsave('D_SUBSET_CONTRASTS.csv',drows);csvsave('D_COMPOSITION_SUMMARY.csv',composition)
 save('D_SUBSET_IDS.json',[dict(size=n,kind=kind,permutation=k,MSA=[design['pools']['MSA'][i] for i in a],EGY=[design['pools']['EGY'][i-12] for i in b]) for n,kind,k,a,b in subsets])

 # Descriptive historical aggregate comparison, verified against pinned input hashes.
 hist=OLD/'results/exp06/Qwen3-8B-Base/B_summary.csv'
 assert sha(hist.read_bytes()) in read(SRC/'SOURCE_INPUTS.json').values()
 with hist.open() as f:old=list(csv.DictReader(f))
 for g in GROUPS:
  for metric in ['own_logit','flip_pp','cont_lp_all','cont_lp_own','cont_lp_matrix']:
   ix=idxs[g];v=values[metric];t=mapping['balanced__top5']
   o=next(x for x in old if x['group']==g and x['condition']=='top5')
   ob=next(x for x in old if x['group']==g and x['condition']=='baseline')
   if not np.isfinite(v[ix,0]).any():continue
   if metric=='flip_pp':h=(float(o['flip_rate'])-float(ob['flip_rate']))*100
   else:h=float(o['d_'+metric])
   new=float(np.nanmean(v[ix,t]-v[ix,0]));historical.append(dict(group=g,metric=metric,historical_top5_delta=h,new_FP32_balanced_top5_delta=new,difference=new-h,scope='aggregate descriptive; precision and reference means changed together; not historical replay'))
 csvsave('HISTORICAL_AGGREGATE_COMPARISON.csv',historical)
 # Exclude the two ambiguous monolingual source mappings as a transparent point sensitivity.
 sensitivity=[]
 for metric,v in values.items():
  g='mono_MSA';ix=idxs[g];keep=np.array([i for i in ix if len(items[i]['canonical_sid_candidates'])==1])
  for mode in MODES:
   t=mapping[mode+'__top5'];rr=[mapping[mode+f'__random5_d{k}'] for k in range(100)]
   net=v[:,t]-v[:,rr].mean(1)
   sensitivity.append(dict(group=g,metric=metric,replacement=mode,all_items=len(ix),retained_items=len(keep),full_net=float(np.nanmean(net[ix])),exclude_ambiguous_net=float(np.nanmean(net[keep]))))
 csvsave('AMBIGUOUS_ALIAS_EXCLUSION.csv',sensitivity)
 save('ANALYSIS_COMPLETE.json',{'offline_analysis_complete':True,'model_forward_calls':0,'bootstrap_resamples_per_group':10000,'bootstrap_seed_rule':'20261024 + group index in fixed GROUPS order','pointwise_intervals_only':True,'four_model_R1_3_closed':False,'four_model_R1_8_closed':False,'manuscript_integrated':False,'historical_gate_resolved':False})
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
