"""Offline ALLaM analysis, same estimands as prior Qwen; no model inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import csv,json,hashlib,gzip
import numpy as np
from scipy.special import logsumexp
ROOT=Path('/workspace/scratch/e3ad691229b6')
SRC=ROOT/'work/three_model_received_20261023/ALLAM/unpacked/ELECTRONICS_ALLAM_FP32_STAGE1_20261023_NIGHT'
OUT=ROOT/'output/Electronics_Three_Model_Review_20261023'
MODES=['balanced','MSA','EGY']; GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 audit=read(OUT/'INDEPENDENT_NIGHT_AUDIT.json');assert audit['ALLaM']['completed_items']==669 and audit['ALLaM']['stored_outputs_verified']
 save('ANALYSIS_SPEC.json',{'model':'ALLaM','scope':'Declared post-review FP32 robustness, not historical replay','choice_time':'Analysis choices written before numerical summaries in this run; original study and Qwen findings already known','bootstrap_seed_rule':'20261024+group_index','resamples':10000,'interval':'pointwise percentile95%, no multiplicity adjustment','estimand':'item weighted means; paired tweet-cluster bootstrap, conditional on fixed heads, references, model and selected items','D':'logsumexp EGY minus MSA; orient toward matrix; pools50, nested sizes12/25,200 frozen permutations, full50 one unique set','contrasts':['top5-baseline','mean100historicalrandom-baseline','top5-mean100historicalrandom'],'replacement_comparisons':['MSA-balanced','EGY-balanced','EGY-MSA'],'missingness':'no eligible continuation token is missing, not zero','historical':'paired current FP32/new means versus historical rows; precision and means change together; D not compared because historical membership unresolved','aliases':'retain ambiguous candidate tuples; report exclusion sensitivity','input_archive_sha256':audit['archive_sha256'],'model_forwards':0})
 items=read(SRC/'ITEMS.json');design=read(SRC/'DESIGN.json');assert len(items)==669
 mapping={x['label']:x['uid'] for x in design['logical']};assert len(mapping)==304
 rng=np.random.default_rng(20261022);perms=[{v:rng.permutation(50).tolist() for v in ['MSA','EGY']} for _ in range(200)];assert perms==design['permutations'];assert design['sizes']==[12,25,50]
 assert design['pools']=={v:read(SRC/'TOKEN_LISTS.json')['ranked'][v][:50] for v in ['MSA','EGY']}
 met=[];pool=[];raw=[]
 for i in range(669):
  with np.load(SRC/f'items/item_{i:04d}.npz',allow_pickle=False) as z:met.append(z['metrics'].astype(float));pool.append(z['pool_logprobs'].astype(float));raw.append(z['target_pool_raw_logits'].astype(float))
 met=np.array(met);pool=np.array(pool);raw=np.array(raw);sign=np.array([1 if it['matrix']=='EGY' else -1 for it in items])[:,None]
 D=(logsumexp(pool[:,:,50:],axis=2)-logsumexp(pool[:,:,:50],axis=2))*sign
 Dr=(logsumexp(raw[:,:,50:],axis=2)-logsumexp(raw[:,:,:50],axis=2))*sign
 save('D_RAW_CROSSCHECK.json',{'normalized_vs_raw_max_abs_error':float(abs(D-Dr).max()),'full_vocabulary_available':False})
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
 save('GROUPS_AND_ALIASES.json',{'groups':group_info,'aliases':aliases})
 def summarise(g,x):
  if x.ndim==1:x=x[:,None]
  valid=np.isfinite(x);assert np.all(valid==valid[:,:1]),'within-block missingness differs'
  denom=weights[g]@valid[:,0].astype(float)
  boot=(weights[g]@np.nan_to_num(x,nan=0.))/np.where(denom>0,denom,np.nan)[:,None]
  return np.nanmean(x,axis=0),np.nanpercentile(boot,[2.5,97.5],axis=0),int(valid[:,0].sum()),int((denom==0).sum())

 values={'own_logit':met[:,:,0],'flip_pp':100*(met[:,:,0]<0),'cont_lp_all':met[:,:,1],'cont_lp_own':met[:,:,2],'cont_lp_matrix':met[:,:,3],'D_to_matrix_n50':D}
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
 for n in [12,25]:
  subsets.append((n,'ranked',-1,list(range(n)),list(range(50,50+n))))
  for k,p in enumerate(perms):subsets.append((n,'permutation',k,p['MSA'][:n],[50+j for j in p['EGY'][:n]]))
 subsets.append((50,'full',-1,list(range(50)),list(range(50,100))))
 S=len(subsets);effects={m:np.empty((669,S,3)) for m in MODES};baselines=np.empty((669,S));nullbands={}
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
   for size in [12,25,50]:
    ss=[s for s,q in enumerate(subsets) if q[0]==size and q[1]==('full' if size==50 else 'permutation')]
    uniq=len({(tuple(sorted(subsets[s][3])),tuple(sorted(subsets[s][4]))) for s in ss})
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     z=point[ss,j];composition.append(dict(group=g,replacement=mode,size=size,contrast=label,draws=len(ss),unique_set_pairs=uniq,min=float(z.min()),p025=float(np.percentile(z,2.5)),median=float(np.median(z)),p975=float(np.percentile(z,97.5)),max=float(z.max()),positive=int((z>0).sum()),negative=int((z<0).sum()),zero=int((z==0).sum()),opposite_sign_to_full=int((np.sign(z)!=np.sign(point[-1,j])).sum())))
   print('D',g,mode,flush=True)
 csvsave('D_SUBSET_CONTRASTS.csv',drows);csvsave('D_COMPOSITION_SUMMARY.csv',composition)
 save('D_SUBSET_IDS.json',[dict(size=n,kind=kind,permutation=k,MSA=[design['pools']['MSA'][i] for i in a],EGY=[design['pools']['EGY'][i-50] for i in b]) for n,kind,k,a,b in subsets])

 sensitivity=[]
 for metric,v in values.items():
  g='mono_MSA';ix=idxs[g];keep=np.array([i for i in ix if len(items[i]['canonical_sid_candidates'])==1])
  for mode in MODES:
   t=mapping[mode+'__top5'];rr=[mapping[mode+f'__random5_d{k}'] for k in range(100)]
   net=v[:,t]-v[:,rr].mean(1)
   sensitivity.append(dict(group=g,metric=metric,replacement=mode,all_items=len(ix),retained_items=len(keep),full_net=float(np.nanmean(net[ix])),exclude_ambiguous_net=float(np.nanmean(net[keep]))))
 csvsave('AMBIGUOUS_ALIAS_EXCLUSION.csv',sensitivity)
 # Paired comparison against saved historical target rows; no D identity claim.
 names=['baseline','top5']+[f'random5_d{k}' for k in range(100)];h=np.full((669,102,4),np.nan);seen=set()
 with gzip.open(SRC/'HISTORICAL_TARGET_ROWS.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   i=int(r['item']);j=names.index(r['condition']);assert (i,j) not in seen;seen.add((i,j));assert all(str(items[i][k])==r[k] for k in ['sid','word','group'])
   h[i,j]=[float(r[k]) if r[k] else np.nan for k in ['own_logit','cont_lp_all','cont_lp_own','cont_lp_matrix']]
 assert len(seen)==669*102
 uids=[0,mapping['balanced__top5']]+[mapping[f'balanced__random5_d{k}'] for k in range(100)];new=met[:,uids];assert np.array_equal(np.isnan(h),np.isnan(new))
 changes=(h[:,:,0]<0)!=(new[:,:,0]<0)
 summary={'paired_rows':669*102,'flip_changed_rows':int(changes.sum()),'flip_changed_items':int(changes.any(1).sum()),'changed_by_condition':{'baseline':int(changes[:,0].sum()),'top5':int(changes[:,1].sum()),'random':int(changes[:,2:].sum())},'metrics':{},'scope':'descriptive precision+reference means jointly changed; not historical PASS; historical D excluded'}
 histrows=[]
 for j,metric in enumerate(['own_logit','cont_lp_all','cont_lp_own','cont_lp_matrix']):
  delta_old=h[:,:,j]-h[:,0:1,j];delta_new=new[:,:,j]-new[:,0:1,j];valid=np.isfinite(h[:,:,j]);d=abs(new[:,:,j]-h[:,:,j]);flip=(delta_old[:,1]*delta_new[:,1]<0)
  summary['metrics'][metric]={'valid_rows':int(valid.sum()),'max_abs_diff':float(np.nanmax(d)) if valid.any() else None,'top5_delta_sign_reversals':int(flip.sum()),'max_baseline_abs_diff':float(np.nanmax(d[:,0])) if np.isfinite(d[:,0]).any() else None}
  for g in GROUPS:
   ix=idxs[g]
   if not np.isfinite(h[ix,0,j]).any():continue
   oldnet=h[ix,1,j]-h[ix,2:,j].mean(1);newnet=new[ix,1,j]-new[ix,2:,j].mean(1)
   vals=np.column_stack([oldnet,newnet,newnet-oldnet]);point,ci,n,empty=summarise(g,vals)
   for k,label in enumerate(['historical_net','FP32_net','FP32_minus_historical_net']):histrows.append(dict(group=g,metric=metric,contrast=label,n_items=n,estimate=point[k],ci_low=ci[0,k],ci_high=ci[1,k]))
 save('HISTORICAL_PAIRED_AUDIT.json',summary);csvsave('HISTORICAL_NET_COMPARISON.csv',histrows)
 save('ANALYSIS_COMPLETE.json',{'offline_analysis_complete':True,'model_forward_calls':0,'bootstrap_resamples_per_group':10000,'pointwise_intervals_only':True,'four_model_R1_3_closed':False,'four_model_R1_8_closed':False,'manuscript_integrated':False,'historical_gate_resolved':False})
 print('ALLAM ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
