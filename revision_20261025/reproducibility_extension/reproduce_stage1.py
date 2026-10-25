"""Portable Stage1 statistical reanalysis from frozen per-item arrays.
Reuses the audited estimator code; it does not reconstruct reference means,
re-run model inference, or certify historical replay.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from scipy.special import logsumexp
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--expected',type=Path,required=True,help='Previously audited analysis directory, used for numeric comparison')
a=p.parse_args();EXPECTED=a.expected.resolve();SRC=a.input.resolve();OUT=a.out.resolve()
if OUT.exists():p.error('Output exists; use a new directory')
if SRC==OUT or SRC in OUT.parents:p.error('Output must be outside the frozen input')
OUT.mkdir(parents=True)
MODES=['balanced','MSA','EGY']; GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 items=read(SRC/'ITEMS.json');design=read(SRC/'DESIGN.json');N=len(items);poolN=len(design['pools']['MSA'])
 assert len(design['pools']['EGY'])==poolN and poolN in [12,50] and N in [624,669,654]
 mapping={x['label']:x['uid'] for x in design['logical']};assert len(mapping)==304
 rng=np.random.default_rng(20261022);perms=[{v:rng.permutation(poolN).tolist() for v in ['MSA','EGY']} for _ in range(200)];assert perms==design['permutations']
 assert design['sizes']==([6,9,12] if poolN==12 else [12,25,50])
 met=[];pool=[];raw=[];hashes={};token_error=0.
 for i,it in enumerate(items):
  f=SRC/f'items/item_{i:04d}.npz';hashes[str(f.relative_to(SRC))]=sha(f.read_bytes())
  with np.load(f,allow_pickle=False) as z:
   assert int(z['item'])==i
   m=z['metrics'];tok=z['continuation_token_logprobs']
   for j,role in enumerate(['all','own','matrix'],1):
    positions=[q-it['t']-1 for lo,hi,lab in it['cont'] if lab and (role=='all' or lab==it[role]) for q in range(lo,hi)]
    if positions:
     rebuilt=np.stack([tok[:,q] for q in positions],axis=1).mean(axis=1,dtype=np.float32)
     assert np.array_equal(rebuilt,m[:,j]),(i,role)
    else:assert np.isnan(m[:,j]).all()
   met.append(m.astype(float));pool.append(z['pool_logprobs'].astype(float));raw.append(z['target_pool_raw_logits'].astype(float))
 met=np.array(met);pool=np.array(pool);raw=np.array(raw);sign=np.array([1 if it['matrix']=='EGY' else -1 for it in items])[:,None]
 D=(logsumexp(pool[:,:,poolN:],axis=2)-logsumexp(pool[:,:,:poolN],axis=2))*sign
 Dr=(logsumexp(raw[:,:,poolN:],axis=2)-logsumexp(raw[:,:,:poolN],axis=2))*sign
 assert float(abs(D-Dr).max())<1e-4
 save('D_RAW_CROSSCHECK.json',{'normalized_vs_raw_max_abs_error':float(abs(D-Dr).max()),'full_vocabulary_available':False})
 save('INPUT_ARRAY_HASHES.json',hashes)
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

 values={'own_logit':met[:,:,0],'flip_pp':100*(met[:,:,0]<0),'cont_lp_all':met[:,:,1],'cont_lp_own':met[:,:,2],'cont_lp_matrix':met[:,:,3],f'D_to_matrix_n{poolN}':D}
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
 for n in design['sizes'][:-1]:
  subsets.append((n,'ranked',-1,list(range(n)),list(range(poolN,poolN+n))))
  for k,p in enumerate(perms):subsets.append((n,'permutation',k,p['MSA'][:n],[poolN+j for j in p['EGY'][:n]]))
 subsets.append((poolN,'full',-1,list(range(poolN)),list(range(poolN,2*poolN))))
 S=len(subsets);effects={m:np.empty((N,S,3)) for m in MODES};baselines=np.empty((N,S));nullbands={}
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
   for size in design['sizes']:
    ss=[s for s,q in enumerate(subsets) if q[0]==size and q[1]==('full' if size==poolN else 'permutation')]
    uniq=len({(tuple(sorted(subsets[s][3])),tuple(sorted(subsets[s][4]))) for s in ss})
    for j,label in enumerate(['top5_minus_baseline','random_mean_minus_baseline','top5_minus_random_mean']):
     z=point[ss,j];composition.append(dict(group=g,replacement=mode,size=size,contrast=label,draws=len(ss),unique_set_pairs=uniq,min=float(z.min()),p025=float(np.percentile(z,2.5)),median=float(np.median(z)),p975=float(np.percentile(z,97.5)),max=float(z.max()),positive=int((z>0).sum()),negative=int((z<0).sum()),zero=int((z==0).sum()),opposite_sign_to_full=int((np.sign(z)!=np.sign(point[-1,j])).sum())))
   print('D',g,mode,flush=True)
 csvsave('D_SUBSET_CONTRASTS.csv',drows);csvsave('D_COMPOSITION_SUMMARY.csv',composition)
 save('D_SUBSET_IDS.json',[dict(size=n,kind=kind,permutation=k,MSA=[design['pools']['MSA'][i] for i in a],EGY=[design['pools']['EGY'][i-poolN] for i in b]) for n,kind,k,a,b in subsets])

 sensitivity=[]
 for metric,v in values.items():
  g='mono_MSA';ix=idxs[g];keep=np.array([i for i in ix if len(items[i]['canonical_sid_candidates'])==1])
  for mode in MODES:
   t=mapping[mode+'__top5'];rr=[mapping[mode+f'__random5_d{k}'] for k in range(100)]
   net=v[:,t]-v[:,rr].mean(1)
   sensitivity.append(dict(group=g,metric=metric,replacement=mode,all_items=len(ix),retained_items=len(keep),full_net=float(np.nanmean(net[ix])),exclude_ambiguous_net=float(np.nanmean(net[keep]))))
 csvsave('AMBIGUOUS_ALIAS_EXCLUSION.csv',sensitivity)

 files=['METRIC_CONTRASTS.csv','REPLACEMENT_PAIRED_DIFFERENCES.csv','CONDITION_MEANS.csv','D_SUBSET_CONTRASTS.csv','D_COMPOSITION_SUMMARY.csv','AMBIGUOUS_ALIAS_EXCLUSION.csv']
 checks=[]
 for name in files:
  expected=list(csv.DictReader((EXPECTED/name).open()));actual=list(csv.DictReader((OUT/name).open()));assert len(expected)==len(actual),(name,len(expected),len(actual))
  err=0.;numeric=0
  for row1,row2 in zip(expected,actual):
   assert row1.keys()==row2.keys(),name
   for k,v in row1.items():
    try:x=float(v);y=float(row2[k])
    except ValueError:assert v==row2[k],(name,k,v,row2[k]);continue
    if np.isnan(x):assert np.isnan(y)
    else:err=max(err,abs(x-y));assert np.isclose(x,y,atol=1e-10,rtol=0),(name,k,x,y)
    numeric+=1
  checks.append({'file':name,'rows':len(actual),'numeric_checks':numeric,'max_abs_error':err})
 save('PORTABLE_RECOMPUTATION.json',{'items':N,'conditions':met.shape[1],'pool_size':poolN,'checks':checks,'model_forwards':0,'continuation_recomputed_exactly':True,'historical_replay_resolved':False,'reference_means_recomputed':False})
 print('PORTABLE STAGE1 PASS',N,flush=True)
if __name__=='__main__':main()
