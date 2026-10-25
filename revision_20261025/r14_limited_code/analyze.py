"""CPU paired analysis; no forwards. Output remains pending independent review."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,sqlite3,hashlib,io,csv
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from pilot_support import sha,save,metrics_from_tokens
from run_model import check_bridge,equal,ARRAYS,GROUPS
P=Path(__file__).resolve().parent

def summarize(x,keys,seed=20261025):
 valid=np.isfinite(x)
 if not np.all(valid==valid[:,:1]):raise RuntimeError('Inconsistent paired missingness')
 keep=valid[:,0];n=int(keep.sum())
 if not n:return None
 unique=list(dict.fromkeys(keys));lookup={k:i for i,k in enumerate(unique)};cluster=np.array([lookup[k] for k in keys]);C=len(unique)
 counts=np.bincount(cluster[keep],minlength=C);sums=np.zeros((C,x.shape[1]));np.add.at(sums,cluster,np.nan_to_num(x,nan=0))
 rng=np.random.default_rng(seed);boot=[];empty=0
 for start in range(0,10000,100):
  ix=rng.integers(0,C,size=(min(100,10000-start),C));den=counts[ix].sum(1);num=sums[ix].sum(1);empty+=int((den==0).sum())
  boot.append(np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0))
 ci=np.nanpercentile(np.concatenate(boot),[2.5,97.5],axis=0)
 return np.nanmean(x,axis=0),ci,n,C,empty

def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);a=ap.parse_args();root=a.input.resolve();state=json.loads((root/'STATE.json').read_text());model=state['model'];src=P/'models'/model
 assert state['status']=='COMPLETE'
 for rel,h in json.loads((root/'OUTPUT_MANIFEST.json').read_text()).items():assert sha(root/rel)==h,rel
 for rel,h in json.loads((P/'MANIFEST.json').read_text()).items():assert sha(P/rel)==h,rel
 out=root/'ANALYSIS';out.mkdir(exist_ok=False)
 d=json.loads((src/'LIMITED_DESIGN.json').read_text());items=json.loads((src/'ITEMS.json').read_text());old=json.loads((src/'DESIGN.json').read_text());ks=[1,10,20];N=len(old['pools']['MSA']);active=[d['batches'][i] for i in d['active_batches']]
 conn=sqlite3.connect(f'file:{root/"RESULTS.sqlite"}?mode=ro',uri=True)
 assert json.loads(conn.execute("SELECT value FROM meta WHERE key='status'").fetchone()[0])=='COMPLETE'
 assert conn.execute('SELECT count(*) FROM batches').fetchone()[0]==len(items)*(len(active)+1)
 meta=json.loads(conn.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]);assert meta['package']==sha(P/'MANIFEST.json') and meta['model']==model
 selected=[];randoms=[];baseline=[];maxD=0.;batches_checked=0
 def values(z,it):
  nonlocal maxD
  m=z['metrics'];tok=z['continuation_token_logprobs'];pool=z['pool_logprobs'];raw=z['target_pool_raw_logits']
  assert np.array_equal(metrics_from_tokens(it,m[:,0],tok,np),m,equal_nan=True)
  sign=1 if it['matrix']=='EGY' else -1
  D=(logsumexp(pool[:,N:].astype(float),axis=1)-logsumexp(pool[:,:N].astype(float),axis=1))*sign
  Dr=(logsumexp(raw[:,N:].astype(float),axis=1)-logsumexp(raw[:,:N].astype(float),axis=1))*sign
  err=float(np.max(np.abs(D-Dr)));maxD=max(maxD,err);assert err<=1e-4
  return np.column_stack([m[:,0],100.*(m[:,0]<0),m[:,1:],D])
 for i,it in enumerate(items):
  with np.load(src/'anchors'/f'item_{i:04d}.npz') as z:anchor={k:z[k].copy() for k in ARRAYS}
  with np.load(src/'selected'/f'item_{i:04d}.npz') as z:
   assert int(z['item'])==i and np.array_equal(z['k'],np.arange(21))
   for k in ARRAYS:assert np.array_equal(z[k][[0,5]],anchor[k],equal_nan=True)
   vv=values(z,it);selected.append(vv[ks]);baseline.append(vv[0])
  cond={}
  for task in [{'id':-1,'k':0,'condition_ids':[]}]+active:
   row=conn.execute('SELECT payload,digest,report FROM batches WHERE item=? AND batch=?',(i,task['id'])).fetchone();assert row is not None
   payload,h,rep=row;assert hashlib.sha256(payload).hexdigest()==h;report=json.loads(rep);assert report['condition_ids']==task['condition_ids'] and report['baseline_checks_passed'] and report['baseline_flip_identical']
   with np.load(io.BytesIO(payload)) as z:
    assert int(z['item'])==i and int(z['k'])==task['k'];assert np.array_equal(z['condition_ids'],task['condition_ids'])
    vv=values(z,it)
    if task['id']==-1:
     assert check_bridge(anchor,{k:z[k][:2] for k in ARRAYS},np)[0];pads=[0,2,3,4,5,6,7]
    else:
     ids=task['condition_ids'];pads=[0]+list(range(1+len(ids),8))
     for j,cid in enumerate(ids,1):assert cid not in cond;cond[cid]=vv[j]
    for j in pads:
     for k in ARRAYS:assert equal(z[k][j],anchor[k][0],np)
     assert (z['metrics'][j,0]<0)==(anchor['metrics'][0,0]<0)
   batches_checked+=1
  randoms.append([np.mean([cond[d['logical_to_unique'][f'{k}:{draw}']] for draw in range(100)],axis=0) for k in ks])
 conn.close();selected=np.array(selected);randoms=np.array(randoms);baseline=np.array(baseline)
 names=['own_logit','flip_pp','cont_lp_all','cont_lp_own','cont_lp_matrix',f'D_to_matrix_n{N}'];rows=[];sensitivity=[]
 np.savez_compressed(out/'PAIRED_VALUES.npz',selected=selected,random_mean=randoms,baseline=baseline,k=ks)
 for gi,g in enumerate(GROUPS):
  ix=np.array([i for i,it in enumerate(items) if it['group']==g]);keys=[tuple(items[i]['canonical_sid_candidates']) for i in ix];assert all(keys)
  owners={}
  for i,key in zip(ix,keys):
   for sid in key:
    if sid in owners:assert owners[sid]==key
    owners[sid]=key
  for mi,name in enumerate(names):
   net=selected[ix,:,mi]-randoms[ix,:,mi];summ=summarize(net,keys,20261025+gi)
   if summ is None:continue
   point,ci,n,C,empty=summ
   for j,k in enumerate(ks):rows.append({'model':model,'group':g,'metric':name,'k':k,'n_items':n,'clusters':C,'selected_delta':float(np.nanmean(selected[ix,j,mi]-baseline[ix,mi])),'random_mean_delta':float(np.nanmean(randoms[ix,j,mi]-baseline[ix,mi])),'selected_minus_random':float(point[j]),'ci_low':float(ci[0,j]),'ci_high':float(ci[1,j]),'empty_bootstraps':empty,'interval':'pointwise95_unadjusted','draws':100})
   keep=np.array([len(x)==1 for x in keys]);ret=ix[keep]
   if len(ret) and not keep.all():
    sub=summarize(selected[ret,:,mi]-randoms[ret,:,mi],[key for key,q in zip(keys,keep) if q],20261025+gi)
    if sub:
     pp,cc,nn,cl,ee=sub
     for j,k in enumerate(ks):sensitivity.append({'model':model,'group':g,'metric':name,'k':k,'n_items':nn,'selected_minus_random':float(pp[j]),'ci_low':float(cc[0,j]),'ci_high':float(cc[1,j])})
 write_csv(out/'LIMITED_RANDOM_CONTRASTS.csv',rows);write_csv(out/'ALIAS_SENSITIVITY.csv',sensitivity)
 save(out/'AUDIT.json',{'model':model,'items':len(items),'batches_checked':batches_checked,'raw_normalized_D_max_error':maxD,'bootstrap_resamples':10000,'seed':'20261025+group_index','random_draws_fixed_not_resampled':True,'interval_scope':'paired tweet clusters conditional on frozen random draws; pointwise95, not simultaneous','independent_review_pending':True,'R1_4_closed':False})
 save(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='MANIFEST.json'})
 print(json.dumps({'model':model,'analysis_rows':len(rows),'independent_review_pending':True}),flush=True)
if __name__=='__main__':main()
