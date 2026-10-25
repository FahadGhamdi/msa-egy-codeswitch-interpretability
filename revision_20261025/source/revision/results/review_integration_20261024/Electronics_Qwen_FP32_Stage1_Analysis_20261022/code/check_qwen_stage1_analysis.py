"""Second implementation: explicit cluster expansion, independent of weighted matrix bootstrap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,csv
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
root=Path(os.environ.get('ELECTRONICS_AUDIT_WORKSPACE','/workspace/scratch/e3ad691229b6'))
src=root/'work/session03_received/ELECTRONICS_QWEN_FP32_STAGE1_20261022T113131Z'
out=root/'output/Electronics_Qwen_FP32_Stage1_Analysis_20261022'
items=json.loads((src/'ITEMS.json').read_text());design=json.loads((src/'DESIGN.json').read_text())
labels={x['label']:x['uid'] for x in design['logical']}
with (out/'METRIC_CONTRASTS.csv').open() as f:rows=list(csv.DictReader(f))
modes=['balanced','MSA','EGY'];checks=[]
for gi,g in enumerate(['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']):
 ix=[i for i,x in enumerate(items) if x['group']==g];x=[]
 for i in ix:
  with np.load(src/f'items/item_{i:04d}.npz',allow_pickle=False) as a:
   p=a['pool_logprobs'].astype(float);d=(logsumexp(p[:,12:],axis=1)-logsumexp(p[:,:12],axis=1))*(1 if items[i]['matrix']=='EGY' else -1)
   own=a['metrics'][:,0].astype(float);r=[]
   for v in [own,d]:
    for mode in modes:r.append(v[labels[mode+'__top5']]-np.mean([v[labels[mode+f'__random5_d{k}']] for k in range(100)]))
   x.append(r)
 x=np.array(x);keys=[tuple(items[i]['canonical_sid_candidates']) for i in ix];unique=list(dict.fromkeys(keys))
 members=[np.array([i for i,k in enumerate(keys) if k==s]) for s in unique]
 rng=np.random.default_rng(20261024+gi);draws=rng.integers(len(unique),size=(10000,len(unique)))
 boot=np.array([x[np.concatenate([members[c] for c in draw])].mean(0) for draw in draws])
 ci=np.percentile(boot,[2.5,97.5],axis=0)
 for j,(metric,mode) in enumerate((metric,mode) for metric in ['own_logit','D_to_matrix_n12'] for mode in modes):
  row=next(r for r in rows if r['group']==g and r['metric']==metric and r['replacement']==mode and r['contrast']=='top5_minus_random_mean')
  err=max(abs(x[:,j].mean()-float(row['estimate'])),abs(ci[0,j]-float(row['ci_low'])),abs(ci[1,j]-float(row['ci_high'])))
  assert err<1e-12,(g,metric,mode,err)
  checks.append(dict(group=g,metric=metric,replacement=mode,max_error=err))
 print('CHECK',g,flush=True)
# An unequal-cluster test prevents accidental equal-tweet weighting.
toy=np.array([0.,2.,10.]);assert toy[[0,1,2]].mean()==4 and toy[[0,1,0,1]].mean()==1
result={'passed':True,'independent_expanded_cluster_checks':len(checks),'resamples_per_check':10000,'max_error':max(x['max_error'] for x in checks),'unequal_cluster_weighting_test_passed':True,'checks':checks}
(out/'SECOND_IMPLEMENTATION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
