import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,csv,hashlib,zipfile,io
import numpy as np
from scipy.special import logsumexp
import argparse
parser=argparse.ArgumentParser(description='Recompute selected-head curves from the frozen returned arrays; no model inference.')
parser.add_argument('--return-root', type=Path, required=True)
parser.add_argument('--issued-zip', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args=parser.parse_args()
ROOT=args.return_root.resolve();ISSUED=args.issued_zip.resolve();OUT=args.out.resolve()
if not ROOT.is_dir() or not ISSUED.is_file():parser.error('The frozen return directory and issued ZIP must exist.')
if OUT.exists():parser.error('Output directory already exists; choose a new directory.')
if ROOT==OUT or ROOT in OUT.parents:parser.error('Output must be outside the immutable return directory.')
OUT.mkdir(parents=True)
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert sha(ISSUED.read_bytes())=='3e7bdc0d012a7b1feca6ee9d4ed33cb3ebdc2936251cdebe74c2912167075692'
Z=zipfile.ZipFile(ISSUED);PREFIX='Electronics_Selected_Curve_20261024/'
manifest=read(ROOT/'RETURN_MANIFEST.json');actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()};assert actual==set(manifest)|{'RETURN_MANIFEST.json'}
for rel,h in manifest.items():assert sha((ROOT/rel).read_bytes())==h,rel
GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY'];MODELS=['Qwen3-8B-Base','ALLaM-7B-Instruct-preview','gemma-2-9b','Fanar-1-9B'];result={};allrows=[];sensitivity=[];shapes=[]
for model in MODELS:
 p=ROOT/model;items=read(p/'ITEMS.json');design=read(p/'DESIGN.json');state=read(p/'STATE.json');session=read(p/'SESSION.json');N=len(items)
 assert sorted(state['completed_items'])==list(range(N)) and len(state['completed_items'])==N
 assert session['forward_calls']==3*N and session['reference_forwards']==0 and session['error'] is None
 for kind in ['OUTPUT_MANIFEST.json']:
  for rel,h in read(p/kind).items():assert sha((p/rel).read_bytes())==h,(model,rel)
 for rel,h in state['outputs'].items():assert sha((p/rel).read_bytes())==h
 for name in ['ITEMS.json','DESIGN.json','reference_means_norms.npz','ANCHOR_PROVENANCE.json']:
  assert (p/name).read_bytes()==Z.read(PREFIX+'models/'+model+'/'+name)
 codes=0
 for f in (p/'executed_package').rglob('*'):
  if f.is_file():assert f.read_bytes()==Z.read(PREFIX+str(f.relative_to(p/'executed_package'))),(model,f);codes+=1
 expected=json.loads(Z.read(PREFIX+'models/'+model+'/RUNTIME.json'));runtime=read(p/'RUNTIME.json');assert all(runtime[k]==v for k,v in expected.items());assert runtime['device']=='mps' and runtime['dtype']=='torch.float32'
 assert len(design['heads'])==21 and all(design['heads'][k]==design['heads'][20][:k] for k in range(21))
 met=[];pl=[];raw=[];token_values=0;bridge_max={};bridge_count=0;anchorchecks=0
 for i,it in enumerate(items):
  f=p/'items'/f'item_{i:04d}.npz'
  with np.load(f,allow_pickle=False) as z:
   assert int(z['item'])==i and np.array_equal(z['k'],np.arange(21));m=z['metrics'];tok=z['continuation_token_logprobs'];lp=z['pool_logprobs'];rr=z['target_pool_raw_logits']
   assert m.shape==(21,4) and np.isfinite(m[:,0]).all() and np.isfinite(tok).all() and np.isfinite(lp).all() and np.isfinite(rr).all()
   assert tok.shape==(21,len(it['ids'])-it['t']-1)
   recompute=np.full((21,4),np.nan,np.float32);recompute[:,0]=m[:,0]
   for j,role in enumerate(['all','own','matrix'],1):
    positions=[]
    for a,b,lab in it['cont']:
     assert it['t']+1<=a<=b<=len(it['ids'])
     if lab and (role=='all' or lab==it[role]):positions.extend(range(a-it['t']-1,b-it['t']-1))
    if positions:
     recompute[:,j]=np.ascontiguousarray(tok[:,positions]).mean(axis=1)
   assert np.array_equal(recompute,m,equal_nan=True),(model,i,'continuation')
   token_values+=tok.size
   anchor_data=Z.read(PREFIX+'models/'+model+'/anchors/'+f.name)
   with np.load(io.BytesIO(anchor_data),allow_pickle=False) as a:
    for k in ['metrics','pool_logprobs','target_pool_raw_logits','continuation_token_logprobs']:
     assert np.array_equal(z[k][[0,5]],a[k],equal_nan=True),(model,i,k)
   met.append(m.astype(float));pl.append(lp.astype(float));raw.append(rr.astype(float))
  checks=read(f.with_suffix('.json'));assert checks['anchor_sha256']==sha(anchor_data) and checks['reused_k']==[0,5] and checks['forward_calls']==3
  for k,v in checks['bridge'].items():
   if isinstance(v,dict):assert v['passed'];bridge_max[k]=max(bridge_max.get(k,0),v['max_abs_diff']);bridge_count+=1
   else:assert v is True
  assert len(checks['anchor_checks'])==5 and all(x['passed'] for x in checks['anchor_checks']);anchorchecks+=len(checks['anchor_checks'])
 met=np.array(met);pl=np.array(pl);raw=np.array(raw);poolN=len(design['pools']['MSA']);assert poolN==len(design['pools']['EGY'])
 sign=np.array([1 if it['matrix']=='EGY' else -1 for it in items])[:,None]
 ds=(logsumexp(pl[:,:,poolN:],axis=2)-logsumexp(pl[:,:,:poolN],axis=2))*sign
 dr=(logsumexp(raw[:,:,poolN:],axis=2)-logsumexp(raw[:,:,:poolN],axis=2))*sign
 rawerr=float(abs(ds-dr).max());assert rawerr<1e-4
 vals={'own_logit':met[:,:,0],'flip_pp':100*(met[:,:,0]<0),'cont_lp_all':met[:,:,1],'cont_lp_own':met[:,:,2],'cont_lp_matrix':met[:,:,3],f'D_to_matrix_n{poolN}':ds}
 supplied=list(csv.DictReader((p/'ANALYSIS/SELECTED_CURVE.csv').open()));lookup={(r['group'],r['metric'],int(r['k'])):r for r in supplied};assert len(lookup)==len(supplied)
 senspath=p/'ANALYSIS/AMBIGUOUS_ALIAS_EXCLUSION.csv';sup_s=list(csv.DictReader(senspath.open())) if senspath.exists() else [];senslookup={(r['group'],r['metric'],int(r['k'])):r for r in sup_s}
 maxerror=0.;maxsens=0.;numeric_checks=0;groups=[];modelrows=[]
 for gi,g in enumerate(GROUPS):
  ix=np.array([i for i,it in enumerate(items) if it['group']==g]);keys=[tuple(items[i]['canonical_sid_candidates']) for i in ix];unique=list(dict.fromkeys(keys));cluster=np.array([unique.index(k) for k in keys]);C=len(unique)
  draws=np.random.default_rng(20261024+gi).integers(0,C,(10000,C));counts=np.zeros((10000,C),np.int16);np.add.at(counts,(np.arange(10000)[:,None],draws),1);weights=counts[:,cluster].astype(float)
  groups.append({'group':g,'items':len(ix),'clusters':C,'ambiguous':sum(len(k)!=1 for k in keys)})
  keep=np.array([len(k)==1 for k in keys])
  if not keep.all():
   skeys=[k for k in keys if len(k)==1];su=list(dict.fromkeys(skeys));sc=np.array([su.index(k) for k in skeys]);sd=np.random.default_rng(20261024+gi).integers(0,len(su),(10000,len(su)));sw=np.zeros((10000,len(su)),np.int16);np.add.at(sw,(np.arange(10000)[:,None],sd),1);sw=sw[:,sc].astype(float)
  for name,v in vals.items():
   x=v[ix]-v[ix,:1];valid=np.isfinite(x);assert np.all(valid==valid[:,:1])
   if not valid.any():continue
   den=weights@valid[:,0].astype(float);boot=(weights@np.nan_to_num(x,nan=0.))/np.where(den>0,den,np.nan)[:,None];ci=np.nanpercentile(boot,[2.5,97.5],axis=0);point=np.nanmean(x,axis=0)
   for k in range(21):
    r=lookup[(g,name,k)];actualvals={'delta_vs_baseline':point[k],'ci_low':ci[0,k],'ci_high':ci[1,k],'baseline_mean':np.nanmean(v[ix,0]),'condition_mean':np.nanmean(v[ix,k])}
    for key,value in actualvals.items():maxerror=max(maxerror,abs(value-float(r[key])));numeric_checks+=1
    assert int(r['n_items'])==valid[:,0].sum() and int(r['clusters'])==C and int(r['empty_bootstraps'])==(den==0).sum()
    row={'model':model,'group':g,'metric':name,'k':k,'n_items':int(valid[:,0].sum()),**{key:float(vv) for key,vv in actualvals.items()}};allrows.append(row);modelrows.append(row)
   if name in ['own_logit','flip_pp']:
    shapes.append({'model':model,'group':g,'metric':name,'adjacent_increases':int((np.diff(point)>1e-12).sum()),'adjacent_decreases':int((np.diff(point)<-1e-12).sum()),'delta_k5':float(point[5]),'delta_k20':float(point[20]),'k20_minus_k5_descriptive':float(point[20]-point[5]),'no_posthoc_optimal_k_selected':True})
   if not keep.all():
    xx=x[keep];vv=np.isfinite(xx[:,0]);dd=sw@vv.astype(float);bb=(sw@np.nan_to_num(xx,nan=0.))/np.where(dd>0,dd,np.nan)[:,None];cc=np.nanpercentile(bb,[2.5,97.5],axis=0);pp=np.nanmean(xx,axis=0)
    for k in range(21):
     r=senslookup[(g,name,k)]
     for key,value in [('estimate',pp[k]),('ci_low',cc[0,k]),('ci_high',cc[1,k])]:maxsens=max(maxsens,abs(value-float(r[key])))
     sensitivity.append({'model':model,'group':g,'metric':name,'k':k,'retained_items':int(vv.sum()),'estimate':float(pp[k]),'ci_low':float(cc[0,k]),'ci_high':float(cc[1,k])})
 assert maxerror<1e-10 and maxsens<1e-10
 assert len(modelrows)==len(supplied)
 sourceafter=read(p/'SOURCE_AFTER.json');packageafter=read(p/'PACKAGE_AFTER.json');assert all(sourceafter.values()) and all(packageafter.values())
 result[model]={'items':N,'scientific_condition_rows':N*21,'forwards_local_log':session['forward_calls'],'seconds_local_log':session['seconds'],'tokens_all_saved':token_values,'exact_reused_anchor_rows':2*N,'executed_files_equal_to_issued':codes,'primary_secondary_curve_rows':len(modelrows),'numeric_scalar_checks':numeric_checks,'independent_curve_maxerror':maxerror,'alias_sensitivity_maxerror':maxsens,'D_raw_normalized_maxerror':rawerr,'local_bridge_report_maxdiff':bridge_max,'local_anchor_checks_reported':anchorchecks,'local_bridge_raw_repeats_saved':False,'source_unchanged_reported':len(sourceafter),'package_unchanged_reported':len(packageafter),'groups':groups}
 print(model,json.dumps(result[model]),flush=True)
 csvsave(OUT/(model+'_SELECTED_CURVE.csv'),modelrows)
save(OUT/'INDEPENDENT_AUDIT.json',{'issued_zip_sha256':sha(ISSUED.read_bytes()),'received_payload_files':len(actual),'manifest_hashes_verified':len(manifest),'models':result,'new_model_forwards_here':0,'limits':['Current raw baseline/top5 bridge repetitions and padding are not retained; bridge checks are local reports, while scientific reused anchors were independently compared exactly to the issued package.','Mac source immutability is reported by local checks; remote files not accessed.','Pointwise intervals, no simultaneous coverage.','No new random controls at every k; no causal completeness or optimal-k claim.','Historical discrepancies unresolved.']})
csvsave(OUT/'ALL_MODELS_SELECTED_CURVE.csv',allrows);csvsave(OUT/'ALIAS_SENSITIVITY.csv',sensitivity);csvsave(OUT/'CURVE_SHAPE_DESCRIPTIVE.csv',shapes)
print('COMPLETE',len(allrows),len(sensitivity))
