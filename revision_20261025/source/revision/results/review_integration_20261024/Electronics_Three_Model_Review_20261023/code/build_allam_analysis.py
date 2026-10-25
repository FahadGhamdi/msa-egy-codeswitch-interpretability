"""Reuse the previously audited Qwen analysis estimands for ALLaM, with explicit pool geometry."""
from pathlib import Path
R=Path('/workspace/scratch/e3ad691229b6');O=R/'output/Electronics_Three_Model_Review_20261023';(O/'code').mkdir(exist_ok=True)
base=(R/'output/Electronics_Qwen_FP32_Stage1_Analysis_20261022/code/analyze_qwen_stage1.py').read_text()
helpers=base[base.index('MODES='):base.index('def main():')]
groups=base[base.index(' aliases='):base.index(' audit=')]
stats=base[base.index(' def summarise('):base.index(' # Descriptive historical aggregate')]
stats=stats.replace('D_to_matrix_n12','D_to_matrix_n50').replace('for n in [6,9]:','for n in [12,25]:').replace('range(12,12+n)','range(50,50+n)').replace('[12+j for j','[50+j for j').replace("subsets.append((12,'full',-1,list(range(12)),list(range(12,24))))","subsets.append((50,'full',-1,list(range(50)),list(range(50,100))))").replace('np.empty((624,S','np.empty((669,S').replace('for size in [6,9,12]:','for size in [12,25,50]:').replace('size==12','size==50').replace('[i-12]','[i-50]')
aliases=base[base.index(' sensitivity=[]'):base.index(" save('ANALYSIS_COMPLETE.json'")]
prefix='''"""Offline ALLaM analysis, same estimands as prior Qwen; no model inference."""
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
'''
start='''def main():
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
'''
history=''' # Paired comparison against saved historical target rows; no D identity claim.
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
'''
code=prefix+helpers+start+groups+" save('GROUPS_AND_ALIASES.json',{'groups':group_info,'aliases':aliases})\n"+stats+aliases+history
compile(code,'analyze_allam.py','exec');(O/'code/analyze_allam.py').write_text(code)
check=(R/'output/Electronics_Qwen_FP32_Stage1_Analysis_20261022/code/check_qwen_stage1_analysis.py').read_text()
check=check.replace('work/session03_received/ELECTRONICS_QWEN_FP32_STAGE1_20261022T113131Z','work/three_model_received_20261023/ALLAM/unpacked/ELECTRONICS_ALLAM_FP32_STAGE1_20261023_NIGHT').replace('output/Electronics_Qwen_FP32_Stage1_Analysis_20261022','output/Electronics_Three_Model_Review_20261023').replace('p[:,12:]','p[:,50:]').replace('p[:,:12]','p[:,:50]').replace('D_to_matrix_n12','D_to_matrix_n50')
(O/'code/check_allam_analysis.py').write_text(check)
