"""Independent offline review of night return; no model imports or inference."""
from pathlib import Path
import json, hashlib, zipfile, subprocess, sys, csv, gzip, collections, re
import numpy as np
R=Path('/workspace/scratch/e3ad691229b6');P=R/'work/three_model_received_20261023';O=R/'output/Electronics_Three_Model_Review_20261023';O.mkdir(exist_ok=True)
ISSUED=R/'output/Electronics_Three_Model_Night_20261023_FIX1'
A=P/'ALLAM/unpacked/ELECTRONICS_ALLAM_FP32_STAGE1_20261023_NIGHT'
read=lambda p:json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def maxdiff(a,b):
 v=np.abs(np.asarray(a,dtype=float)-b);return float(np.nanmax(v)) if np.isfinite(v).any() else 0.
def same(a,b):return np.isclose(a,b,atol=1e-4,rtol=1e-4,equal_nan=True).all()
archive=R/'upload/ELECTRONICS_THREE_MODEL_RUN_OUTPUT_20261023_REVIEW.zip'
assert sha(archive)=='d1af50d2d9925e63522438102486d07fdd5ff147c2b7b3836a5e0de368c5f52e'
report={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'model_forwards_here':0,'pilots':{},'outer_payloads':212,'historical_gate':'unresolved','raw_reference_units_available_here':False}
for slug in ['gemma-2-9b','Fanar-1-9B']:
 B=P/'THREE_MODEL'/slug;S=B/'PILOT';F=B/'FROZEN_TASK';prep=B/'PREPARED';man=read(S/'RETURN_CONTENTS.json')
 for e in man['core']:assert sha(S/e['path'])==e['sha256'] and (S/e['path']).stat().st_size==e['bytes']
 assert {e['path'] for e in man['core']}=={str(p.relative_to(S)) for p in S.rglob('*') if p.is_file()}-{'RETURN_CONTENTS.json'}
 with zipfile.ZipFile(B/'PILOT_REVIEW_CORE.zip') as z:
  assert z.testzip() is None
  for e in man['core']:assert z.read('PILOT/'+e['path'])==(S/e['path']).read_bytes()
 for p in (S/'executed_code').glob('*.py'):assert p.read_bytes()==(ISSUED/'NewModelPilot/code'/p.name).read_bytes()==(F/'code'/p.name).read_bytes()
 for rel,h in read(F/'TASK_HASHES.json').items():assert sha(F/rel)==h
 for rel,h in read(F/'EXPECTED_PREPARED.json').items():assert sha(prep/rel)==h
 for n in ['PLAN.json','FROZEN_DESIGN.json','TASK_HASHES.json']:assert (S/n).read_bytes()==(F/n).read_bytes()
 result=subprocess.run([sys.executable,'-B',str(ISSUED/'NewModelPilot/code/verify_return.py'),str(S)],capture_output=True,text=True);assert result.returncode==0,result.stderr
 checks=read(prep/'INPUT_CHECKS.json');assert all(x['match'] for x in checks)
 assert len(read(prep/'AGGREGATE_GATE.json'))==288 and all(x['match'] for x in read(prep/'AGGREGATE_GATE.json'))
 items=read(S/'ITEMS.json');design=read(S/'FROZEN_DESIGN.json');plan=read(S/'PLAN.json');records=read(S/'COMPARISONS.json');summary=read(S/'SUMMARY.json')
 assert len(items)==654 and summary['forward_calls']==640 and summary['reference_calls']==600 and summary['pilot_calls']==40 and summary['error'] is None
 assert all(read(S/'SOURCE_AFTER.json').values()) and set(read(S/'SOURCE_AFTER.json'))==set(read(S/'SOURCE_INPUTS.json'))
 rng=np.random.default_rng(20261022);assert design['permutations']==[{v:rng.permutation(50).tolist() for v in ['MSA','EGY']} for _ in range(200)]
 legacy=read(S/'LEGACY_CONDITIONS.json');assert legacy==read(ISSUED/'NewModelPilot/reference'/slug/'conditions.json')
 mapping={x['label']:x['uid'] for x in design['logical']};assert len(mapping)==304
 for label,u in mapping.items():
  if label=='baseline':continue
  mode,n=label.split('__');heads=sorted([list(map(int,re.fullmatch(r'L(\d+)H(\d+)',h).groups())) for h in legacy[n]['heads']]);assert design['unique'][u]['heads']==heads and design['unique'][u]['replacement']==mode
 assert design['unique_condition_count']==(301 if slug=='gemma-2-9b' else 292)
 poolspread=0.;first={};repeat=collections.Counter();maxmet=np.zeros(4);maxdelta=np.zeros(4);signchanges=0;tokens=0
 for c in plan['calls']:
  it=items[c['item']]
  with np.load(S/f"calls/call_{c['id']:02d}.npz",allow_pickle=False) as z:
   m=z['metrics'];lp=z['continuation_token_logprobs'];pl=z['pool_logprobs'];raw=z['target_pool_raw_logits'];tokens+=lp.size
   assert np.isfinite(m[:,0]).all() and np.isfinite(lp).all() and np.isfinite(pl).all() and np.isfinite(raw).all()
   for j,role in enumerate(['all','own','matrix'],1):
    ix=[k-it['t']-1 for a,b,lab in it['cont'] if lab and (role=='all' or lab==it[role]) for k in range(a,b)]
    if ix:assert np.array_equal(np.ascontiguousarray(lp[:,ix]).mean(1),m[:,j])
    else:assert np.isnan(m[:,j]).all()
   poolspread=max(poolspread,float(np.ptp(raw.astype(float)-pl,axis=1).max()))
   if c['mode']=='batch8_first':first[c['item']]={n:(m[k].copy(),pl[k].copy(),lp[k].copy()) for k,n in enumerate(c['conditions'])}
   elif c['mode'] in ['batch8_repeat','batch2']:
    for k,n in enumerate(c['conditions']):
     old=first[c['item']][n];assert same(m[k],old[0]) and same(pl[k],old[1]) and same(lp[k],old[2]);repeat[c['mode']]+=1
     if c['mode']=='batch8_repeat':assert np.array_equal(m[k],old[0],equal_nan=True) and np.array_equal(pl[k],old[1]) and np.array_equal(lp[k],old[2])
     else:maxmet=np.maximum(maxmet,np.nan_to_num(abs(m[k].astype(float)-old[0])))
    if c['mode']=='batch2':
     old=first[c['item']];d=m[1].astype(float)-m[0];prev=old['top5'][0].astype(float)-old['baseline'][0];maxdelta=np.maximum(maxdelta,np.nan_to_num(abs(d-prev)));valid=np.isfinite(d)&np.isfinite(prev);signchanges+=int(((d[valid]*prev[valid])<0).sum())
 assert repeat=={'batch8_repeat':96,'batch2':24} and signchanges==0
 hashes=[r for r in records if r['mode']=='batch8_repeat'];baseline={(r['item'],r['condition']):r for r in records if r['mode']=='batch8_first'}
 assert all(r['logits_sha256']==baseline[r['item'],r['condition']]['logits_sha256'] for r in hashes)
 with np.load(S/'reference_means_norms.npz',allow_pickle=False) as z:
  ratios=[]
  for n in [f'random5_d{i}' for i in range(100)]:
   for a,b in zip(legacy['top5']['heads'],legacy[n]['heads']):
    L,h=map(int,re.fullmatch(r'L(\d+)H(\d+)',a).groups());K,j=map(int,re.fullmatch(r'L(\d+)H(\d+)',b).groups());ratios.append(float(z[f'norm_{K}'][j]/z[f'norm_{L}'][h]));assert L==K
 assert ratios==[r['ratio'] for r in read(S/'LEGACY_RANDOM_FP32_NORMS.json')];assert all(.5<=r<=2 for r in ratios)
 local=read(B/'LOCAL_CPU_AUDIT.json');assert len(local['means_reconstruction'])==6 and all(x['bitwise_equal'] and x['max_abs_error']==0 for x in local['means_reconstruction'])
 report['pilots'][slug]={'payloads':len(man['core']),'items_prepared':654,'pilot_calls':40,'rows':248,'reference_forwards_local':600,'continuation_token_values_recomputed':tokens,'continuation_mean_max_error':0,'batch8_repeat_rows_bitwise_equal':96,'reported_full_logit_hash_matches':96,'full_logit_hashes_recomputed_here':False,'batch2_vs8_max_metric_diff':maxmet.tolist(),'delta_top5_vs_baseline_max_diff':maxdelta.tolist(),'delta_sign_changes':signchanges,'pool_normalizer_spread_max':poolspread,'norm_ratios_recomputed':500,'norm_violations':0,'norms_recomputed_from_weights_here':False,'means_reconstruction_local_report_only':local['means_reconstruction'],'means_reconstructed_here':False,'loader_check':read(S/'EMBEDDING_OUTPUT_AUDIT.json'),'forecast':read(S/'FORECAST.json'),'status':summary['status'],'seconds_local':summary['seconds']}
 print('PILOT VERIFIED',slug,flush=True)
# Reuse issued checker as secondary cross-check, plus independent loops below.
pilot=R/'work/allam_pilot_received/ELECTRONICS_ALLAM_FP32_PILOT_20261022_A'
r=subprocess.run([sys.executable,'-B',str(ISSUED/'ALLaM/Task/code/verify_return.py'),str(A),'--pilot',str(pilot)],capture_output=True,text=True);assert r.returncode==0,r.stderr
secondary=json.loads(r.stdout)
man=read(A/'RETURN_CONTENTS.json')['core'];assert {x['path'] for x in man}=={str(p.relative_to(A)) for p in A.rglob('*') if p.is_file()}-{'RETURN_CONTENTS.json'}
for p in (A/'executed_continuation_code').glob('*.py'):assert p.read_bytes()==(ISSUED/'ALLaM/Task/code'/p.name).read_bytes()
for n,src in [('CONTINUATION_PLAN.json','CONTINUATION_PLAN.json'),('CONTINUATION_TASK_HASHES.json','TASK_HASHES.json')]:assert (A/n).read_bytes()==(ISSUED/'ALLaM/Task'/src).read_bytes()
prior=R/'work/allam_stage1_received/ELECTRONICS_ALLAM_FP32_STAGE1_20261022_A';prev=read(prior/'STATE.json');state=read(A/'STATE.json');items=read(A/'ITEMS.json');done=state['completed_items'];assert len(done)==669 and set(done)==set(range(669)) and done[:310]==prev['completed_items']
for i in prev['completed_items']:
 for suffix in ['.npz','_checks.json']:assert (prior/f'items/item_{i:04d}{suffix}').read_bytes()==(A/f'items/item_{i:04d}{suffix}').read_bytes()
assert (A/'actions.jsonl').read_bytes().startswith((prior/'actions.jsonl').read_bytes())
for n in ['ITEMS.json','DESIGN.json','reference_means_norms.npz','HISTORICAL_TARGET_ROWS.csv.gz']:assert (A/n).read_bytes()==(prior/n).read_bytes()
poolspread=0.;tokens=0;baseline_max=0.;finite=0;missing=0;integration_max=0.
for i,it in enumerate(items):
 with np.load(A/f'items/item_{i:04d}.npz',allow_pickle=False) as z:
  m=z['metrics'];lp=z['continuation_token_logprobs'];tokens+=lp.size;finite+=int(np.isfinite(m).sum());missing+=int(np.isnan(m).sum());assert np.isfinite(m[:,0]).all()
  for j,role in enumerate(['all','own','matrix'],1):
   ix=[k-it['t']-1 for a,b,lab in it['cont'] if lab and (role=='all' or lab==it[role]) for k in range(a,b)]
   if ix:assert np.array_equal(np.ascontiguousarray(lp[:,ix]).mean(1),m[:,j])
   else:assert np.isnan(m[:,j]).all()
  poolspread=max(poolspread,float(np.ptp(z['target_pool_raw_logits'].astype(float)-z['pool_logprobs'],axis=1).max()))
  for a,b in [('baseline_repeat_metrics','metrics'),('baseline_repeat_pool_logprobs','pool_logprobs'),('baseline_repeat_token_logprobs','continuation_token_logprobs')]:baseline_max=max(baseline_max,maxdiff(z[a],z[b][0]))
for x in read(A/'INTEGRATION_CHECKS.json'):integration_max=max(integration_max,x['max_metric_diff'])
events=[json.loads(x) for x in (A/'actions.jsonl').read_text().splitlines()];assert [e['item'] for e in events]==done and all(e['driver_bytes']<=.8*e['recommended_bytes'] for e in events)
report['ALLaM']={**secondary,'payloads':len(man),'preserved_prior_item_files':620,'rows_recomputed':669*304,'continuation_token_values':tokens,'finite_metric_cells':finite,'expected_missing_metric_cells':missing,'baseline_repeat_max_difference':baseline_max,'integration_metric_max_difference':integration_max,'pool_normalizer_spread_max':poolspread,'group_counts':dict(collections.Counter(x['group'] for x in items)),'session2':read(A/'SESSION_2.json'),'source_unchanged_local_checks':len(read(A/'SOURCE_AFTER_SESSION_2.json')),'max_MPS_GiB':max(e['driver_bytes'] for e in events)/2**30}
save('INDEPENDENT_NIGHT_AUDIT.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
