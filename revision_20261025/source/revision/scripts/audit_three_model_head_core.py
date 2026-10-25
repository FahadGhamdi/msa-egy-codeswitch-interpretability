"""Independent, read-only audit of received three-model CORE head replay and subset selection."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import ast,collections,datetime,hashlib,itertools,json,zipfile
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
import sys
V=Path(__file__).resolve().parents[1];R=V.parents[1];slug=sys.argv[1]
A=next(p for p in (R/'local_evidence_received/three_model_head_core'/slug).iterdir() if p.is_dir())
Z=R/'upload'/f'ELECTRONICS_HEAD_STABILITY_{slug}_REVIEW_CORE.zip'
O=V/'results/three_model_head_core_20261022'/slug;O.mkdir(parents=True,exist_ok=True)
expected=json.loads((V/'protocols/three_model_head_stability'/slug/'EXPECTED_INPUTS.json').read_text())
NH=expected['num_attention_heads'];NL=expected['n_layers'];DIM=expected['hidden_size'];RL=expected['readout_layer'];LAYERS=sorted(expected['original_top_layers'])
J=lambda p:json.loads(p.read_text());jl=lambda p:[json.loads(s) for s in p.read_text().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();dig=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
transfer=J(R/'upload/TRANSFER_INDEX.json')['models'][slug]
pack=next(p for p in transfer['packages'] if p['name']==Z.name)
assert sha(Z)==pack['sha256'] and Z.stat().st_size==pack['bytes']
with zipfile.ZipFile(Z) as z:
 m=json.loads(z.read('RETURN_MANIFEST.json'));assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))
 original_map={e['path']:e for e in m['files']}
 received=0;deferred=[]
 for e in transfer['files']:
  if e['path']!='RETURN_MANIFEST.json':
   old=original_map[e['path']];assert old['sha256']==e['sha256'] and old['bytes']==e['bytes']
  if e['transfer_archive']==Z.name:
   data=z.read(e['transfer_member']);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];received+=1
   if e['path']!='RETURN_MANIFEST.json':assert sha(A.parent/e['path'])==e['sha256']
  else:
   assert '/directions/unit_' in e['path'] and e['path'].endswith('.npz');deferred.append(e['path'])
 assert len(deferred)==800 and len(transfer['files'])==len(m['files'])+1
 for e in pack['added_files']:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
 assert set(z.namelist())=={e['transfer_member'] for e in transfer['files'] if e['transfer_archive']==Z.name}|{e['path'] for e in pack['added_files']}
frozen=J(A/'manifest/PROTOCOL_FROZEN.json')
for rel,h in frozen['files'].items():assert sha(A/rel)==h
assert J(A/'manifest/EXPECTED_INPUTS.json')==expected
for name in ['ANALYSIS_PLAN.txt','NEXT_LOCAL_TASK_AR.txt']:assert (A/'manifest'/name).read_bytes()==(V/'protocols/three_model_head_stability'/name).read_bytes()
before=J(A/'manifest/inputs_before.json');assert before==J(A/'manifest/inputs_after.json')
origroot=R/'local_evidence_received/verified_payload';orig=origroot/'results/exp02'/slug;rawroot=R/'local_evidence_received/verified_outer/ELECTRONICS_REVISION_AUDIT_20261021T192824Z/candidate_inventory/source_and_dataset'
for e in J(A/'manifest/EXPECTED_INPUTS.json')['source_files']:
 rel=e['path_in_New_Paper'];p=(origroot if rel.startswith('results/') else rawroot)/rel;assert sha(p)==e['sha256']
 assert before['/Users/fahad/Documents/Research-Projects/New-Paper/'+rel]['sha256']==e['sha256']
source=ast.parse((rawroot/'exp02_causal_patching.py').read_text());helpers=ast.parse((A/'code/original_helpers.py').read_text())
for f in J(A/'manifest/helper_origins.json'):
 a=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==f['name']);b=next(n for n in helpers.body if isinstance(n,ast.FunctionDef) and n.name==f['name']);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
rows={}
for split in ['train','validation']:
 d=pd.read_parquet(rawroot/'data/raw/lince_msaea/lid_msaea'/f'{split}.parquet')
 for i,r in d.iterrows():rows[f'{split}-{i}']=(list(r.words),list(r.lid))
pools=jl(A/'selection/prefix_pool.jsonl');index={(p['variety'],p['pool_index']):p for p in pools};aliases=collections.defaultdict(set)
for p in pools:
 ws,ls=rows[p['sid']];assert ws==p['original_words'] and ls==p['original_labels'];assert ' '.join(ws)==p['text_before_filter']
 keep=[i for i,w in enumerate(ws) if not w.startswith(('http','@'))];assert ' '.join(ws[i] for i in keep)==p['text_after_filter'];assert p['original_word_indices']==keep[:p['word_boundary_after_filtered_words']]
 assert p['prefix_text']==' '.join(ws[i] for i in p['original_word_indices']);assert dig(p['token_ids'])==p['token_ids_sha256'] and len(p['token_ids'])==12
 aliases[p['token_ids_sha256']].add(p['sid'])
for p in pools:assert sorted(aliases[p['token_ids_sha256']])==p['alias_sids']
pairs=jl(A/'selection/pairs.jsonl');assert [p['pair'] for p in pairs]==list(range(300))
old1=pd.read_csv(orig/'P1_scores_long.csv')
for p in pairs:
 rr=old1[old1.pair==p['pair']];assert set(rr.word)=={p['word']} and set(rr.target)=={p['target']}
 es=[index[(v,p[v+'_pool_index'])] for v in ['EGY','MSA']];assert p['prefix_sids']==sorted(set(s for e in es for s in e['alias_sids']))
 assert p['prefix_hashes']==sorted(set(e['token_ids_sha256'] for e in es))
 # Gemma/Fanar prepend BOS2 under the pinned original helper; ALLaM does not.
 bos=[] if slug.startswith('ALLaM') else [2]
 assert p['clean']==bos+es[0]['token_ids']+p['word_token_ids'] and p['corrupt']==bos+es[1]['token_ids']+p['word_token_ids']
# Recompute exact word sampling and prefix-index RNG without invoking tokenizer/model.
f=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='type_classes');ns={'Counter':collections.Counter,'defaultdict':collections.defaultdict};exec(compile(ast.Module(body=[f],type_ignores=[]),'<pure-type-classes>','exec'),ns)
types=ns['type_classes'](list(rows.values()));rng=np.random.default_rng(13);pos=0
for v,ts in zip(['MSA','EGY'],types):
 for w in rng.choice(ts,min(150,len(ts)),replace=len(ts)<150):
  p=pairs[pos];assert p['word']==w and p['target']==v
  for pv in ['EGY','MSA']:assert int(rng.integers(sum(q['variety']==pv for q in pools)))==p[pv+'_pool_index']
  pos+=1
msa=pairs[:150];parent=list(range(150))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
seen={}
for p in msa:
 for key in [('s',s) for s in p['prefix_sids']]+[('h',h) for h in p['prefix_hashes']]:
  if key in seen:
   a,b=find(p['pair']),find(seen[key]);parent[max(a,b)]=min(a,b)
  seen[key]=p['pair']
comps=collections.defaultdict(list)
for i in range(150):comps[find(i)].append(i)
components=sorted(comps.values(),key=min);groups=np.empty(150,int)
for k,ids in enumerate(components):groups[ids]=k
assert components==J(A/'selection/components.json')['components'];folds=J(A/'selection/FOLDS.json')['folds']
for k,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=13).split(np.zeros((150,1)),groups=groups)):
 f=folds[k];assert tr.tolist()==f['train']['pair_ids'] and te.tolist()==f['test']['pair_ids']
 for field in ['prefix_sids','prefix_hashes']:
  assert not set(s for i in tr for s in msa[i][field])&set(s for i in te for s in msa[i][field])
 assert not set(msa[i]['word'] for i in tr)&set(msa[i]['word'] for i in te)
old2=pd.read_csv(orig/'P2_patching_long.csv');old2=old2[(old2.target=='MSA')&(old2.component=='attn')]
new1=pd.read_csv(A/'results/P1_replayed.csv');new2=pd.read_csv(A/'results/P2_attention_replayed.csv');maxima={}
for label,old,new,keys,metrics in [('P1',old1[old1.target=='MSA'],new1,['pair','prefix','layer'],['s']),('P2',old2,new2,['pair','layer'],['s_clean','s_corrupt','s_patched','delta','gap'])]:
 q=old.merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(q)==len(old)==len(new)
 mx=0.
 for c in metrics:
  delta=np.abs(q[c+'_new']-q[c+'_old']);assert np.isfinite(delta).all() and (delta<=1e-4+1e-4*np.abs(q[c+'_old'])).all();mx=max(mx,float(delta.max()))
 maxima[label]=mx
raw=pd.read_csv(A/'results/HEAD_DELTAS.csv');assert len(raw)==150*len(LAYERS)*NH and not raw.duplicated(['pair','layer','head']).any()
assert set(map(tuple,raw[['pair','layer','head']].values))==set(itertools.product(range(150),LAYERS,range(NH)))
assert np.isfinite(raw[['s_corrupt','s_patched','delta']]).all().all() and np.max(np.abs(raw.delta-(raw.s_patched-raw.s_corrupt)))<1e-12
agg=raw.groupby(['layer','head']).delta.agg(['mean','std','count']).reset_index();old3=pd.read_csv(orig/'P3_heads.csv');q=old3.merge(agg,on=['layer','head'],suffixes=('_old','_new'),validate='one_to_one');assert len(q)==3*NH and (q.count_old==q.count_new).all()
maxima['P3']=max(float(np.abs(q[c+'_new']-q[c+'_old']).max()) for c in ['mean','std']);assert maxima['P3']<1e-4
original_order=old3.sort_values(['mean','layer','head'],ascending=[False,True,True])[['layer','head']].values.tolist();assert original_order==agg.sort_values(['mean','layer','head'],ascending=[False,True,True])[['layer','head']].values.tolist()
reported=J(A/'results/STABILITY_ANALYSIS.json');summaries=[];selected_sets=[];train_raw=[];test_raw=[]
for f in folds:
 k=f['fold'];tr=f['train']['pair_ids'];te=f['test']['pair_ids'];layer_ranking=new2[new2.pair.isin(tr)].groupby('layer').delta.mean().reset_index().sort_values(['delta','layer'],ascending=[False,True]);ls=layer_ranking.layer.iloc[:3].tolist()
 assert ls==J(A/'selection/LAYER_SELECTION.json')['folds'][k]['top_layers']==reported['folds'][k]['top3_layers']
 rankings=raw[raw.pair.isin(tr)&raw.layer.isin(ls)].groupby(['layer','head']).delta.mean().reset_index().sort_values(['delta','layer','head'],ascending=[False,True,True]);chosen=rankings.iloc[:5];keys=[(int(r.layer),int(r['head'])) for _,r in chosen.iterrows()]
 assert keys==[(r['layer'],r['head']) for r in reported['folds'][k]['top5_heads']];selected_sets.append(set(keys));effects=[]
 for rank,((L,H),(_,r)) in enumerate(zip(keys,chosen.iterrows()),1):
  d=raw[(raw.pair.isin(te))&(raw.layer==L)&(raw['head']==H)].delta;rr=reported['folds'][k]['top5_heads'][rank-1]
  for a,b in [(d.mean(),rr['test_mean_delta']),(d.std(ddof=1),rr['test_sd']),(r.delta,rr['train_mean_delta'])]:assert abs(a-b)<1e-13
  assert len(d)==rr['test_n'];effects.extend(d.tolist());test_raw.append({'fold':k,'rank':rank,'layer':L,'head':H,'train_mean_delta':r.delta,'test_n':len(d),'test_mean_delta':d.mean(),'test_sd':d.std(ddof=1)})
 assert abs(np.mean(effects)-reported['folds'][k]['test_mean_of_individual_selected_head_deltas'])<1e-13
 summaries.append({'fold':k,'train_pairs':len(tr),'test_pairs':len(te),'top_layers':ls,'top_heads':keys,'overlap_original_top5':len(set(keys)&set(map(tuple,original_order[:5]))),'test_mean_individual_head_delta':float(np.mean(effects))})
frequency=pd.read_csv(A/'results/HEAD_SELECTION_FREQUENCIES.csv')
for r in frequency.itertuples():
 count=sum((r.layer,r.head) in x for x in selected_sets);eligible=sum(r.layer in x['top_layers'] for x in summaries)
 assert r.selected_folds==count and r.eligible_folds==eligible and r.unconditional_frequency==count/5
 assert eligible and abs(r.conditional_frequency_given_eligible-count/eligible)<1e-14
assert sorted(set(L for f in summaries for L in f['top_layers'])|set(expected['original_top_layers']))==LAYERS==J(A/'selection/LAYER_SELECTION.json')['union']
# Verify all per-pair head records agree with the assembled raw CSV.
records=[]
for path in sorted((A/'results').glob('heads_pair_*_layer_*.json')):
 rec=J(path)
 records.extend(rec['records'])
detail=pd.DataFrame(records).sort_values(['pair','layer','head']).reset_index(drop=True)
flat=raw.sort_values(['pair','layer','head']).reset_index(drop=True)
assert len(detail)==len(flat)
for col in ['pair','layer','head','s_corrupt','s_patched','delta']:assert np.allclose(detail[col],flat[col],atol=1e-13,rtol=0)

jaccards=[len(a&b)/len(a|b) for a,b in itertools.combinations(selected_sets,2)];assert jaccards==[x['jaccard'] for x in reported['pairwise_fold_jaccards']]
samples=jl(A/'selection/direction_sample.jsonl');counts={};mu_errors={};direction_ids=set();direction_rng=np.random.default_rng(13)
for var,lab,other in [('MSA','lang1','lang2'),('EGY','lang2','lang1')]:
 mono=[sid for sid,(ws,ls) in rows.items() if lab in ls and other not in ls]
 expected_ids=[mono[i] for i in direction_rng.choice(len(mono),400,replace=False)]
 assert expected_ids==[x['sid'] for x in samples if x['variety']==var]
 acc=None;n=0
 for s in [x for x in samples if x['variety']==var]:
  ws,ls=rows[s['sid']];assert ws==s['words'] and ls==s['labels'] and other not in ls and lab in ls;direction_ids.add(s['sid'])
  assert s['text']==' '.join(ws) and hashlib.sha256(s['text'].encode()).hexdigest()==s['text_sha256'] and dig(s['encoded']['input_ids'])==s['token_ids_sha256']
  c2w=np.full(len(s['text'])+1,-1);offset=0
  for wi,word in enumerate(ws):c2w[offset:offset+len(word)]=wi;offset+=len(word)+1
  last={}
  for ti,(start,end) in enumerate(s['offsets']):
   if end>start and c2w[end-1]>=0:last[int(c2w[end-1])]=ti
  positions=[last[wi] for wi,label in enumerate(ls) if label==lab and wi in last]
  assert positions==s['positions'] and len(positions)==s['n_words']
  n+=s['n_words']
 stored=np.load(A/f'directions/mu_{var}.npy',allow_pickle=False)
 assert stored.shape==(NL+1,DIM) and stored.dtype==np.float32 and np.isfinite(stored).all()
 counts[var]=n;mu_errors[var]=None

overlap=sorted(direction_ids&set(s for p in msa for s in p['prefix_sids']));assert overlap==J(A/'selection/overlap_audit.json')['readout_prefix_source_overlap_sids']
runtime=J(A/'manifest/model_runtime.json');assert runtime['device']=='mps' and runtime['dtype']=='bfloat16' and runtime['eval'] and runtime['readout_layer']==RL and runtime['num_heads']==NH and runtime['num_layers']==NL
cfg=J(A/'manifest/config_gate.json');env=J(A/'manifest/environment.json')
assert expected['revision'] in env['snapshot']
assert cfg['head_dim']==runtime['head_dim'] and cfg['num_attention_heads']==NH and cfg['hidden_size']==DIM
if not slug.startswith('ALLaM'):assert cfg['head_dim']==256
info=J(A/'manifest/model_loading_info.json');assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
events=jl(A/'logs/actions.jsonl');load_event=next(e for e in events if e['action']=='model_loading');assert frozen['utc']<load_event['utc']
state=J(A/'STATUS.json');assert state['status']=='COMPLETE' and state['forward_calls']==2150
pd.DataFrame(test_raw).to_csv(O/'heldout_head_effects_verified.csv',index=False);pd.DataFrame(summaries).to_csv(O/'fold_summary_verified.csv',index=False)
report={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'CORE_AUDIT_PASS_DIRECTIONS_PENDING','model':slug,'archive':Z.name,'archive_sha256':sha(Z),'received_original_members_verified':received,'deferred_direction_contributions':len(deferred),'original_inventory_unchanged_files':len(before),'original_helpers_AST_exact':True,'issued_protocol_match':True,'folds_regenerated_from_raw_source_components':True,'components':len(components),'fold_test_sizes':[x['test_pairs'] for x in summaries],'folds':summaries,'all_original_head_ranks_reproduced':True,'original_head_count':3*NH,'raw_head_effects':len(raw),'original_max_errors':maxima,'pairwise_jaccards':jaccards,'minimum_of_25_heldout_head_mean_deltas':min(x['test_mean_delta'] for x in test_raw),'direction_word_counts':counts,'mean_vector_CPU_reconstruction_max_errors':mu_errors,'readout_prefix_source_overlap_tweets':len(overlap),'observed_forward_calls':state['forward_calls'],'scope':['Received core evidence audited independently, not full archive verification.','800 per-tweet direction contributions not received; independent mean reconstruction pending.','Internal selection stability with fixed readout and overlapping direction-source tweets; not independent corpus confirmation.','No cloud tokenizer/model rerun. Conditional historical linkage only.','Single-head patch effects do not establish joint ablation, mediation, next-token preferences or significance.']}
(O/'AUDIT_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['folds','scope']},indent=2))
