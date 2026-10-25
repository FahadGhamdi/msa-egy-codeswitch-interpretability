from common import *
import ast,collections,copy,importlib.metadata,platform,traceback
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold

def main():
 log('prepare_start',code_sha256=sha(Path(__file__)))
 checks=[]
 for e in EXPECTED['source_files']:
  p=ROOT/e['path_in_New_Paper'];checks.append(dict(e,match=p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']))
 save('manifest/expected_checks.json',checks)
 if not all(x['match'] for x in checks):raise RuntimeError('Pinned input identity mismatch before inference')
 sources=[ROOT/e['path_in_New_Paper'] for e in EXPECTED['source_files']]
 sources += [p for p in (ROOT/'results').rglob('*') if p.is_file() and not p.is_symlink()]
 sources += [p for p in SNAP.rglob('*') if p.is_file()]
 sources += [TASK/n for n in ['MODEL_BASELINES.json','ANALYSIS_PLAN.txt','NEXT_LOCAL_TASK_AR.txt','BASE_QWEN_TASK_AR.txt','QWEN_REFERENCE_CODE_HASHES.json']]+[TASK/SLUG/'EXPECTED_INPUTS.json']
 sources += [ROOT/'Electronics_Submission_Provenance'/n for n in ['PROTOCOL_AR.txt','SUBMISSION_BASELINE.json','RECORDED_ENVIRONMENT.txt']]
 sources += [ROOT/'Electronics_Head_Stability_Expansion'/n for n in ['READ_FIRST_AR.txt','QWEN_HEAD_STABILITY_REPORT_AR.txt','revision/STATUS.json']]
 sources += [ROOT/'ELECTRONICS_QWEN_HEAD_STABILITY_20261021T215744Z/code'/n for n in json.loads((TASK/'QWEN_REFERENCE_CODE_HASHES.json').read_text())]
 save('manifest/inputs_before.json',inventory(sources));log('inputs_verified',files=len(set(sources)))
 cfg=json.loads((SNAP/'config.json').read_text());assert cfg['num_hidden_layers']==NL and cfg['num_attention_heads']==NH and cfg['hidden_size']==DIM and R==round(.8*NL)
 head_dim=cfg.get('head_dim') or DIM//NH
 save('manifest/config_gate.json',{'hidden_size':DIM,'num_attention_heads':NH,'num_hidden_layers':NL,'readout_layer':R,'head_dim':head_dim,'explicit_head_dim':cfg.get('head_dim'),'o_proj_expected_input_features':NH*head_dim})
 index=json.loads((SNAP/'model.safetensors.index.json').read_text());assert all((SNAP/p).is_file() for p in set(index['weight_map'].values()))
 baseline=json.loads((ROOT/'Electronics_Submission_Provenance/SUBMISSION_BASELINE.json').read_text());bm=next(m for m in baseline['models'] if m['model_id']==EXPECTED['model']);assert bm['recorded_revision']==EXPECTED['revision']
 save('manifest/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'libraries':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','transformers','tokenizers','scikit-learn','safetensors','pyarrow']},'snapshot':str(SNAP),'config':cfg,'local_files_only':True,'trust_remote_code':False,'mps_fallback':'1'})
 source=(ROOT/'exp02_causal_patching.py').read_text();names={'load_lince','type_classes','mono_sentences','prefix_pool','word_ids','with_bos','get_layers','variety_directions','make_scorer','run_scores','capture_clean','patched_scores'}
 nodes=[copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));helper='from pathlib import Path\nfrom collections import Counter,defaultdict\nimport numpy as np\nimport pandas as pd\nimport torch\nLINCE_DIR=Path('+repr(str(ROOT/'data/raw/lince_msaea/lid_msaea'))+')\n'+ast.unparse(module)+'\n'
 (WORK/'code/original_helpers.py').write_text(helper);save('manifest/helper_origins.json',[{'name':n.name,'source':'exp02_causal_patching.py','sha256':sha(ROOT/'exp02_causal_patching.py'),'lineno':n.lineno,'end_lineno':n.end_lineno,'text':ast.get_source_segment(source,n)} for n in nodes])
 import original_helpers as h
 rows=[];sidrows=[]
 for split in ['train','validation']:
  df=pd.read_parquet(ROOT/'data/raw/lince_msaea/lid_msaea'/f'{split}.parquet')
  for i,(ws,ls) in enumerate(zip(df.words,df.lid)):
   rows.append((list(ws),list(ls)));sidrows.append((f'{split}-{i}',list(ws),list(ls)))
 assert rows==h.load_lince()
 tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False)
 types=dict(zip(['MSA','EGY'],h.type_classes(rows)));summary=json.loads((EXPDIR/'summary.json').read_text())
 assert {v:len(ts) for v,ts in types.items()}==summary['n_unambiguous_types']
 mono={};pools={};aliases=collections.defaultdict(set)
 for v,lab,other in [('MSA','lang1','lang2'),('EGY','lang2','lang1')]:
  mono[v]=[(sid,ws,ls) for sid,ws,ls in sidrows if lab in ls and other not in ls];assert [(ws,ls) for sid,ws,ls in mono[v]]==h.mono_sentences(rows,v)
  pool=[]
  for si,(sid,ws,ls) in enumerate(mono[v]):
   keep=[i for i,w in enumerate(ws) if not w.startswith(('http','@'))];clean=[ws[i] for i in keep]
   for j in range(1,len(clean)+1):
    ids=tok(' '.join(clean[:j]),add_special_tokens=False).input_ids
    if len(ids)==12:
     row={'variety':v,'pool_index':len(pool),'sid':sid,'split':sid.rsplit('-',1)[0],'source_index':int(sid.rsplit('-',1)[1]),'mono_index':si,'original_words':ws,'original_labels':ls,'text_before_filter':' '.join(ws),'text_after_filter':' '.join(clean),'prefix_text':' '.join(clean[:j]),'word_boundary_after_filtered_words':j,'original_word_indices':keep[:j],'token_ids':ids,'token_ids_sha256':digest(ids)};pool.append(row);aliases[digest(ids)].add(sid);break
    if len(ids)>12:break
  assert [x['token_ids'] for x in pool]==h.prefix_pool(tok,h.mono_sentences(rows,v),12);pools[v]=pool
 for pool in pools.values():
  for x in pool:x['alias_sids']=sorted(aliases[x['token_ids_sha256']])
 jsonl('selection/prefix_pool.jsonl',[x for v in ['MSA','EGY'] for x in pools[v]])
 rng=np.random.default_rng(13);rnglog=[{'event':'initial','state':rng.bit_generator.state}];pairs=[]
 for v in ['MSA','EGY']:
  words=rng.choice(types[v],min(150,len(types[v])),replace=len(types[v])<150);rnglog.append({'event':'word_selection','variety':v,'words':words.tolist(),'state_after':rng.bit_generator.state})
  for word in words:
   word=str(word);wi=h.word_ids(tok,word);ei=int(rng.integers(len(pools['EGY'])));mi=int(rng.integers(len(pools['MSA'])));ep=pools['EGY'][ei];mp=pools['MSA'][mi]
   pairs.append({'pair':len(pairs),'target':v,'word':word,'word_token_ids':wi,'EGY_pool_index':ei,'MSA_pool_index':mi,'clean':h.with_bos(tok,ep['token_ids']+wi),'corrupt':h.with_bos(tok,mp['token_ids']+wi),'alone':h.with_bos(tok,tok(word,add_special_tokens=False).input_ids),'prefix_sids':sorted(set(ep['alias_sids']+mp['alias_sids'])),'prefix_hashes':sorted({ep['token_ids_sha256'],mp['token_ids_sha256']})})
   rnglog.append({'event':'prefix_draws','pair':len(pairs)-1,'EGY_pool_index':ei,'MSA_pool_index':mi,'state_after':rng.bit_generator.state})
 jsonl('selection/pairs.jsonl',pairs);save('selection/pair_rng.json',rnglog)
 p1=pd.read_csv(EXPDIR/'P1_scores_long.csv');assert len(p1)==300*3*(NL+1) and not p1.duplicated(['pair','prefix','layer']).any()
 diffs=[]
 for p in pairs:
  d=p1[p1.pair==p['pair']]
  if set(d.target)!={p['target']} or set(d.word)!={p['word']} or set(d.prefix)!={'EGY','MSA','none'} or set(d.layer)!=set(range(NL+1)):diffs.append(p['pair'])
 assert not diffs and len(pairs)==300 and summary['n_pairs']=={'MSA':150,'EGY':150} and summary['n_layers']==NL and summary['readout_layer']==R
 save('manifest/selection_check.json',{'all_300_pair_target_word_match':True,'P1_rows':len(p1),'type_counts':{v:len(x) for v,x in types.items()},'prefix_pool_counts':{v:len(x) for v,x in pools.items()},'layer_count':NL,'identity_status':'conditional on exact source, data, tokenizer and reconstructed settings; not historical prefix IDs','forward_calls':0})
 # Independent original make-pairs AST is executed for an exact token-sequence and RNG check.
 main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main');loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='target_var' for x in n.target.elts))
 from types import SimpleNamespace
 ns=dict(vars(h),rng=np.random.default_rng(13),args=SimpleNamespace(n_pairs=150,reverse=False),msa_types=types['MSA'],egy_types=types['EGY'],pools={v:[x['token_ids'] for x in a] for v,a in pools.items()},tok=tok,pairs=[])
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loop)],type_ignores=[])),'<original-pair-selection>','exec'),ns)
 assert all(all(p[k]==q[k] for k in ['target','word','clean','corrupt','alone']) for p,q in zip(pairs,ns['pairs'])) and digest(rng.bit_generator.state)==digest(ns['rng'].bit_generator.state)
 # Freeze original direction sample and exact tokenization/measurement positions.
 drng=np.random.default_rng(13);directions=[];drnglog=[]
 for v,lab in [('MSA','lang1'),('EGY','lang2')]:
  pick=drng.choice(len(mono[v]),min(400,len(mono[v])),replace=False);drnglog.append({'variety':v,'indices':pick.tolist(),'state_after':drng.bit_generator.state})
  for rank,i in enumerate(pick):
   sid,ws,ls=mono[v][i];text='';spans=[]
   for word in ws:
    if text:text+=' '
    spans.append((len(text),len(text)+len(word)));text+=word
   enc=tok(text,return_offsets_mapping=True,truncation=True,max_length=128);offs=enc.pop('offset_mapping');c2w=np.full(len(text)+1,-1)
   for wi,(s,e) in enumerate(spans):c2w[s:e]=wi
   last={}
   for ti,(s,e) in enumerate(offs):
    if e>s and c2w[e-1]>=0:last[int(c2w[e-1])]=ti
   pos=[last[wi] for wi,l in enumerate(ls) if l==lab and wi in last]
   directions.append({'unit':len(directions),'variety':v,'rank':rank,'mono_index':int(i),'sid':sid,'words':ws,'labels':ls,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'encoded':dict(enc),'offsets':offs,'positions':pos,'n_words':len(pos),'token_ids_sha256':digest(enc['input_ids'])})
 jsonl('selection/direction_sample.jsonl',directions);save('selection/direction_rng.json',drnglog)
 # Connected components, including all duplicate prefix aliases, before splitting.
 msa=pairs[:150];parent=list(range(150))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 seen={}
 for p in msa:
  for key in [('sid',s) for s in p['prefix_sids']]+[('tokens',s) for s in p['prefix_hashes']]:
   if key in seen:union(p['pair'],seen[key])
   else:seen[key]=p['pair']
 comps=collections.defaultdict(list)
 for i in range(150):comps[find(i)].append(i)
 components=sorted(comps.values(),key=min);groups=np.empty(150,int)
 for c,ids in enumerate(components):groups[ids]=c
 save('selection/components.json',{'components':components,'sizes':list(map(len,components)),'group_by_pair':groups.tolist(),'rule':'any source sid including duplicate aliases OR identical prefix token IDs; connected transitive closure'})
 overlap=sorted(set(x['sid'] for x in directions)&set(s for p in msa for s in p['prefix_sids']))
 save('selection/overlap_audit.json',{'readout_prefix_source_overlap_sids':overlap,'readout_prefix_source_overlap_n':len(overlap),'readout_not_crossfit':True,'prefix_duplicate_sets':[{'token_ids_sha256':h,'sids':sorted(v)} for h,v in aliases.items() if len(v)>1],'MSA_target_unique_types':len(set(p['word'] for p in msa)),'author_thread_independence':'unknown; metadata unavailable'})
 if len(components)<5:
  save('STATUS.json',{'status':'STOPPED','substatus':'STOPPED_SPLIT','reason':'Fewer than five source-connected components','n_components':len(components),'forward_calls':0});return
 folds=[]
 for k,(tr,te) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=13).split(np.zeros((150,1)),groups=groups)):
  def identities(ids):return {'pair_ids':ids.tolist(),'sids':sorted(set(s for i in ids for s in msa[i]['prefix_sids'])),'prefix_hashes':sorted(set(s for i in ids for s in msa[i]['prefix_hashes'])),'component_ids':sorted(set(int(groups[i]) for i in ids))}
  train,test=identities(tr),identities(te);assert not set(train['sids'])&set(test['sids']) and not set(train['prefix_hashes'])&set(test['prefix_hashes'])
  folds.append({'fold':k,'train':train,'test':test,'train_test_type_overlap':sorted(set(msa[i]['word'] for i in tr)&set(msa[i]['word'] for i in te))})
 save('selection/FOLDS.json',{'method':'GroupKFold(n_splits=5,shuffle=True,random_state=13)','sklearn_version':importlib.metadata.version('scikit-learn'),'folds':folds})
 if min(len(f['test']['pair_ids']) for f in folds)<10:
  save('STATUS.json',{'status':'STOPPED','substatus':'STOPPED_SPLIT','reason':'A fixed test fold has fewer than ten pairs','test_sizes':[len(f['test']['pair_ids']) for f in folds],'n_components':len(components),'forward_calls':0});return
 p2=pd.read_csv(EXPDIR/'P2_patching_long.csv');p2=p2[(p2.target=='MSA')&(p2.component=='attn')];assert len(p2)==150*R and not p2.duplicated(['pair','layer']).any()
 def rank(ids):return p2[p2.pair.isin(ids)].groupby('layer').delta.mean().reset_index().sort_values(['delta','layer'],ascending=[False,True]).to_dict('records')
 original=summary['top_attn_layers_by_patching'];assert original==EXPECTED['original_top_layers'];full=rank(range(150));assert [x['layer'] for x in full[:3]]==original
 selected=[];layers=set(original)
 for f in folds:
  ranking=rank(f['train']['pair_ids']);top=[x['layer'] for x in ranking[:3]];layers.update(top);selected.append({'fold':f['fold'],'ranking':ranking,'top_layers':top})
 save('selection/LAYER_SELECTION.json',{'folds':selected,'original_layers':original,'original_full_ranking':full,'union':sorted(layers),'limit':8,'tie_break':'delta descending, layer ascending'})
 if len(layers)>8:save('STATUS.json',{'status':'STOPPED','substatus':'STOPPED_BUDGET','reason':'Frozen candidate union exceeds eight layers','union':sorted(layers),'forward_calls':0});return
 save('STATUS.json',{'status':'PREPARED','n_pairs':300,'n_MSA_pairs':150,'n_components':len(components),'test_sizes':[len(f['test']['pair_ids']) for f in folds],'union':sorted(layers),'forward_calls':0})
 log('prepare_complete',status='PREPARED',union=sorted(layers),fold_sizes=[len(f['test']['pair_ids']) for f in folds])
if __name__=='__main__':
 try:main()
 except Exception as e:
  (WORK/'logs/prepare_failure.txt').write_text(traceback.format_exc());save('STATUS.json',{'status':'STOPPED','reason':str(e),'forward_calls':0});log('prepare_error',status='STOPPED',reason=str(e));raise
