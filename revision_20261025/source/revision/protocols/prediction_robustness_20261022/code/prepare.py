"""Read-only reconstruction of exp06 selection. No model loading or forward calls.
Run with the existing project .venv; safe-loads only the hash-verified saved probe.
"""
import argparse, ast, collections, copy, hashlib, json, os, sys
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
TASK=Path(__file__).resolve().parents[1]
GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_helpers(root,np,pd,torch):
 env=dict(np=np,pd=pd,torch=torch,Counter=collections.Counter,defaultdict=collections.defaultdict,
          LANG={'lang1':'MSA','lang2':'EGY'},LINCE_DIR=root/'data/raw/lince_msaea/lid_msaea')
 names={'exp02_causal_patching.py':['mono_sentences','word_ids','with_bos','get_layers'],
        'exp03_natural_vs_controlled.py':['load_rows_with_ids'],
        'readout.py':['type_classes_mono','word_positions','probe_split','head_geometry','collect_head_inputs','head_output_norms','AblationHooks']}
 for file,want in names.items():
  tree=ast.parse((root/file).read_text());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in want]
  assert {n.name for n in nodes}==set(want),(file,'missing helper')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/file),'exec'),env)
 return env

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--slug',default='Qwen3-8B-Base');a=ap.parse_args()
 root=a.project.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 (out/'logs').mkdir();(out/'tmp').mkdir();os.environ['TMPDIR']=str(out/'tmp')
 spec=json.loads((TASK/'EXPECTED_INPUTS.json').read_text());m=next(x for x in spec['models'] if x['slug']==a.slug)
 checks=[]
 try:
  for e in spec['common_inputs']+m['snapshot_files']+m['reference_files']+[{'path':m['probe_path'],'sha256':m['probe_sha256']}]:
   rel=e.get('path',e.get('relative_to_New_Paper'));p=root/rel
   actual=sha(p) if p.is_file() else None;checks.append({'path':rel,'expected':e['sha256'],'actual':actual,'match':actual==e['sha256']})
   save(out/'INPUT_CHECKS.json',checks)
   if actual!=e['sha256']:raise RuntimeError('Input hash mismatch/missing: '+rel)
  import numpy as np,pandas as pd,torch
  from transformers import AutoTokenizer
  allowed=[np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType,np.dtypes.Int32DType]
  with torch.serialization.safe_globals(allowed):probe=torch.load(root/m['probe_path'],map_location='cpu',weights_only=True)
  assert int(probe['layer'])==m['layer'] if 'layer' in probe else True
  assert probe['stats']['n_train']=={'MSA':4811,'EGY':4800}
  assert torch.isfinite(probe['w']).all() and torch.isfinite(torch.as_tensor(probe['b'])).all()
  snap=(root/m['snapshot_files'][0]['path']).parent
  tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
  env=load_helpers(root,np,pd,torch)
  from types import SimpleNamespace
  args=SimpleNamespace(purity_source='mono',seed=13,max_K=48,n_next=3,min_next=1,max_items=300,mono_items=150,n_pref=50)
  env.update(args=args,tok=tok,probe=probe,GROUPS=GROUPS,rng=np.random.default_rng(13),device='cpu')
  tree=ast.parse((root/'exp06_behavioral.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  # Execute exactly the submitted item construction, but replace the prohibited
  # get_probe call with the already hash-checked safe-loaded probe. No training.
  start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets))
  end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='g' and isinstance(n.iter,ast.Name) and n.iter.id=='items')
  nodes=copy.deepcopy(main.body[start:end+1])
  nodes=[n for n in nodes if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='probe' for t in n.targets))]
  # The historical D block is not executed: its set iteration breaks ties by
  # process hash order. Deterministic lists are separately labelled NEW below.
  first_idx=next(i for i,n in enumerate(nodes) if isinstance(n,ast.ImportFrom) and n.module=='collections')
  build_idx=next(i for i,n in enumerate(nodes) if isinstance(n,ast.FunctionDef) and n.name=='build')
  nodes=nodes[:first_idx]+nodes[build_idx:]
  # Add identity only; does not change eligibility, input, order, or RNG.
  for n in ast.walk(ast.Module(body=nodes,type_ignores=[])):
   if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='cont' for k in n.value.keys):
    n.value.keys.append(ast.Constant('word_index'));n.value.values.append(ast.Name(id='i',ctx=ast.Load()))
  code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));(out/'selection_executed.py').write_text(ast.unparse(code)+'\n');exec(compile(code,'selection_executed.py','exec'),env)
  items=[x for g in GROUPS for x in env['items'][g]];counts={g:len(env['items'][g]) for g in GROUPS};assert counts==m['expected_items'],counts
  for i,it in enumerate(items):it['item']=i
  # Identify monolingual aliases, retaining every possible source if duplicate.
  lookup=collections.defaultdict(list)
  for sid,ws,ls in env['rows']:lookup[(tuple(ws),tuple(ls))].append(sid)
  for it in items:
   if it['group'].startswith('mono_'):
    v=it['own'];idx=int(it['sid'].rsplit('-',1)[1]);ws,ls=env['split'][v][1][idx];it['canonical_sid_candidates']=lookup[(tuple(ws),tuple(ls))]
   else:it['canonical_sid_candidates']=[it['sid']]
  path=root/f'results/exp06/{a.slug}/B_long.csv'
  if not path.is_file():raise RuntimeError('Missing B_long.csv: cannot authorize inference without item-level replay target')
  old=pd.read_csv(path);base=old[old.condition=='baseline'].sort_values('item')
  assert len(base)==len(items) and base.item.is_unique
  for it,row in zip(items,base.to_dict('records')):
   for k in ('item','sid','group','word'):assert it[k]==row[k],(it['item'],k)
  assert not old.duplicated(['item','condition']).any()
  summ=pd.read_csv(root/f'results/exp06/{a.slug}/B_summary.csv');aggregate=[]
  old['cls']=old.condition.str.replace(r'_d\d+$','',regex=True)
  for row in summ.to_dict('records'):
   d=old[(old.group==row['group'])&(old.cls==row['condition'])]
   for key,actual in [('n_items',d.item.nunique()),('flip_rate',(d.own_logit<0).mean()),('D_to_matrix',d.D_to_matrix.mean())]+[(k,d.groupby('item')[k].mean().mean()) for k in ['d_own_logit','d_D_to_matrix','d_cont_lp_matrix','d_cont_lp_own','d_cont_lp_all']]:
    ok=bool(np.isclose(actual,row[key],atol=1e-10,rtol=1e-10,equal_nan=True));aggregate.append({'group':row['group'],'condition':row['condition'],'metric':key,'match':ok});assert ok,(row['group'],row['condition'],key,actual,row[key])
  wfreq=collections.Counter(w for ws,_ in env['plain'] for w in ws);first={v:collections.Counter() for v in ['MSA','EGY']}
  for v in first:
   for word in sorted(env['types'][v]):
    ids=env['word_ids'](tok,word)
    if ids and tok.decode([ids[0]]).strip():first[v][ids[0]]+=wfreq[word]
  shared=set(first['MSA'])&set(first['EGY']);ranked={v:sorted((i for i in first[v] if i not in shared),key=lambda i:(-first[v][i],i)) for v in first}
  n=min(50,*[len(x) for x in ranked.values()]);assert n==m['expected_n_pref']
  lists={'status':'NEW deterministic lists; historical identity not established','order':'descending aggregate frequency, ascending token ID for ties','balanced_n':n,'ranked':ranked,'frequency':{v:{str(i):first[v][i] for i in ranked[v]} for v in first},'shared_removed':len(shared),'sizes':[s for s in [6,9,12,25,50] if s<=n]}
  lists['decoded']={v:{str(i):tok.decode([i]) for i in ranked[v]} for v in ranked}
  lists['boundary_ties']={v:{str(size):{'frequency':first[v][ranked[v][size-1]],'all_tied_ids':[i for i in ranked[v] if first[v][i]==first[v][ranked[v][size-1]]],'selected_ids':[i for i in ranked[v][:size] if first[v][i]==first[v][ranked[v][size-1]]]} for size in lists['sizes']} for v in ranked}
  refs=[]
  for v,lab in [('MSA','lang1'),('EGY','lang2')]:
   for j,(ws,ls) in enumerate(env['split'][v][0][:300]):
    ids,pos,_=env['word_positions'](tok,ws,ls,lab)
    if pos:refs.append({'variety':v,'index':j,'ids':ids,'positions':pos,'sid_candidates':lookup[(tuple(ws),tuple(ls))]})
  legacy=json.loads((root/f'results/exp06/{a.slug}/conditions.json').read_text())
  names=['baseline','top1','top3','top5']+[f'random5_d{i}' for i in range(4)]
  assert all(k in legacy and legacy[k]['where']=='target' for k in names)
  pilot=[it['item'] for g in GROUPS for it in env['items'][g][:3]]
  # Baseline and all historically recorded target random sets are preserved.
  old[old.item.isin(pilot)&old.condition.isin(names)].to_csv(out/'pilot_historical.csv',index=False)
  save(out/'ITEMS.json',items);save(out/'REFERENCES.json',refs);save(out/'TOKEN_LISTS.json',lists);save(out/'LEGACY_CONDITIONS.json',legacy);save(out/'AGGREGATE_GATE.json',aggregate)
  save(out/'PREPARED.json',{'status':'PREPARED_NO_MODEL_INFERENCE','model':m,'project':str(root),'pilot_items':pilot,'pilot_conditions':names,'B_long_sha256':sha(path),'counts':counts,'probe_weights_only':True,'historical_config_status':'conditional reconstruction, full saved item sequence and aggregate matched; flags not independently logged','token_list_identity':'NEW deterministic list; do not claim exact historical D token membership','forward_calls':0})
  save(out/'FROZEN_FILES.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
  print(json.dumps({'status':'PREPARED','slug':a.slug,'items':counts,'pilot_items':pilot,'out':str(out)},ensure_ascii=False))
 except Exception as exc:
  save(out/'STOP.json',{'status':'STOPPED','error':repr(exc),'forward_calls':0});raise
if __name__=='__main__':main()
