from common import *
import ast,copy,shutil
import numpy as np
from transformers import AutoTokenizer
import original_helpers as h
from sklearn.model_selection import GroupKFold
import importlib.metadata

source=(ROOT/'exp02_causal_patching.py').read_text();tree=ast.parse(source);fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variety_directions'))
outer=next(n for n in fn.body if isinstance(n,ast.For));inner=next(n for n in outer.body if isinstance(n,ast.For));cut=next(i for i,n in enumerate(inner.body) if isinstance(n,ast.With));inner.body=inner.body[:cut]+ast.parse("recorded.append({'variety':var,'mono_index':int(i),'encoded':{k:v[0].tolist() for k,v in enc.items()},'positions':pos,'text':text,'n_words':len(pos)})").body
outer.body=[n for n in outer.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))]
fn.body=[ast.parse('rng=np.random.default_rng(seed)').body[0],ast.parse('recorded=[]').body[0],outer,ast.parse('return recorded').body[0]];fn.name='original_direction_selection_no_inference'
module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));(WORK/'code/original_direction_selection.py').write_text(ast.unparse(module)+'\n');ns=dict(vars(h));exec(compile(module,'<exact-original-direction-selection>','exec'),ns)
tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False);rows=h.load_lince();expected=ns[fn.name](None,tok,rows,None,n_per=400,seed=13)
frozen=[json.loads(s) for s in (WORK/'selection/direction_sample.jsonl').read_text().splitlines()];actual=[x for x in frozen if x['positions']]
assert len(expected)==len(actual)
for a,b in zip(actual,expected):
 for key in ['variety','mono_index','encoded','positions','text','n_words']:assert a[key]==b[key],(a['unit'],key)
pairs=[json.loads(s) for s in (WORK/'selection/pairs.jsonl').read_text().splitlines()][:150];comps=read('selection/components.json');discovered=[];remaining=set(range(150))
while remaining:
 found={min(remaining)};changed=True
 while changed:
  sids=set(s for i in found for s in pairs[i]['prefix_sids']);hashes=set(s for i in found for s in pairs[i]['prefix_hashes']);new={i for i in remaining-found if sids&set(pairs[i]['prefix_sids']) or hashes&set(pairs[i]['prefix_hashes'])};changed=bool(new);found|=new
 discovered.append(sorted(found));remaining-=found
assert discovered==comps['components']
save('manifest/INDEPENDENT_SELECTION_VALIDATION.json',{'all_passed':True,'direction_plan_matches_exact_original_AST':True,'direction_measurement_tweets':len(expected),'direction_sampled_tweets':len(frozen),'directions_with_no_positions':sum(not x['positions'] for x in frozen),'counts':{v:sum(x['n_words'] for x in frozen if x['variety']==v) for v in ['MSA','EGY']},'components_independent_graph_reconstruction':True,'component_count':len(discovered),'original_pair_AST_exact_tokens_and_rng_match':'asserted in prepare.py','no_forward_calls':True,'AST_adaptation':'Only inference and accumulation removed from original variety_directions; append selected tokenizer inputs and positions instead. Original RNG, monolingual selection, text, truncation and positions retained.'})
shutil.copy2(ROOT/'Electronics_Submission_Provenance/PROTOCOL_AR.txt',WORK/'manifest/SOURCE_PROTOCOL_AR.txt')
state=read('STATUS.json');assert state['status']=='PREPARED'
files=[p for p in WORK.rglob('*') if p.is_file() and (p.is_relative_to(WORK/'code') or p.is_relative_to(WORK/'selection') or p.is_relative_to(WORK/'manifest')) and p.name!='PROTOCOL_FROZEN.json']
save('manifest/PROTOCOL_FROZEN.json',{'utc':now(),'status':'FROZEN_BEFORE_FIRST_FORWARD','forward_calls':0,'files':{str(p.relative_to(WORK)):sha(p) for p in sorted(files)},'model':EXPECTED,'p1_p2_p3_atol':1e-4,'p1_p2_p3_rtol':1e-4,'early_gate_pair_ids':list(range(6)),'full_pair_ids':list(range(150)),'head_layers_union':state['union'],'head_batch':NH,'P2_attention_batch':R,'P1_condition_batch':1,'max_inference_seconds':2700,'dtype':'bfloat16','device':'mps','default_attention_implementation':True,'P2_baselines':'Reuse P1 clean/corrupt readout R from identical batch=1 inputs and original scorer; no redundant baseline forwards','direction_accumulation':'Exact ordered original torch CPU float32 addition; per-tweet contributions saved for no-repeat resume','gate_failure_policy':'STOPPED, no tuning or full extraction after early failure','incomplete_policy':'PARTIAL at OOM/nonfinite/budget, absent files not successful zeros','readout_not_crossfit':True,'historical_linkage':'conditional, not proof of missing historical source IDs'})
state.update(status='FROZEN',frozen_utc=now());save('STATUS.json',state);log('selection_independently_verified_and_frozen',status='FROZEN',code_sha256=sha(Path(__file__)),protocol_sha256=sha(WORK/'manifest/PROTOCOL_FROZEN.json'),components=len(discovered))
