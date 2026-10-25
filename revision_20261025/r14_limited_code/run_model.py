"""R1.4 limited production k=1,10,20. FP32 MPS; transactional checkpoints."""
import os,sys
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='1')
sys.dont_write_bytecode=True
import argparse,datetime,fcntl,json,signal,time,traceback,shutil
from checkpoint import Store,atomic_json
class PlannedPause(Exception):pass
from pathlib import Path
from pilot_support import sha,save,require_hash,metrics_from_tokens
from legacy_helpers import load_helpers
PKG=Path(__file__).resolve().parent
GROUPS=['MSA_in_EGY','EGY_in_MSA','mono_MSA','mono_EGY']
ARRAYS=['metrics','pool_logprobs','continuation_token_logprobs','target_pool_raw_logits']
def equal(a,b,np):return a.shape==b.shape and bool(np.isclose(a,b,atol=1e-4,rtol=1e-4,equal_nan=True).all())
def maxdiff(a,b,np):
 x=np.abs(a-b);return float(np.nanmax(x)) if np.isfinite(x).any() else 0.
def order_items(items):
 first=[max((i for i,x in enumerate(items) if x['group']==g),key=lambda i:(len(items[i]['ids']),-i)) for g in GROUPS]
 return first+[i for i in range(len(items)) if i not in first]
def check_bridge(anchor,current,np):
 report={};ok=True
 for name in ARRAYS:
  a=anchor[name];b=current[name];passed=equal(a,b,np);report[name]={'passed':passed,'max_abs_diff':maxdiff(a,b,np)};ok=ok and passed
 da=current['metrics'][1]-current['metrics'][0];db=anchor['metrics'][1]-anchor['metrics'][0]
 report['top5_delta']={'passed':equal(da,db,np),'max_abs_diff':maxdiff(da,db,np)};ok=ok and report['top5_delta']['passed']
 report['anchor_flip_identical']=bool(np.array_equal(current['metrics'][:,0]<0,anchor['metrics'][:,0]<0));ok=ok and report['anchor_flip_identical']
 return bool(ok),report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 if sys.flags.optimize:raise RuntimeError('Python optimization must be disabled')
 root=a.project.resolve();out=a.out.resolve();src=PKG/'models'/a.model
 if Path(sys.prefix).resolve()!=(root/'.venv').resolve():raise RuntimeError('Use the existing project .venv/bin/python')
 if root not in out.parents:raise RuntimeError('Output must be inside project')
 if out.exists() and not a.resume:raise RuntimeError('Existing output; use the queue for a verified planned resume')
 out.mkdir(parents=True,exist_ok=a.resume);(out/'tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(out/'tmp')
 lock=(root/'ELECTRONICS_R14_RANDOM_MPS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.monotonic();forward=0;state={'status':'STARTING','completed_items':[],'outputs':{},'model':a.model};sources={};timings=[];forecasts=[];error=None;db=None;manifest={}
 def log(event,**kw):
  row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**kw}
  with (out/'actions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 def tick():
  if time.monotonic()-started>21000:raise TimeoutError('Stop before six-hour session limit')
 def alarm(*args):raise TimeoutError('Six-hour hard limit')
 def register(p):state['outputs'][str(p.relative_to(out))]=sha(p);save(out/'STATE.tmp',state);(out/'STATE.tmp').replace(out/'STATE.json')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(21600)
 try:
  manifest=json.loads((PKG/'MANIFEST.json').read_text())
  for rel,h in manifest.items():tick();require_hash(PKG/rel,h)
  state['package_manifest_sha256']=sha(PKG/'MANIFEST.json')
  spec=json.loads((PKG/'EXPECTED_INPUTS.json').read_text());m=next(x for x in spec['models'] if x['slug']==a.model)
  pinned=spec['common_inputs']+m['snapshot_files']+[{'path':m['probe_path'],'sha256':m['probe_sha256']}]
  for rec in pinned:
   tick();p=root/rec.get('path',rec.get('relative_to_New_Paper'));require_hash(p,rec['sha256']);sources[str(p)]=rec['sha256']
  identity={'model':a.model,'package':state['package_manifest_sha256'],'source_hashes':sources}
  db=Store(out/'RESULTS.sqlite',identity,resume=a.resume)
  db.set('status','RUNNING')
  save(out/'SOURCE_INPUTS.json',sources)
  for name in ['ITEMS.json','DESIGN.json','reference_means_norms.npz','ANCHOR_PROVENANCE.json','RANDOM_DESIGN.json','LIMITED_DESIGN.json']:shutil.copy2(src/name,out/name);register(out/name)
  import numpy as np,pandas as pd,torch,transformers
  from transformers import AutoModelForCausalLM
  expected=json.loads((src/'RUNTIME.json').read_text())
  for key,value in [('torch',torch.__version__),('numpy',np.__version__),('transformers',transformers.__version__)]:
   if expected[key]!=value:raise RuntimeError(f'Environment drift {key}: {value} != {expected[key]}')
  if 'torch_threads' in expected:torch.set_num_threads(expected['torch_threads'])
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable. No CPU/CUDA substitute or retry')
  env=load_helpers(root,np,pd,torch);items=json.loads((src/'ITEMS.json').read_text());design=json.loads((src/'DESIGN.json').read_text())
  assert [it['item'] for it in items]==list(range(len(items)))
  random_design=json.loads((src/'LIMITED_DESIGN.json').read_text())
  rank=design['heads'][20];assert len({tuple(h) for h in rank})==20
  layers=sorted({l for l,h in rank});order=order_items(items);save(out/'EXECUTION_ORDER.json',order)
  with np.load(src/'reference_means_norms.npz',allow_pickle=False) as z:
   means={L:(.5*(torch.from_numpy(z[f'MSA_{L}'].copy())+torch.from_numpy(z[f'EGY_{L}'].copy()))).to('mps') for L in layers}
  allowed=[np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType,np.dtypes.Int32DType]
  with torch.serialization.safe_globals(allowed):probe=torch.load(root/m['probe_path'],map_location='cpu',weights_only=True)
  R=m['layer'];w=probe['w'].float().to('mps');b=torch.as_tensor(probe['b']).float().to('mps')
  snap=(root/m['snapshot_files'][0]['path']).parent
  model,info=AutoModelForCausalLM.from_pretrained(str(snap),local_files_only=True,dtype=torch.float32,output_loading_info=True)
  save(out/'LOADING_INFO.json',json.loads(json.dumps(info,default=str)))
  assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
  model=model.to('mps').eval();nh,hd=env['head_geometry'](model)
  assert model.config.hidden_size==w.numel() and all(L<R and 0<=h<nh for L,h in rank)
  assert all(v.numel()==nh*hd for v in means.values())
  runtime={'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'class':type(model).__name__,'backend':model.config._attn_implementation,'dtype':str(next(model.parameters()).dtype),'device':'mps','num_layers':len(env['get_layers'](model)),'num_heads':nh,'head_dim':hd,'readout_layer':R,'torch_threads':torch.get_num_threads(),'eval':not model.training}
  assert all(runtime[k]==v for k,v in expected.items()),(runtime,expected);save(out/'RUNTIME.json',runtime)
  ei=model.get_input_embeddings().weight;eo=model.get_output_embeddings().weight
  if a.model=='Fanar-1-9B':assert ei.data_ptr()!=eo.data_ptr() and not torch.equal(ei[:64].detach().cpu(),eo[:64].detach().cpu())
  save(out/'EMBEDDING_AUDIT.json',{'distinct_storage':ei.data_ptr()!=eo.data_ptr(),'config_modified':False,'weights_modified':False})
  pool=design['pools']['MSA']+design['pools']['EGY'];poolids=torch.tensor(pool,device='mps');peak=0
  def resource():
   nonlocal peak
   tick();used=torch.mps.driver_allocated_memory();cap=torch.mps.recommended_max_memory();peak=max(peak,used)
   if used>.8*cap:raise RuntimeError('MPS driver memory exceeds 80% recommended limit')
  resource()
  state['status']='RUNNING';save(out/'STATE.json',state)
  tasks=[{'id':-1,'k':0,'condition_ids':[]}]+[random_design['batches'][i] for i in random_design['active_batches']]
  save(out/'TASK_COUNTS.json',{'items':len(items),'batches_per_item':len(tasks),'expected_batches':len(items)*len(tasks),'active_k':[1,10,20]})
  max_batch=0.
  for index,itemid in enumerate(order):
   it=items[itemid];begin=time.monotonic()
   anchor_path=src/'anchors'/f'item_{itemid:04d}.npz'
   with np.load(anchor_path,allow_pickle=False) as z:anchor={k:z[k].copy() for k in ARRAYS}
   for task in tasks:
    if db.has(itemid,task['id']):continue
    if time.monotonic()-started+max(120.,max_batch*1.5)>20700:
     raise PlannedPause('Clean checkpoint before six-hour session boundary')
    if shutil.disk_usage(out).free<5*1024**3:raise RuntimeError('Less than 5 GiB free; stop without retry')
    resource();cids=task['condition_ids'];hs=[random_design['conditions'][c]['heads'] for c in cids]
    if task['id']==-1:hs=[[],design['heads'][5]]+[[]]*6
    else:hs=[[]]+hs+[[]]*(7-len(hs))
    assert len(hs)==8
    seq=it['ids'];t=it['t'];call_begin=time.monotonic();forward+=1
    with env['AblationHooks'](model,[(t,h) for h in hs],hd,means),torch.no_grad():
     o=model(input_ids=torch.tensor([seq]*8,device='mps'),output_hidden_states=True,logits_to_keep=len(seq)-t)
    own=((o.hidden_states[R][:,t].float()*w).sum(-1)+b).cpu().numpy();own=own if it['own']=='EGY' else -own
    lp=torch.log_softmax(o.logits.float(),-1);pl=lp[:,0,poolids].cpu().numpy();raw=o.logits[:,0,poolids].float().cpu().numpy()
    targets=torch.tensor(seq[t+1:],device='mps');tl=lp[:,:-1].gather(-1,targets[None,:,None].expand(8,-1,1))[...,0].cpu().numpy()
    met=metrics_from_tokens(it,own,tl,np);cur=dict(metrics=met,pool_logprobs=pl,continuation_token_logprobs=tl,target_pool_raw_logits=raw)
    assert all(np.isfinite(cur[k]).all() for k in ARRAYS if k!='metrics') and np.isfinite(met[:,0]).all()
    assert np.array_equal(np.isnan(met),np.repeat(np.isnan(anchor['metrics'][:1]),8,axis=0))
    padding=[j for j,h in enumerate(hs) if not h]
    baseline_ok=all(equal(v[j],anchor[name][0],np) for j in padding for name,v in cur.items())
    flip_ok=bool(np.all((met[padding,0]<0)==(anchor['metrics'][0,0]<0)))
    if task['id']==-1:passed,bridge=check_bridge(anchor,{k:v[:2] for k,v in cur.items()},np)
    else:passed=True;bridge=None
    report={'batch':task['id'],'k':task['k'],'condition_ids':cids,'baseline_checks_passed':baseline_ok,'baseline_flip_identical':flip_ok,'bridge':bridge,'seconds':time.monotonic()-call_begin}
    if not baseline_ok or not flip_ok or not passed:
     np.savez_compressed(out/'FAILURE_ARRAYS.npz',**cur);save(out/'FAILURE_CHECKS.json',{'item':itemid,**report});raise RuntimeError('Bridge/baseline failure; no retry or tolerance change')
    cur.update(item=np.array(itemid),k=np.array(task['k']),condition_ids=np.array(cids,dtype=np.int64))
    db.put(itemid,task['id'],cur,report,np)
    max_batch=max(max_batch,time.monotonic()-call_begin)
    del o,lp;resource()
   state['completed_items'].append(itemid);timings.append(time.monotonic()-begin)
   atomic_json(out/'PROGRESS.json',{'model':a.model,'complete_items_this_scan':len(state['completed_items']),'items_total':len(items),'stored_batches':db.count(),'expected_batches':len(items)*len(tasks),'session_calls':forward,'elapsed_seconds':time.monotonic()-started,'max_batch_seconds':max_batch})
   log('item_complete',item=itemid,completed=len(state['completed_items']),total=len(items),stored_batches=db.count(),session_calls=forward)
   if index<4:
    forecasts.append({'group':it['group'],'item':itemid,'seconds_this_session':timings[-1]})
    save(out/'FORECAST.json',{'longest_per_group':forecasts,'elapsed_seconds':time.monotonic()-started,'full_run_forecast_not_a_gate':True,'sessions_bounded_to_six_hours':True})
  if db.count()!=len(items)*len(tasks):raise RuntimeError('Unexpected batch count')
  state['status']='COMPLETE'
 except PlannedPause as exc:
  state['status']='PAUSED_BUDGET';log('planned_pause',reason=str(exc))
 except BaseException as exc:
  error=repr(exc);state['status']='STOPPED_PENDING_REVIEW';save(out/'STOP.json',{'error':error,'traceback':traceback.format_exc(),'forward_calls':forward});log('stopped',error=error)
 finally:
  signal.alarm(0)
  unchanged={p:Path(p).is_file() and sha(Path(p))==h for p,h in sources.items()};save(out/'SOURCE_AFTER.json',unchanged)
  package_after={rel:(PKG/rel).is_file() and sha(PKG/rel)==h for rel,h in json.loads((PKG/'MANIFEST.json').read_text()).items()};save(out/'PACKAGE_AFTER.json',package_after)
  if not all(package_after.values()):state['status']='STOPPED_PACKAGE_CHANGED'
  if not all(unchanged.values()):state['status']='STOPPED_SOURCE_CHANGED'
  if db is not None:
   db.set('status',state['status']);state['stored_batches']=db.count();db.close()
  atomic_json(out/'STATE.json',state);save(out/'SESSION.json',{'status':state['status'],'error':error,'items':len(state['completed_items']),'forward_calls':forward,'reference_forwards':0,'seconds':time.monotonic()-started,'automatic_retry':False,'automatic_resume':'only clean PAUSED_BUDGET; user-authorized overnight queue'})
  shutil.copytree(PKG,out/'executed_package',dirs_exist_ok=True,ignore=shutil.ignore_patterns('models','__pycache__'))
  save(out/'OUTPUT_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_MANIFEST.json' and 'tmp' not in p.relative_to(out).parts})
 return 0 if state['status']=='COMPLETE' else (75 if state['status']=='PAUSED_BUDGET' else 1)
if __name__=='__main__':raise SystemExit(main())
