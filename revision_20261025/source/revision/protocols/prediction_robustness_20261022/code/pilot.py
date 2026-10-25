"""Bounded twelve-item exp06 replay. Does not run the full robustness study."""
import argparse,collections,gc,hashlib,json,os,re,sys,time,zipfile,shutil
from pathlib import Path
from prepare import sha,save,load_helpers,TASK
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ENABLE_MPS_FALLBACK='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True,type=Path);a=ap.parse_args();p=a.prepared.resolve();s=json.loads((p/'PREPARED.json').read_text());root=Path(s['project']);m=s['model'];out=p/'pilot';out.mkdir(exist_ok=False);(out/'reference_units').mkdir();shutil.copytree(TASK/'code',out/'executed_code',ignore=shutil.ignore_patterns('__pycache__'))
 begun=time.monotonic();forward=0
 def log(event,**kw):
  with (out/'actions.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':event,**kw},default=str)+'\n')
  print(event,flush=True)
 try:
  for rel,h in json.loads((p/'FROZEN_FILES.json').read_text()).items():assert sha(p/rel)==h,rel
  for check in json.loads((p/'INPUT_CHECKS.json').read_text()):assert sha(root/check['path'])==check['expected'],check['path']
  assert sha(root/f"results/exp06/{m['slug']}/B_long.csv")==s['B_long_sha256']
  import numpy as np,pandas as pd,torch,transformers
  from transformers import AutoModelForCausalLM
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU or different model fallback')
  allowed=[np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType,np.dtypes.Int32DType]
  with torch.serialization.safe_globals(allowed):probe=torch.load(root/m['probe_path'],map_location='cpu',weights_only=True)
  w=probe['w'].float().to('mps');b=torch.as_tensor(probe['b']).float().to('mps');env=load_helpers(root,np,pd,torch)
  snap=(root/m['snapshot_files'][0]['path']).parent
  model,info=AutoModelForCausalLM.from_pretrained(str(snap),local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
  save(out/'loading_info.json',json.loads(json.dumps(info,default=lambda x:sorted(x) if isinstance(x,set) else str(x))))
  assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
  model=model.to('mps').eval();nh,hd=env['head_geometry'](model)
  runtime={'class':type(model).__name__,'backend':model.config._attn_implementation,'num_layers':len(env['get_layers'](model)),'num_heads':nh,'head_dim':hd,'dtype':str(next(model.parameters()).dtype),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'mps','eval':not model.training}
  for k in ['class','backend','num_layers','num_heads','head_dim']:assert runtime[k]==m['runtime'][k],(k,runtime[k],m['runtime'][k])
  assert runtime['num_layers']==m['num_hidden_layers'] and w.numel()==model.config.hidden_size
  save(out/'runtime.json',runtime)
  def resource():
   if time.monotonic()-begun>1800:raise RuntimeError('Thirty-minute pilot resource limit; preserve partial evidence')
   used=torch.mps.driver_allocated_memory();cap=torch.mps.recommended_max_memory()
   if used>0.8*cap:raise RuntimeError('MPS memory exceeds 80% recommended limit')
   return {'mps_driver':used,'recommended':cap,'seconds':time.monotonic()-begun,'forward_calls':forward}
  legacy=json.loads((p/'LEGACY_CONDITIONS.json').read_text());top=[tuple(map(int,re.fullmatch(r'L(\d+)H(\d+)',x).groups())) for x in legacy['top5']['heads']];layers=sorted({L for L,h in top})
  refs=json.loads((p/'REFERENCES.json').read_text());captures={v:{L:[] for L in layers} for v in ['MSA','EGY']}
  for i,r in enumerate(refs):
   cap=env['collect_head_inputs'](model,r['ids'],r['positions'],layers,'mps');forward+=1
   assert all(torch.isfinite(x).all() for x in cap.values())
   np.savez_compressed(out/'reference_units'/f'unit_{i:04d}.npz',**{str(L):x.numpy() for L,x in cap.items()})
   for L in layers:captures[r['variety']][L].append(cap[L])
   if (i+1)%20==0:log('reference_progress',done=i+1,total=len(refs),**resource())
  captures={v:{L:torch.cat(xs) for L,xs in d.items()} for v,d in captures.items()}
  separate={v:{L:captures[v][L].mean(0) for L in layers} for v in captures}
  means={L:(0.5*(separate['MSA'][L]+separate['EGY'][L])).to('mps') for L in layers}
  norms={L:env['head_output_norms'](model,torch.cat([captures['MSA'][L],captures['EGY'][L]]),L,nh,hd) for L in layers}
  np.savez_compressed(out/'reference_means_norms.npz',**{f'{v}_{L}':separate[v][L].numpy() for v in separate for L in layers},**{f'norm_{L}':norms[L] for L in layers})
  del captures;gc.collect();log('reference_complete',**resource())
  # Audit realized historical target random sets against stated norm bounds.
  norm_checks=[]
  for name,c in legacy.items():
   if not name.startswith('random5_d'):continue
   hs=[tuple(map(int,re.fullmatch(r'L(\d+)H(\d+)',x).groups())) for x in c['heads']]
   for (L,h),(L2,h2) in zip(top,hs):
    ratio=float(norms[L2][h2]/max(norms[L][h],1e-8));norm_checks.append({'condition':name,'source':[L,h],'random':[L2,h2],'ratio':ratio,'same_layer':L==L2,'within_half_double':0.5<=ratio<=2})
  save(out/'legacy_norm_matching.json',norm_checks)
  lists=json.loads((p/'TOKEN_LISTS.json').read_text());n=lists['balanced_n'];ids={v:torch.tensor(lists['ranked'][v][:n],device='mps') for v in ['MSA','EGY']}
  items=json.loads((p/'ITEMS.json').read_text());hist=pd.read_csv(p/'pilot_historical.csv').set_index(['item','condition']);records=[];gates=[]
  for itemid in s['pilot_items']:
   it=items[itemid];assert it['item']==itemid;names=s['pilot_conditions'];t=it['t'];seq=it['ids'];k=len(seq)-t
   plans=[(t,[tuple(map(int,re.fullmatch(r'L(\d+)H(\d+)',h).groups())) for h in legacy[c]['heads']]) for c in names]
   with env['AblationHooks'](model,plans,hd,means),torch.no_grad():o=model(input_ids=torch.tensor([seq]*len(names),device='mps'),output_hidden_states=True,logits_to_keep=k)
   forward+=1;z=((o.hidden_states[m['layer']][:,t].float()*w).sum(-1)+b).cpu().numpy();lp=torch.log_softmax(o.logits.float(),-1)
   D=(torch.logsumexp(lp[:,0,ids['EGY']],-1)-torch.logsumexp(lp[:,0,ids['MSA']],-1)).cpu().numpy();assert np.isfinite(z).all() and np.isfinite(D).all()
   tgt=torch.tensor(seq[t+1:],device='mps');tokenlp=lp[:,:-1].gather(-1,tgt[None,:,None].expand(len(names),-1,1))[...,0].cpu().numpy();assert np.isfinite(tokenlp).all()
   for bi,name in enumerate(names):
    row={'item':itemid,'condition':name,'own_logit':float(z[bi] if it['own']=='EGY' else -z[bi]),'D_new_to_matrix':float(D[bi] if it['matrix']=='EGY' else -D[bi])}
    for role in ['matrix','own','all']:
     vals=[tokenlp[bi,a-t-1:e-t-1] for a,e,lab in it['cont'] if lab and (role=='all' or lab==it[role])]
     row['cont_lp_'+role]=float(np.concatenate(vals).mean()) if vals else np.nan
    for metric in ['own_logit','cont_lp_matrix','cont_lp_own','cont_lp_all']:
     old=float(hist.loc[(itemid,name),metric]);new=row[metric];atol=0.01 if metric=='own_logit' else 0.0001
     ok=bool(np.isclose(old,new,atol=atol,rtol=0.0001,equal_nan=True));gates.append({'item':itemid,'condition':name,'metric':metric,'historical':None if np.isnan(old) else old,'new':None if np.isnan(new) else new,'atol':atol,'rtol':0.0001,'passed':ok})
    row['D_historical_to_matrix']=float(hist.loc[(itemid,name),'D_to_matrix']);row['D_difference']=row['D_new_to_matrix']-row['D_historical_to_matrix'];records.append(row)
   del o,lp;resource();pd.DataFrame(records).to_csv(out/'pilot_comparison.csv',index=False);save(out/'REPLAY_GATES.json',gates);log('pilot_item',item=itemid,**resource())
   if not all(x['passed'] for x in gates):raise RuntimeError('Replay mismatch; no full experiment authorized')
  save(out/'SUMMARY.json',{'status':'PILOT_PASS_NO_FULL_STUDY','items':12,'conditions_per_item':8,'comparisons':len(gates),'all_non_D_replay_passed':all(x['passed'] for x in gates),'D_identity':'new deterministic lists, even agreement does not prove historical identity','max_abs_D_difference':max(abs(x['D_difference']) for x in records),'norm_violations':sum(not x['within_half_double'] or not x['same_layer'] for x in norm_checks),'forward_calls':forward,'seconds':time.monotonic()-begun,'full_experiment_run':False})
  for check in json.loads((p/'INPUT_CHECKS.json').read_text()):assert sha(root/check['path'])==check['expected'],check['path']
  assert sha(root/f"results/exp06/{m['slug']}/B_long.csv")==s['B_long_sha256'];log('originals_unchanged')
 except Exception as e:
  save(out/'STOP.json',{'status':'STOPPED','error':repr(e),'forward_calls':forward,'seconds':time.monotonic()-begun});raise
 finally:
  paths=[x for x in p.rglob('*') if x.is_file() and 'tmp' not in x.relative_to(p).parts and '__pycache__' not in x.parts]
  deferred=[x for x in paths if 'reference_units' in x.parts]
  save(p/'RETURN_CONTENTS.json',{'core':[{ 'path':str(x.relative_to(p)),'sha256':sha(x)} for x in paths if x not in deferred], 'deferred_reference_units':[{'path':str(x.relative_to(p)),'sha256':sha(x)} for x in deferred],'note':'Keep reference units locally; core is not the full raw reference evidence.'})
  dest=p.parent/(p.name+'_REVIEW_CORE.zip')
  with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
   for x in paths+[p/'RETURN_CONTENTS.json']:
    if x not in deferred:z.write(x,p.name+'/'+str(x.relative_to(p)))
  with zipfile.ZipFile(dest) as z:assert z.testzip() is None
  print('RETURN',str(dest),sha(dest),flush=True)
if __name__=='__main__':main()
