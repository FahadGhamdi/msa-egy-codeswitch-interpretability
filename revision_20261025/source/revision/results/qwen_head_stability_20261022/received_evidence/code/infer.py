from common import *
import gc,resource,time,traceback
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
import original_helpers as h
class EngineeringStop(RuntimeError):pass
class ResourceStop(RuntimeError):pass

def memory(label):
 import psutil
 x={'utc':now(),'label':label,'rss':psutil.Process().memory_info().rss,'system_available':psutil.virtual_memory().available,'mps_driver':torch.mps.driver_allocated_memory(),'mps_recommended':torch.mps.recommended_max_memory(),'mps_allocated':torch.mps.current_allocated_memory()}
 with (WORK/'logs/memory.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 return x

def main():
 frozen=read('manifest/PROTOCOL_FROZEN.json');state=read('STATUS.json')
 if state['status'] not in ['FROZEN','PARTIAL']:raise RuntimeError('Inference requires frozen protocol or documented partial resume')
 for p,d in frozen['files'].items():
  if sha(WORK/p)!=d:raise RuntimeError('Frozen input/code changed: '+p)
 for p,d in read('manifest/inputs_before.json').items():
  if sha(p)!=d['sha256']:raise RuntimeError('Original input changed: '+p)
 if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable: native GPU permission required; no CPU substitution')
 pairs=[json.loads(x) for x in (WORK/'selection/pairs.jsonl').read_text().splitlines()][:150]
 samples=[json.loads(x) for x in (WORK/'selection/direction_sample.jsonl').read_text().splitlines()]
 layers=read('selection/LAYER_SELECTION.json')['union'];p1old=pd.read_csv(ROOT/'results/exp02/Qwen3-8B-Base/P1_scores_long.csv').set_index(['pair','prefix','layer']);p2old=pd.read_csv(ROOT/'results/exp02/Qwen3-8B-Base/P2_patching_long.csv');p2old=p2old[(p2old.target=='MSA')&(p2old.component=='attn')].set_index(['pair','layer'])
 progress=read('manifest/INFERENCE_CHECKPOINT.json') if (WORK/'manifest/INFERENCE_CHECKPOINT.json').exists() else {'forward_calls':0,'inference_seconds':0.,'direction_units':0,'p12_pairs':0,'head_pairs':0}
 base_seconds=progress['inference_seconds'];clock_start=None;model=None;phase='load';cache=None;mu_msa=mu_egy=None
 def elapsed():return base_seconds+(time.perf_counter()-clock_start if clock_start is not None else 0.)
 def check_time():
  if elapsed()>=2700:raise ResourceStop('45 minute inference budget exhausted')
 def checkpoint(event):
  progress.update(inference_seconds=elapsed(),utc=now(),phase=phase,event=event);save('manifest/INFERENCE_CHECKPOINT.json',progress)
 def committed(path):
  progress.setdefault('committed_files',{})[str(path.relative_to(WORK))]=sha(path)
 def finite(x):
  if not np.isfinite(x).all():raise ResourceStop('Nonfinite values; no clamp or substitute')
 def compare(records,path):
  df=pd.DataFrame(records);df.to_csv(WORK/path,index=False)
  ok=all(x['pass'] for x in records);return {'passed':ok,'values':len(records),'failed':sum(not x['pass'] for x in records),'max_absolute_difference':max((x['absolute_difference'] for x in records),default=None),'max_tolerance_ratio':max((x['absolute_difference']/x['tolerance'] for x in records),default=None),'atol':1e-4,'rtol':1e-4}
 def cmp(kind,i,L,metric,new,old,**kw):
  finite([new,old]);d=abs(new-old);tol=1e-4+1e-4*abs(old)
  return dict(kind=kind,pair=i,layer=L,metric=metric,new=float(new),old=float(old),absolute_difference=d,tolerance=tol,**kw,**{'pass':bool(d<=tol)})
 # Every actual model.forward is counted and time-limited without changing its arguments/results.
 try:
  log('model_loading',snapshot=str(SNAP));loadstart=time.perf_counter()
  model,info=AutoModelForCausalLM.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,output_loading_info=True)
  save('manifest/model_loading_info.json',info)
  if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise EngineeringStop('Model key conflict')
  model=model.to('mps').eval();torch.mps.synchronize();cfg=model.config
  assert len(h.get_layers(model))==NL and cfg.num_attention_heads==NH and {p.dtype for p in model.parameters()}=={torch.bfloat16} and all(p.device.type=='mps' for p in model.parameters())
  head_dim=getattr(cfg,'head_dim',None) or cfg.hidden_size//NH
  save('manifest/model_runtime.json',{'class':type(model).__name__,'backend':cfg._attn_implementation,'dtype':'bfloat16','device':'mps','eval':not model.training,'num_layers':NL,'readout_layer':R,'num_heads':NH,'head_dim':head_dim,'load_seconds':time.perf_counter()-loadstart,'fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],'default_attention_preserved':True,'memory':memory('after_load')})
  clock_start=time.perf_counter();forward_original=model.forward
  def forward_counted(*a,**kw):
   check_time();progress['forward_calls']+=1
   return forward_original(*a,**kw)
  model.forward=forward_counted
  for file,d in progress.get('committed_files',{}).items():
   if sha(WORK/file)!=d:raise EngineeringStop('Resume committed output digest conflict: '+file)
  # Exact original per-tweet float32 summation on device, then CPU float32 accumulation.
  phase='directions';accs={};counts={};done=0
  for var in ['MSA','EGY']:
   acc=None;n=0
   for sample in [s for s in samples if s['variety']==var]:
    uid=sample['unit'];p=WORK/f'directions/unit_{uid:03d}.npz'
    if p.exists():
     data=np.load(p,allow_pickle=False);vec=data['sum'];assert int(data['n_words'])==sample['n_words'];finite(vec)
    else:
     check_time();pos=sample['positions']
     if pos:
      with torch.no_grad():
       hs=model(**{k:torch.tensor([v],device='mps') for k,v in sample['encoded'].items()},output_hidden_states=True).hidden_states
      sums=torch.stack([x[0,pos].float().sum(0) for x in hs]).cpu();vec=sums.numpy().copy();del hs,sums
     else:vec=np.zeros((NL+1,cfg.hidden_size),np.float32)
     finite(vec);np.savez(p,sum=vec,n_words=np.int64(sample['n_words']));committed(p)
    if sample['n_words']:acc=torch.from_numpy(vec.copy()) if acc is None else acc+torch.from_numpy(vec);n+=sample['n_words']
    done+=1;progress['direction_units']=done;checkpoint('direction_committed')
    if done==20 and not (WORK/'manifest/FIRST_20_DIRECTION_RESOURCE_GATE.json').exists():
     mem=memory('first20_direction');safe=mem['mps_driver']<=mem['mps_recommended'];save('manifest/FIRST_20_DIRECTION_RESOURCE_GATE.json',{'memory':mem,'passed':safe,'elapsed_seconds':elapsed(),'estimated_800_tweet_seconds':elapsed()*40,'direction_tweets_completed':20});log('direction_resource_gate',status='PASS' if safe else 'PARTIAL')
     if not safe:raise ResourceStop('Direction memory resource gate exceeded')
    if done%50==0:log('directions_progress',completed=done,seconds=elapsed(),forward_calls=progress['forward_calls'])
   assert n>0;accs[var]=acc;counts[var]=n
  mu_msa=accs['MSA']/counts['MSA'];mu_egy=accs['EGY']/counts['EGY'];finite(mu_msa.numpy());finite(mu_egy.numpy());del accs
  for var,mu in [('MSA',mu_msa),('EGY',mu_egy)]:np.save(WORK/f'directions/mu_{var}.npy',mu.numpy(),allow_pickle=False)
  save('directions/SUMMARY.json',{'counts':counts,'layer_indices':list(range(NL+1)),'shape':list(mu_msa.shape),'dtype':'float32','sampled_tweets_each':400,'completed_tweets':done,'method':'exact original device float32 last-word-subtoken sum then ordered CPU float32 accumulation and division; no saved logistic probe','seconds_through_directions':elapsed()})
  score=h.make_scorer(mu_msa,mu_egy,'mps');denom=((mu_egy-mu_msa)**2).sum(-1);assert bool((denom>0).all());del mu_msa,mu_egy;mu_msa=mu_egy=None
  # Early gate six pairs, then remaining pairs, using the same saved rows without repeated forwards.
  phase='p1_p2';all1=[];all2=[];comparisons=[]
  for p in pairs:
   i=p['pair'];dest=WORK/f'results/p12_pair_{i:03d}.json'
   if dest.exists():rec=json.loads(dest.read_text());assert rec['pair']==i
   else:
    check_time();start=time.perf_counter();s1=[]
    for cond in ['clean','corrupt','alone']:
     sc=h.run_scores(model,[p[cond]],'mps',score)[0];finite(sc)
     for L,v in enumerate(sc):s1.append({'pair':i,'target':'MSA','word':p['word'],'condition':cond,'prefix':{'clean':'EGY','corrupt':'MSA','alone':'none'}[cond],'layer':L,'s':float(v)})
    clean=next(x['s'] for x in s1 if x['condition']=='clean' and x['layer']==R);corr=next(x['s'] for x in s1 if x['condition']=='corrupt' and x['layer']==R)
    cache=h.capture_clean(model,p['clean'],'mps');sp=h.patched_scores(model,p['corrupt'],'mps',cache,'attn',list(range(R)),R,score);finite(sp);del cache;cache=None
    s2=[{'pair':i,'target':'MSA','component':'attn','layer':L,'s_clean':clean,'s_corrupt':corr,'s_patched':float(v),'delta':float(v-corr),'gap':clean-corr} for L,v in enumerate(sp)]
    rec={'pair':i,'P1':s1,'P2':s2,'seconds':time.perf_counter()-start,'cumulative_seconds':elapsed()};save(str(dest.relative_to(WORK)),rec);committed(dest)
   all1+=rec['P1'];all2+=rec['P2']
   for x in rec['P1']:comparisons.append(cmp('P1',i,x['layer'],'s',x['s'],float(p1old.loc[(i,x['prefix'],x['layer']),'s']),prefix=x['prefix']))
   for x in rec['P2']:
    for key in ['s_clean','s_corrupt','s_patched','delta','gap']:comparisons.append(cmp('P2',i,x['layer'],key,x[key],float(p2old.loc[(i,x['layer']),key])))
   progress['p12_pairs']=i+1;checkpoint('p12_pair_committed')
   if i==0 and not (WORK/'manifest/FIRST_PAIR_RESOURCE_GATE.json').exists():
    mem=memory('first_patching_pair');ok=mem['mps_driver']<=mem['mps_recommended'];save('manifest/FIRST_PAIR_RESOURCE_GATE.json',{'memory':mem,'passed':ok,'pair_seconds':rec['seconds'],'forward_calls':progress['forward_calls']})
    if not ok:raise ResourceStop('Patching first-pair memory gate exceeded')
   if i==5:
    result=compare(comparisons,'results/EARLY_GATE_COMPARISON.csv');save('results/EARLY_GATE.json',result);log('early_gate',status='PASS' if result['passed'] else 'STOPPED',**result)
    if not result['passed']:raise EngineeringStop('Early six-pair P1/P2 numerical gate failed')
   if i>=6 and any(not x['pass'] for x in comparisons[-(3*(NL+1)+5*R):]):
    compare(comparisons,'results/FULL_GATE_COMPARISON.csv');raise EngineeringStop('Full MSA P1/P2 gate failed at pair '+str(i))
   if (i+1)%10==0:log('p12_progress',completed=i+1,seconds=elapsed(),forward_calls=progress['forward_calls'])
  pd.DataFrame(all1).to_csv(WORK/'results/P1_replayed.csv',index=False);pd.DataFrame(all2).to_csv(WORK/'results/P2_attention_replayed.csv',index=False)
  result=compare(comparisons,'results/FULL_GATE_COMPARISON.csv');save('results/FULL_GATE.json',result)
  if not result['passed']:raise EngineeringStop('Full 150-pair numerical gate failed')
  newp2=pd.DataFrame(all2);selected=read('selection/LAYER_SELECTION.json');folds=read('selection/FOLDS.json')['folds'];ranking_checks=[]
  for fid,ids,old in [(-1,list(range(150)),selected['original_full_ranking'])]+[(f['fold'],f['train']['pair_ids'],selected['folds'][f['fold']]['ranking']) for f in folds]:
   ranked=newp2[newp2.pair.isin(ids)].groupby('layer').delta.mean().reset_index().sort_values(['delta','layer'],ascending=[False,True]).to_dict('records');same=[x['layer'] for x in old]==[x['layer'] for x in ranked];ranking_checks.append({'fold':fid,'same_full_order':same,'old':old,'new':ranked})
  save('results/LAYER_RANKING_GATE.json',{'passed':all(x['same_full_order'] for x in ranking_checks),'checks':ranking_checks})
  if not all(x['same_full_order'] for x in ranking_checks):raise EngineeringStop('New layer ranking membership/order changed; frozen union not replaced')
  log('full_p12_and_layer_ranking_passed',status='PASS',max_difference=result['max_absolute_difference'])
  phase='heads';allheads=[]
  for p in pairs:
   i=p['pair'];start=time.perf_counter();corr=float(newp2[newp2.pair==i].s_corrupt.iloc[0]);cache=None
   for L in layers:
    dest=WORK/f'results/heads_pair_{i:03d}_layer_{L:02d}.json'
    if dest.exists():rec=json.loads(dest.read_text())
    else:
     check_time()
     if cache is None:cache=h.capture_clean(model,p['clean'],'mps')
     sp=h.patched_scores(model,p['corrupt'],'mps',cache,'head',[(L,hd) for hd in range(NH)],R,score,head_dim);finite(sp)
     rec={'pair':i,'layer':L,'cumulative_seconds':elapsed(),'records':[{'pair':i,'layer':L,'head':hd,'s_corrupt':corr,'s_patched':float(v),'delta':float(v-corr)} for hd,v in enumerate(sp)]};save(str(dest.relative_to(WORK)),rec);committed(dest)
    allheads+=rec['records'];checkpoint('head_layer_committed')
   if cache is not None:del cache;cache=None
   progress['head_pairs']=i+1;checkpoint('head_pair_committed')
   if (i+1)%10==0:log('head_progress',pairs=i+1,layers=layers,seconds=elapsed(),forward_calls=progress['forward_calls'])
  raw=pd.DataFrame(allheads);raw.to_csv(WORK/'results/HEAD_DELTAS.csv',index=False);agg=raw.groupby(['layer','head']).delta.agg(['mean','std','count']).reset_index();agg.to_csv(WORK/'results/P3_reaggregated.csv',index=False)
  orig=pd.read_csv(ROOT/'results/exp02/Qwen3-8B-Base/P3_heads.csv');checks=[]
  for x in orig.to_dict('records'):
   y=agg[(agg.layer==x['layer'])&(agg['head']==x['head'])].iloc[0];assert int(y['count'])==int(x['count'])==150
   for key in ['mean','std']:checks.append(cmp('P3',-1,int(x['layer']),key,float(y[key]),float(x[key]),head=int(x['head'])))
  result=compare(checks,'results/P3_AGGREGATE_COMPARISON.csv');save('results/P3_GATE.json',result)
  if not result['passed']:raise EngineeringStop('P3 original-layer aggregate mean/sample SD gate failed')
  save('STATUS.json',{'status':'INFERENCE_COMPLETE_PENDING_ANALYSIS','reason':None,'forward_calls':progress['forward_calls'],'inference_seconds':elapsed(),'p12_pairs':150,'head_pairs':150,'head_layers':layers,'head_rows':len(raw),'directions':counts});log('inference_complete',status='PASS',seconds=elapsed())
 except Exception as e:
  sub='PARTIAL' if isinstance(e,ResourceStop) or 'out of memory' in str(e).lower() else 'STOPPED'
  (WORK/'logs/inference_failure.txt').write_text(traceback.format_exc());save('STATUS.json',{'status':sub,'reason':str(e),'phase':phase,'forward_calls':progress['forward_calls'],'inference_seconds':elapsed(),'direction_units':progress['direction_units'],'p12_pairs':progress['p12_pairs'],'head_pairs':progress['head_pairs'],'head_layers':layers});log('inference_stopped',status=sub,reason=str(e))
 finally:
  checkpoint('process_end')
  if cache is not None:del cache
  if model is not None:del model
  gc.collect();torch.mps.empty_cache()
  log('model_process_finishing',inference_seconds=progress['inference_seconds'],forward_calls=progress['forward_calls'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  (WORK/'logs/inference_preflight_failure.txt').write_text(traceback.format_exc());save('STATUS.json',{'status':'STOPPED','reason':str(e),'stage':'inference_preflight'});raise
