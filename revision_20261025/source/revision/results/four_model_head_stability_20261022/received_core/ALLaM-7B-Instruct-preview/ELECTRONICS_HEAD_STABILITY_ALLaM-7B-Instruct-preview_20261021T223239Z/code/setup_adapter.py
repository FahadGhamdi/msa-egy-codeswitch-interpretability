from pathlib import Path
import datetime,json,hashlib,shutil,difflib,ast
ROOT=Path('/Users/fahad/Documents/Research-Projects/New-Paper');TASK=ROOT/'Electronics_Head_Stability_Expansion/revision/protocols/three_model_head_stability';Q=ROOT/'ELECTRONICS_QWEN_HEAD_STABILITY_20261021T215744Z/code'
B=json.loads((TASK/'MODEL_BASELINES.json').read_text());H=json.loads((TASK/'QWEN_REFERENCE_CODE_HASHES.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in H.items():assert sha(Q/n)==h==sha(TASK/'qwen_reference_code'/n),(n,'reference mismatch')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');CONTROL=ROOT/('ELECTRONICS_HEAD_STABILITY_EXPANSION_'+stamp);CONTROL.mkdir();(CONTROL/'code').mkdir();shutil.copy2(Path(__file__),CONTROL/'code/setup.py')
status={'created_utc':stamp,'protocol':str(TASK),'models':{}}
for E in B['models']:
 slug=E['slug'];assert E==json.loads((TASK/slug/'EXPECTED_INPUTS.json').read_text())
 W=ROOT/f'ELECTRONICS_HEAD_STABILITY_{slug}_{stamp}';W.mkdir()
 for d in ['code','logs','manifest','selection','results','directions']:(W/d).mkdir()
 for n in ['NEXT_LOCAL_TASK_AR.txt','ANALYSIS_PLAN.txt','BASE_QWEN_TASK_AR.txt','MODEL_BASELINES.json','QWEN_REFERENCE_CODE_HASHES.json']:shutil.copy2(TASK/n,W/'manifest'/n)
 shutil.copy2(TASK/slug/'EXPECTED_INPUTS.json',W/'manifest/EXPECTED_INPUTS.json')
 for name in ['common.py','prepare.py','infer.py','analyze_verify.py','verify_selection_freeze.py','additional_audit.py','package.py','launch.py','verify_return.py']:
  original=(Q/name).read_text();s=original
  s=s.replace("Electronics_Four_Model_Controls/revision/protocols/qwen_head_stability","Electronics_Head_Stability_Expansion/revision/protocols/three_model_head_stability")
  for file in ['summary.json','P1_scores_long.csv','P2_patching_long.csv','P3_heads.csv']:
   s=s.replace("ROOT/'results/exp02/Qwen3-8B-Base/"+file+"'","EXPDIR/"+repr(file))
  if name=='common.py':
   s=s.replace("SNAP=ROOT/'.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots'/EXPECTED['revision']","SLUG=EXPECTED['slug'];DIM=EXPECTED['hidden_size'];EXPDIR=ROOT/'results/exp02'/SLUG\nSNAP=ROOT/'.cache/huggingface/hub'/('models--'+EXPECTED['model'].replace('/','--'))/'snapshots'/EXPECTED['revision']")
  if name=='prepare.py':
   s=s.replace("[TASK/n for n in ['EXPECTED_INPUTS.json','ANALYSIS_PLAN.txt','NEXT_LOCAL_TASK_AR.txt']]","[TASK/n for n in ['MODEL_BASELINES.json','ANALYSIS_PLAN.txt','NEXT_LOCAL_TASK_AR.txt','BASE_QWEN_TASK_AR.txt','QWEN_REFERENCE_CODE_HASHES.json']]+[TASK/SLUG/'EXPECTED_INPUTS.json']")
   s=s.replace("ROOT/'Electronics_Four_Model_Controls'/n for n in ['READ_FIRST_AR.txt','FOUR_MODEL_REPORT_AR.txt','revision/STATUS.json']","ROOT/'Electronics_Head_Stability_Expansion'/n for n in ['READ_FIRST_AR.txt','QWEN_HEAD_STABILITY_REPORT_AR.txt','revision/STATUS.json']")
   s=s.replace("save('manifest/inputs_before.json',inventory(sources))","sources += [ROOT/'ELECTRONICS_QWEN_HEAD_STABILITY_20261021T215744Z/code'/n for n in json.loads((TASK/'QWEN_REFERENCE_CODE_HASHES.json').read_text())]\n save('manifest/inputs_before.json',inventory(sources))")
   s=s.replace("assert cfg['num_hidden_layers']==NL and cfg['num_attention_heads']==NH and R==round(.8*NL)","assert cfg['num_hidden_layers']==NL and cfg['num_attention_heads']==NH and cfg['hidden_size']==DIM and R==round(.8*NL)\n head_dim=cfg.get('head_dim') or DIM//NH\n save('manifest/config_gate.json',{'hidden_size':DIM,'num_attention_heads':NH,'num_hidden_layers':NL,'readout_layer':R,'head_dim':head_dim,'explicit_head_dim':cfg.get('head_dim'),'o_proj_expected_input_features':NH*head_dim})")
   s=s.replace("original=summary['top_attn_layers_by_patching'];", "original=summary['top_attn_layers_by_patching'];assert original==EXPECTED['original_top_layers'];")
  if name=='verify_selection_freeze.py':s=s.replace("'head_batch':32,'P2_attention_batch':29","'head_batch':NH,'P2_attention_batch':R")
  if name=='infer.py':
   s=s.replace("assert len(h.get_layers(model))==NL and cfg.num_attention_heads==NH", "assert len(h.get_layers(model))==NL and cfg.num_attention_heads==NH and cfg.hidden_size==DIM")
   s=s.replace("head_dim=getattr(cfg,'head_dim',None) or cfg.hidden_size//NH", "head_dim=getattr(cfg,'head_dim',None) or cfg.hidden_size//NH\n  assert head_dim==read('manifest/config_gate.json')['head_dim']\n  assert all(layer.self_attn.o_proj.in_features==NH*head_dim for layer in h.get_layers(model))")
   # Fail safely on any unjournaled partial output during resume; never silently trust it.
   s=s.replace("if p.exists():\n     data=np.load", "if p.exists():\n     if str(p.relative_to(WORK)) not in progress.get('committed_files',{}):raise EngineeringStop('Unjournaled direction file; manual audit required before resume')\n     data=np.load")
   s=s.replace("if dest.exists():rec=json.loads(dest.read_text());assert rec['pair']==i", "if dest.exists():\n    if str(dest.relative_to(WORK)) not in progress.get('committed_files',{}):raise EngineeringStop('Unjournaled pair output; manual audit required')\n    rec=json.loads(dest.read_text());assert rec['pair']==i")
   s=s.replace("if dest.exists():rec=json.loads(dest.read_text())", "if dest.exists():\n     if str(dest.relative_to(WORK)) not in progress.get('committed_files',{}):raise EngineeringStop('Unjournaled head output; manual audit required')\n     rec=json.loads(dest.read_text())")
   s=s.replace("def checkpoint(event):\n  progress.update", "def checkpoint(event):\n  progress.update")
  if name=='analyze_verify.py':
   s=s.replace('150*len(union)*32','150*len(union)*NH').replace('range(32)','range(NH)').replace('(37,4096)','(NL+1,DIM)').replace('[37,4096]','[NL+1,DIM]').replace('len(joined)==96',"len(joined)==len(layers['original_layers'])*NH")
  if name=='additional_audit.py':s=s.replace('(37,4096)','(NL+1,DIM)')
  if name=='package.py':
   s=s.replace("'تقرير Qwen لثبات اختيار الرؤوس — electronics-4637664'","'تقرير '+SLUG+' لثبات اختيار الرؤوس — electronics-4637664'")
   s=s.replace("state.get('reason',state.get('substatus','اكتملت جميع البوابات'))","state.get('reason') or state.get('substatus') or 'اكتملت جميع البوابات'")
   s=s.replace('حالة COMPLETE تخص Qwen وهذا التكليف فقط.','حالة COMPLETE تخص هذا النموذج وهذا التكليف فقط.').replace("ELECTRONICS_QWEN_HEAD_STABILITY_RETURN_{stamp}","ELECTRONICS_HEAD_STABILITY_{SLUG}_RETURN_{stamp}")
   s=s.replace("before=read('manifest/inputs_before.json');", "before=read('manifest/inputs_before.json');")
   s=s.replace("report=['تقرير", "report=['تقرير")
   s=s.replace("(WORK/'HEAD_STABILITY_REPORT_AR.txt').write_text", "if (WORK/'selection/overlap_audit.json').exists():report+=['عدد تغريدات اتجاه القراءة المشتركة مع مصادر البوادئ: '+str(read('selection/overlap_audit.json')['readout_prefix_source_overlap_n'])]\n if (WORK/'RUNTIME_NOTES_AR.txt').exists():report+=[(WORK/'RUNTIME_NOTES_AR.txt').read_text()]\n (WORK/'HEAD_STABILITY_REPORT_AR.txt').write_text")
   s=s.replace("print(str(dest),flush=True)","dest.with_suffix('.workdir.json').write_text(dumps({'workdir':str(WORK),'status':state['status'],'archive':str(dest)}));print(str(dest),flush=True)")
  ast.parse(s);(W/'code'/name).write_text(s)
  (W/'code'/(name+'.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(Q/name),tofile=str(W/'code'/name))))
 (W/'manifest/ADAPTATION.json').write_text(json.dumps({'source_code':str(Q),'source_code_hashes':H,'all_local_source_and_reference_hashes_match':True,'model':E,'created_utc':stamp,'changes':['model identity and pinned result paths','dynamic NL/DIM/NH and explicit head_dim/o_proj validation','model-specific counts and batch sizes','safe refusal to trust unjournaled resume files','model-specific reports/archive names; same original helpers and numerical calculations'],'no_Qwen_execution':True},indent=2)+'\n')
 (W/'STATUS.json').write_text(json.dumps({'status':'PENDING','model':E['model']},indent=2)+'\n')
 status['models'][slug]={'workdir':str(W),'status':'PENDING'}
(CONTROL/'HEAD_STABILITY_EXPANSION_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(CONTROL)
