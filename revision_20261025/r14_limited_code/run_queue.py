"""One-worker queue. Automatic continuation only after clean time-budget pauses."""
import argparse,subprocess,sys,json,fcntl,datetime,zipfile,hashlib,time
from pathlib import Path
from checkpoint import atomic_json
from pilot_support import sha
P=Path(__file__).resolve().parent
MODELS=['Qwen3-8B-Base','ALLaM-7B-Instruct-preview','gemma-2-9b','Fanar-1-9B']
def next_action(code,status):
 if code==0 and status=='COMPLETE':return 'complete'
 if code==75 and status=='PAUSED_BUDGET':return 'resume'
 return 'stop'
def archive(root,model):
 src=root/model;dest=root/(model+'_REVIEW_CORE.zip')
 if dest.exists():raise RuntimeError('Refuse overwrite archive')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(src.rglob('*')):
   if f.is_file() and 'tmp' not in f.relative_to(src).parts:z.write(f,model+'/'+str(f.relative_to(src)))
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 return {'model':model,'file':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
 lock=(root/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.monotonic()
 state={'status':'RUNNING','completed_models':[],'archives':[],'scientific_review_pending':True};atomic_json(root/'QUEUE_STATE.json',state)
 try:
  for model in MODELS:
   out=root/model
   if out.exists():raise RuntimeError('Existing model output requires review; initial queue is not a general retry tool')
   for session in range(1,7):
    command=[sys.executable,str(P/'run_model.py'),'--project',str(a.project.resolve()),'--model',model,'--out',str(out)]
    if session>1:command+=['--resume']
    print(json.dumps({'event':'start_session','model':model,'session':session}),flush=True)
    try:
     remaining=36*3600-(time.monotonic()-started)
     if remaining<=0:raise RuntimeError('36-hour total queue limit; review remaining work')
     r=subprocess.run(command,timeout=min(22000,remaining))
    except subprocess.TimeoutExpired:raise RuntimeError('Session watchdog fired; no automatic retry')
    status=json.loads((out/'STATE.json').read_text()).get('status') if (out/'STATE.json').exists() else 'NO_STATE'
    action=next_action(r.returncode,status)
    if action=='stop':raise RuntimeError(f'{model}: code={r.returncode}, status={status}; no retry')
    if action=='complete':break
   else:raise RuntimeError('Session limit reached; review needed')
   subprocess.run([sys.executable,str(P/'analyze.py'),'--input',str(out)],check=True)
   state['completed_models'].append(model);state['archives'].append(archive(root,model));atomic_json(root/'QUEUE_STATE.json',state)
  state['status']='COLLECTION_AND_LOCAL_ANALYSIS_COMPLETE_PENDING_INDEPENDENT_REVIEW'
 except BaseException as exc:
  state['status']='STOPPED_PENDING_REVIEW';state['error']=repr(exc)
  # Preserve a return archive for a stopped model too. No restart of failed work.
  if 'model' in locals() and (root/model).exists() and not (root/(model+'_REVIEW_CORE.zip')).exists():
   try:state['archives'].append(archive(root,model))
   except Exception as err:state['archive_error']=repr(err)
 finally:
  atomic_json(root/'QUEUE_STATE.json',state);print(json.dumps(state),flush=True)
 return 0 if state['status'].startswith('COLLECTION_AND') else 1
if __name__=='__main__':raise SystemExit(main())
