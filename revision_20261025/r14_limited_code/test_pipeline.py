"""End-to-end CPU fixture: synthetic random controls equal saved baselines."""
import tempfile,json,hashlib,sys,contextlib,io
from pathlib import Path
import numpy as np
import analyze
from checkpoint import Store
from pilot_support import sha
P=Path(__file__).resolve().parent

def run_fixture():
 with tempfile.TemporaryDirectory() as tmp:
  t=Path(tmp);pkg=t/'pkg';model='Qwen3-8B-Base';m=pkg/'models'/model;m.mkdir(parents=True);(m/'anchors').mkdir();(m/'selected').mkdir();out=t/'return';out.mkdir()
  src=P/'models'/model;items=json.loads((src/'ITEMS.json').read_text());chosen=[next(i for i,x in enumerate(items) if x['group']==g) for g in analyze.GROUPS];new=[]
  for newi,oldi in enumerate(chosen):
   it=dict(items[oldi]);it['item']=newi;new.append(it)
   for sub in ['anchors','selected']:
    with np.load(src/sub/f'item_{oldi:04d}.npz') as z:arrays={k:z[k].copy() for k in z.files}
    if sub=='selected':arrays['item']=np.array(newi)
    np.savez_compressed(m/sub/f'item_{newi:04d}.npz',**arrays)
  (m/'ITEMS.json').write_text(json.dumps(new));(m/'DESIGN.json').write_bytes((src/'DESIGN.json').read_bytes())
  d={'active_batches':[0,1,2],'batches':[{'id':i,'k':k,'condition_ids':[i]} for i,k in enumerate([1,10,20])],'logical_to_unique':{f'{k}:{draw}':i for i,k in enumerate([1,10,20]) for draw in range(100)}}
  (m/'LIMITED_DESIGN.json').write_text(json.dumps(d));(pkg/'MANIFEST.json').write_text(json.dumps({str(f.relative_to(pkg)):sha(f) for f in pkg.rglob('*') if f.is_file()}))
  db=Store(out/'RESULTS.sqlite',{'package':sha(pkg/'MANIFEST.json'),'model':model})
  for i,it in enumerate(new):
   with np.load(m/'anchors'/f'item_{i:04d}.npz') as z:a={k:z[k].copy() for k in analyze.ARRAYS}
   for task in [{'id':-1,'k':0,'condition_ids':[]}]+d['batches']:
    arrays={k:np.repeat(v[:1],8,axis=0) for k,v in a.items()}
    if task['id']==-1:
     for k in analyze.ARRAYS:arrays[k][1]=a[k][1]
    arrays.update(item=np.array(i),k=np.array(task['k']),condition_ids=np.array(task['condition_ids'],dtype=np.int64))
    db.put(i,task['id'],arrays,{'condition_ids':task['condition_ids'],'baseline_checks_passed':True,'baseline_flip_identical':True},np)
  db.set('status','COMPLETE');db.close();(out/'STATE.json').write_text(json.dumps({'status':'COMPLETE','model':model}));(out/'OUTPUT_MANIFEST.json').write_text(json.dumps({f.name:sha(f) for f in out.iterdir() if f.is_file()}))
  oldP=analyze.P;argv=sys.argv
  try:
   analyze.P=pkg;sys.argv=['analyze.py','--input',str(out)]
   with contextlib.redirect_stdout(io.StringIO()):analyze.main()
  finally:analyze.P=oldP;sys.argv=argv
  with np.load(out/'ANALYSIS/PAIRED_VALUES.npz') as z:
   np.testing.assert_allclose(z['random_mean'],np.repeat(z['baseline'][:,None,:],3,axis=1),equal_nan=True)
  audit=json.loads((out/'ANALYSIS/AUDIT.json').read_text());assert audit['batches_checked']==16 and audit['independent_review_pending']
  return True
if __name__=='__main__':print('PIPELINE_FIXTURE_PASS',run_fixture())
