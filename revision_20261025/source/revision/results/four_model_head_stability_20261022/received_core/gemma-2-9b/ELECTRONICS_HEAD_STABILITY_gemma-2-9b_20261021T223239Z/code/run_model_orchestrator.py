"""Run each authorized model to its gates and verified return, strictly in declared order."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,traceback
CONTROL=Path(__file__).resolve().parents[1];ROOT=CONTROL.parent;STATUS=CONTROL/'HEAD_STABILITY_EXPANSION_STATUS.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
allstatus=json.loads(STATUS.read_text());slug=sys.argv[1];assert slug in allstatus['models'];entry=allstatus['models'][slug];WORK=Path(entry['workdir'])
assert entry['status']=='PENDING','Already attempted: inspect state before any resume'
for prior in list(allstatus['models'])[:list(allstatus['models']).index(slug)]:assert allstatus['models'][prior]['status'] in ['COMPLETE','STOPPED','PARTIAL']
import torch
assert torch.backends.mps.is_available(),'MPS native permission required; no inference attempted and no CPU substitution'
(CONTROL/'logs').mkdir(exist_ok=True)
def log(action,**kw):
 row=dict(utc=now(),model=slug,action=action,command=sys.argv,**kw)
 with (CONTROL/'logs/actions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(action,kw.get('stage',kw.get('status','')),flush=True)
def stage(name,inside=True,args=None):
 script=WORK/'code'/f'{name}.py';cmd=[str(ROOT/'.venv/bin/python'),'-B',str(script)]+(args or []);dest=WORK/'logs' if inside else CONTROL/'logs';prefix=name if inside else slug+'_'+name;attempt=len(list(dest.glob(prefix+'_*.stdout.txt')))+1;out=dest/f'{prefix}_{attempt:03d}.stdout.txt';err=dest/f'{prefix}_{attempt:03d}.stderr.txt';start=now()
 log('stage_start',stage=name,subprocess_command=cmd,script_sha256=sha(script),stdout=str(out),stderr=str(err))
 with out.open('x') as fo,err.open('x') as fe:p=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=fo,stderr=fe)
 row=dict(stage=name,start=start,end=now(),subprocess_command=cmd,returncode=p.returncode,stdout=str(out),stderr=str(err),stdout_sha256=sha(out),stderr_sha256=sha(err))
 log('stage_end',**row)
 if inside:
  with (WORK/'logs/stage_receipts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 return p.returncode,out
entry.update(status='RUNNING',started_utc=now());dump(STATUS,allstatus)
shutil.copy2(__file__,WORK/'code/run_model_orchestrator.py');shutil.copy2(CONTROL/'code/setup.py',WORK/'code/setup_adapter.py')
try:
 rc,_=stage('prepare');s=json.loads((WORK/'STATUS.json').read_text())
 if not rc and s['status']=='PREPARED':
  rc,_=stage('verify_selection_freeze')
  if not rc:
   rc,_=stage('infer')
   s=json.loads((WORK/'STATUS.json').read_text())
   if not rc and s['status']=='INFERENCE_COMPLETE_PENDING_ANALYSIS':
    rc,_=stage('analyze_verify');s=json.loads((WORK/'STATUS.json').read_text())
    if not rc and s['status']=='COMPLETE':
     rc,_=stage('additional_audit')
     if rc:raise RuntimeError('Independent extra audit failed')
  else:raise RuntimeError('Selection verification/freeze failed')
 elif rc:raise RuntimeError('Preparation failed; see logs')
except Exception as e:
 s=json.loads((WORK/'STATUS.json').read_text());s.update(status='STOPPED',orchestration_reason=str(e));dump(WORK/'STATUS.json',s);(WORK/'logs/orchestration_failure.txt').write_text(traceback.format_exc())
# Child has fully exited; all model allocations belong to that terminated process.
torch.mps.empty_cache()
warn=[]
for p in (WORK/'logs').glob('infer_*.stderr.txt'):
 for line in p.read_text().splitlines():
  if 'tied weights mapping' in line:warn.append(line)
if warn:
 dump(WORK/'manifest/LOADER_WARNINGS.json',{'warnings':warn,'action':'Default loader behavior preserved; no config/weight/backend change; full stderr retained'})
 (WORK/'RUNTIME_NOTES_AR.txt').write_text('تنبيه التحميل موثق في stderr: config يشير إلى ربط embedding وlm_head لكن checkpoint يحفظ قيمًا مختلفة؛ أبقى المحمل الوزنين منفصلين دون تعديل config أو الأوزان أو البيئة. نجحت أو أخفقت البوابات كما تسجل ملفاتها؛ لم يغيّر التنبيه العتبات.\n')
# If an input identity failure prevented the full inventory, still return an explicit stopped audit.
if not (WORK/'manifest/inputs_before.json').exists():
 (WORK/'manifest/inputs_before.json').write_text('{}\n');(WORK/'manifest/INVENTORY_LIMITATION.txt').write_text('Input gate failed before complete inventory; absence of inventory is not proof of preserved source identity.\n')
rc,out=stage('package',inside=False)
if rc:raise RuntimeError('Packaging failure; preserve outputs and inspect')
archive=Path(out.read_text().strip().splitlines()[-1]);assert archive.is_file()
rc,_=stage('verify_return',inside=False,args=[str(archive)])
if rc:raise RuntimeError('Archive independent CRC/SHA verification failed')
verification=json.loads(archive.with_suffix('.verification.json').read_text());s=json.loads((WORK/'STATUS.json').read_text());entry.update(s);entry.update(archive=str(archive),archive_verification=verification,finished_utc=now(),workdir=str(WORK))
# Mandatory splitting only if the verified ZIP exceeds 450 MiB; exact byte-preserving parts.
if archive.stat().st_size>450*1024*1024:
 parts=[];full=hashlib.sha256()
 with archive.open('rb') as f:
  i=1
  while True:
   data=f.read(450*1024*1024)
   if not data:break
   part=archive.with_name(archive.name+f'.part{i:03d}');part.write_bytes(data);parts.append({'name':part.name,'bytes':len(data),'sha256':sha(part)});full.update(part.read_bytes());i+=1
 assert full.hexdigest()==verification['sha256']
 pm=archive.with_suffix('.parts.json');dump(pm,{'archive_name':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':verification['sha256'],'parts_in_order':parts,'reconstruction':'Concatenate exact parts in listed order, verify each part SHA256 then reconstructed archive SHA256, then use verify_return.py for CRC/member hashes; original verified ZIP is retained.'})
 restore=archive.with_suffix('.reassemble.py');restore.write_text("import pathlib,json,hashlib,sys\np=pathlib.Path(sys.argv[1]);m=json.loads(p.read_text());out=p.parent/('REASSEMBLED_'+m['archive_name']);h=hashlib.sha256()\nwith out.open('xb') as f:\n for e in m['parts_in_order']:\n  b=(p.parent/e['name']).read_bytes();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'];f.write(b);h.update(b)\nassert out.stat().st_size==m['archive_bytes'] and h.hexdigest()==m['archive_sha256']\nprint(out)\n")
 entry.update(parts_manifest=str(pm),parts=[str(archive.parent/x['name']) for x in parts],reassembly_script=str(restore));log('zip_split_verified',parts=len(parts),full_sha256_matches=True)
allstatus['updated_utc']=now();dump(STATUS,allstatus);log('model_finished',status=s['status'],archive=str(archive),bytes=verification['bytes']);print(json.dumps(entry,ensure_ascii=False,indent=2),flush=True)
