import datetime,hashlib,json,os,socket,sys,tempfile
from pathlib import Path
ROOT=Path('/Users/fahad/Documents/Research-Projects/New-Paper');WORK=Path(__file__).resolve().parents[1]
TASK=ROOT/'Electronics_Four_Model_Controls/revision/protocols/qwen_head_stability'
EXPECTED=json.loads((WORK/'manifest/EXPECTED_INPUTS.json').read_text())
SNAP=ROOT/'.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots'/EXPECTED['revision']
R=EXPECTED['readout_layer'];NL=EXPECTED['n_layers'];NH=EXPECTED['num_attention_heads']
sys.dont_write_bytecode=True
(WORK/'logs/tmp').mkdir(exist_ok=True);tempfile.tempdir=str(WORK/'logs/tmp')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(WORK/'logs/tmp'))
if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in [None,'1']:raise RuntimeError('Original MPS fallback conflicts')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
def no_network(*a,**k):raise RuntimeError('Network forbidden; pinned local snapshot only')
socket.socket.connect=no_network;socket.create_connection=no_network
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def extra(x):
 if isinstance(x,set):return sorted(x)
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def dumps(x):return json.dumps(x,ensure_ascii=False,indent=2,default=extra)+'\n'
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':'),default=extra).encode()).hexdigest()
def save(rel,obj):
 p=WORK/rel;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(dumps(obj));os.replace(t,p)
def read(rel):return json.loads((WORK/rel).read_text())
def jsonl(rel,rows):
 with (WORK/rel).open('w') as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False,default=extra)+'\n')
def log(action,**kw):
 with (WORK/'logs/actions.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),action=action,command=sys.argv,**kw),ensure_ascii=False,default=extra)+'\n')
 print(action,kw.get('status',''),flush=True)
def inventory(paths):return {str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'resolved':str(p.resolve())} for p in sorted(set(paths))}
