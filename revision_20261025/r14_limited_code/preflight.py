"""Read-only environment/source gate. No model loading, installation or download."""
import sys,json,shutil,argparse,ast
from pathlib import Path
from pilot_support import sha,require_hash
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);a=ap.parse_args();root=a.project.resolve()
if sys.flags.optimize:raise SystemExit('Do not run Python with -O/PYTHONOPTIMIZE')
if Path(sys.prefix).resolve()!=(root/'.venv').resolve():raise SystemExit('Use existing project .venv')
for rel,h in json.loads((P/'MANIFEST.json').read_text()).items():require_hash(P/rel,h)
import numpy as np,pandas,torch,transformers,scipy
if not torch.backends.mps.is_available():raise SystemExit('MPS unavailable')
versions={'numpy':np.__version__,'torch':torch.__version__,'transformers':transformers.__version__}
for p in (P/'models').iterdir():
 exp=json.loads((p/'RUNTIME.json').read_text())
 for key,v in versions.items():
  if exp[key]!=v:raise SystemExit(f'{key}: {v} != expected {exp[key]}')
spec=json.loads((P/'EXPECTED_INPUTS.json').read_text());seen=set()
for m in spec['models']:
 for rec in spec['common_inputs']+m['snapshot_files']+[{'path':m['probe_path'],'sha256':m['probe_sha256']}]:
  rel=rec.get('path',rec.get('relative_to_New_Paper'))
  if rel in seen:continue
  require_hash(root/rel,rec['sha256']);seen.add(rel)
if shutil.disk_usage(root).free<20*1024**3:raise SystemExit('Require at least 20 GiB free for checkpoint/output margin')
for f in P.glob('*.py'):ast.parse(f.read_text())
print(json.dumps({'status':'PREFLIGHT_PASS','source_files_checked':len(seen),'versions':versions,'reference_forwards':0,'model_forwards':0}),flush=True)
