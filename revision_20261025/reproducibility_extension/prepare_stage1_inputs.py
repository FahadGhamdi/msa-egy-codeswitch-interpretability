"""Verify and prepare previously received Stage1 archives. No model inference."""
from pathlib import Path
import argparse,json,hashlib,zipfile,io,shutil
p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=Path,default=Path(__file__).with_name('STAGE1_INPUT_INDEX.json'));a=p.parse_args()
if a.out.exists():p.error('Output exists; choose a new directory')
a.out.mkdir(parents=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
def extract(z,d):
 for n in z.namelist():
  f=(d/n).resolve()
  assert f.is_relative_to(d.resolve()) and not f.exists(),n
  if n.endswith('/'):f.mkdir(parents=True,exist_ok=True)
  else:f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(n))
index=read(a.index)
for r in index:
 matches=list(a.archives.rglob(r['name']));assert len(matches)==1,r['name']
 f=matches[0];assert f.stat().st_size==r['bytes'] and sha(f.read_bytes())==r['sha256'],f
 if 'COMPLETION' in f.name:
  with zipfile.ZipFile(f) as z:extract(z,a.out/'completion')
 else:
  with zipfile.ZipFile(f) as z:b=z.read('ALLAM/ELECTRONICS_ALLAM_NIGHT_20261023_REVIEW.zip')
  with zipfile.ZipFile(io.BytesIO(b)) as z:b=z.read('ELECTRONICS_ALLAM_FP32_STAGE1_20261023_NIGHT_REVIEW_CORE_SESSION_02.zip')
  with zipfile.ZipFile(io.BytesIO(b)) as z:extract(z,a.out/'allam')
root=a.out/'completion';manifest=read(root/'RETURN_MANIFEST.json');assert len(manifest)==2932
for r in manifest:
 f=root/r['path'];assert f.stat().st_size==r['bytes'] and sha(f.read_bytes())==r['sha256']
assert {r['path'] for r in manifest}=={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}-{'RETURN_MANIFEST.json'}
report={'archives_verified':len(index),'completion_payload_verified':len(manifest),'models':{}}
for model in ['gemma-2-9b','Fanar-1-9B']:
 first=root/model/'part01/RESULTS';dest=a.out/'merged'/model;dest.mkdir(parents=True);(dest/'items').mkdir();seen=set();parts=[]
 names=['ITEMS.json','DESIGN.json','TOKEN_LISTS.json','HISTORICAL_TARGET_ROWS.csv.gz','reference_means_norms.npz']
 for name in names:shutil.copy2(first/name,dest/name)
 for part in sorted((root/model).glob('part[0-9][0-9]')):
  src=part/'RESULTS';content=read(src/'RETURN_CONTENTS.json')['core']
  for r in content:
   f=src/r['path'];assert f.stat().st_size==r['bytes'] and sha(f.read_bytes())==r['sha256']
  for name in names:assert (src/name).read_bytes()==(first/name).read_bytes()
  ids=read(src/'COMPLETED_ITEMS.json');state=read(src/'STATE.json');assert ids==read(src/'PLAN.json')['item_ids']==state['completed_items'];assert not seen.intersection(ids);seen.update(ids)
  for name,h in state['outputs'].items():assert sha((src/name).read_bytes())==h
  for i in ids:shutil.copy2(src/f'items/item_{i:04d}.npz',dest/f'items/item_{i:04d}.npz')
  parts.append({'part':part.name,'items':len(ids),'payload_verified':len(content)})
 assert seen==set(range(654));report['models'][model]={'items':654,'parts':parts,'input':str(dest.resolve())}
allam=list((a.out/'allam').rglob('ITEMS.json'));assert len(allam)==1
report['models']['ALLaM']={'input':str(allam[0].parent.resolve()),'items':len(read(allam[0]))}
(a.out/'PREPARATION_CHECKS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
