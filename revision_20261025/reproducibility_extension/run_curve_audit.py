"""Portable wrapper for the frozen 2,601-item curve audit; no model inference."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,zipfile
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--archives',type=Path,default=Path(__file__).resolve().parent/'archives');a=p.parse_args()
root=a.out.resolve();archives=a.archives.resolve()
if root.exists():p.error('Output directory exists. Use a new directory; existing data are never overwritten.')
expected={
'ELECTRONICS_SELECTED_CURVE_RETURN_20261024_A_REVIEW_PART_001.zip':'a039cd21a6fccd872218061e3c31aa83d7aa4ff37aeb17bfef804725ba7a47e2',
'ELECTRONICS_SELECTED_CURVE_RETURN_20261024_A_REVIEW_PART_002.zip':'837b8dc050d536a686bdb20bf2b8de303aa949a79d9d2bd0b0566d654b6763b2',
'Electronics_Selected_Curve_20261024.zip':'3e7bdc0d012a7b1feca6ee9d4ed33cb3ebdc2936251cdebe74c2912167075692'}
for name,h in expected.items():
 f=archives/name
 if not f.is_file():p.error('Missing frozen input: '+str(f))
 if hashlib.sha256(f.read_bytes()).hexdigest()!=h:p.error('Input hash mismatch: '+name)
root.mkdir(parents=True);unpacked=root/'inputs';unpacked.mkdir()
for name in list(expected)[:2]:
 with zipfile.ZipFile(archives/name) as z:
  for info in z.infolist():
   target=(unpacked/info.filename).resolve()
   if unpacked.resolve() not in target.parents:raise ValueError('Unsafe archive path')
   if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
   content=z.read(info)
   if target.exists():
    if target.read_bytes()!=content:raise ValueError('Conflicting duplicate member')
   else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
subprocess.run([sys.executable,str(Path(__file__).with_name('reproduce_curves_raw.py')),'--return-root',str(unpacked/'ELECTRONICS_SELECTED_CURVE_RETURN_20261024_A'),'--issued-zip',str(archives/'Electronics_Selected_Curve_20261024.zip'),'--out',str(root/'analysis')],check=True)
(root/'INPUT_HASHES.json').write_text(json.dumps(expected,indent=2))
print('Verified output:',root/'analysis/INDEPENDENT_AUDIT.json')
