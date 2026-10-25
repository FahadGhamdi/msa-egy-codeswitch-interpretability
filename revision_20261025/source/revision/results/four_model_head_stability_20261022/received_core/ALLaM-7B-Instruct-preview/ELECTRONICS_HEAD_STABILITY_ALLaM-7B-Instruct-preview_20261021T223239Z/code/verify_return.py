"""Independent archive verification; no inference or changes to archive contents."""
from pathlib import Path
import datetime,hashlib,json,sys,zipfile
p=Path(sys.argv[1]);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names))
 manifest_raw=z.read('RETURN_MANIFEST.json');manifest=json.loads(manifest_raw);expected={x['path'] for x in manifest['files']}
 assert set(names)==expected|{'RETURN_MANIFEST.json'}
 for entry in manifest['files']:
  data=z.read(entry['path']);assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
  assert not Path(entry['path']).is_absolute() and '..' not in Path(entry['path']).parts
 result={'archive':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'status':manifest['status'],'crc_ok':True,'all_member_sha256_verified':True,'files_verified':len(expected),'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':sys.argv}
p.with_suffix('.verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');p.with_suffix('.sha256').write_text(result['sha256']+'  '+p.name+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
