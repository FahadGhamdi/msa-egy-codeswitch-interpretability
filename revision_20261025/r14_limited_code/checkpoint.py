"""Transactional batch checkpoints. Only planned pauses may resume automatically."""
import sqlite3,json,hashlib,io,os
from pathlib import Path
class Store:
 def __init__(self,path,identity,resume=False):
  path=Path(path);exists=path.exists()
  if exists and not resume:raise RuntimeError('Existing checkpoint requires --resume')
  if resume and not exists:raise RuntimeError('Missing resume checkpoint')
  self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
  self.db.execute('CREATE TABLE IF NOT EXISTS batches (item INTEGER,batch INTEGER,payload BLOB,digest TEXT,report TEXT,PRIMARY KEY(item,batch))')
  if not exists:self.set('identity',identity);self.set('status','NEW')
  if self.get('identity')!=identity:raise RuntimeError('Task/input/package identity changed')
  if self.get('status') not in ('NEW','PAUSED_BUDGET','COMPLETE'):raise RuntimeError('Previous run was not a clean planned pause; review required')
  if self.db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('Checkpoint integrity failure')
  for payload,digest in self.db.execute('SELECT payload,digest FROM batches'):
   if hashlib.sha256(payload).hexdigest()!=digest:raise RuntimeError('Checkpoint payload changed')
 def get(self,key):
  r=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(r[0]) if r else None
 def set(self,key,value):
  with self.db:self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value,allow_nan=False)))
 def has(self,item,batch):return self.db.execute('SELECT 1 FROM batches WHERE item=? AND batch=?',(item,batch)).fetchone() is not None
 def put(self,item,batch,arrays,report,np):
  if self.has(item,batch):raise RuntimeError('Refuse duplicate batch')
  buf=io.BytesIO();np.savez_compressed(buf,**arrays);payload=buf.getvalue()
  with self.db:self.db.execute('INSERT INTO batches VALUES (?,?,?,?,?)',(item,batch,payload,hashlib.sha256(payload).hexdigest(),json.dumps(report,allow_nan=False)))
 def count(self):return self.db.execute('SELECT count(*) FROM batches').fetchone()[0]
 def close(self):self.db.close()
def atomic_json(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
