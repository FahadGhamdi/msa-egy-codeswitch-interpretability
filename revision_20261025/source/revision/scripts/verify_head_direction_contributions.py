"""CPU-only audit of a complete model's transferred per-tweet contributions."""
from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
V=Path(__file__).resolve().parents[1];R=V.parents[1];slug=sys.argv[1]
D=R/'local_evidence_received/three_model_head_core'/slug;A=next(p for p in D.iterdir() if p.is_dir());J=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
transfer=J(R/'upload/TRANSFER_INDEX.json')['models'][slug];expected=J(A/'manifest/EXPECTED_INPUTS.json');records=[json.loads(s) for s in (A/'selection/direction_sample.jsonl').read_text().splitlines()]
assert len(records)==800 and {r['unit'] for r in records}==set(range(800))
# Validate the complete transferred original payload against both recorded manifests.
manifest=J(D/'RETURN_MANIFEST.json');original={e['path']:e for e in manifest['files']}
for e in transfer['files']:
 p=D/e['path'];assert p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
 if e['path']!='RETURN_MANIFEST.json':assert original[e['path']]['sha256']==e['sha256']
assert len(transfer['files'])==len(original)+1
assert {p.name for p in (A/'directions').glob('unit_*.npz')}=={f'unit_{i:03d}.npz' for i in range(800)}
results={}
for variety in ['MSA','EGY']:
 samples=[r for r in records if r['variety']==variety];assert len(samples)==400
 acc=None;n=0
 for r in samples:
  with np.load(A/f"directions/unit_{r['unit']:03d}.npz",allow_pickle=False) as z:
   v=z['sum'];count=int(z['n_words'])
   assert v.shape==(expected['n_layers']+1,expected['hidden_size']) and v.dtype==np.float32 and np.isfinite(v).all()
   assert count==r['n_words']==len(r['positions']);n+=count
   acc=v.copy() if acc is None else acc+v
 calculated=acc/np.float32(n);stored=np.load(A/f'directions/mu_{variety}.npy',allow_pickle=False)
 assert stored.dtype==np.float32 and np.isfinite(stored).all()
 delta=float(np.max(np.abs(calculated-stored)));exact=calculated.tobytes()==stored.tobytes();assert exact
 results[variety]={'tweets':400,'word_positions':n,'shape':list(stored.shape),'dtype':'float32','max_absolute_error':delta,'bitwise_equal_arrays':exact}
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':slug,'status':'DIRECTION_RECONSTRUCTION_PASS','complete_transferred_original_members_verified':len(transfer['files']),'per_tweet_contributions_verified':800,'reconstructed_means':results,'new_model_inference':False,'scope':'Completes deferred direction reconstruction for this model; internal sensitivity analysis only, not independent-corpus confirmation. Original ZIP container was not received; complete transferred member contents verified.'}
out=V/'results/three_model_head_core_20261022'/slug/'DIRECTION_RECONSTRUCTION_AUDIT.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
