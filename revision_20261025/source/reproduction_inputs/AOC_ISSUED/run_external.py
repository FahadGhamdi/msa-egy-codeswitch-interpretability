"""Frozen AOC readout collection; local MPS FP32 only, no training or retries."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import argparse,hashlib,json,time,sys,zipfile,gc
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,r):
    temp=p.with_name(p.name+'.tmp')
    with temp.open('w') as f:
        json.dump(r,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)

def check_items(items):
    assert len({x['item'] for x in items})==len(items)
    assert len({(x['comment_id'],x['word_index']) for x in items})==len(items)
    for x in items:
        assert x['t']==len(x['ids'])-1 and x['ids'] and all(type(i)==int and i>=0 for i in x['ids'])
        assert x['own'] in ['MSA','EGY'] and x['comment_id']==x['utterance']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.project.resolve();out=a.out.resolve();pkg=Path(__file__).resolve().parent
    assert Path(sys.prefix).resolve()==root/'.venv','Use project .venv'
    assert root in out.parents and not out.exists() and not out.with_suffix('.zip').exists(),'New output required; do not delete or retry'
    for rel,h in json.loads((pkg/'MANIFEST.json').read_text()).items():assert sha(pkg/rel)==h,rel
    spec=json.loads((pkg/'EXPECTED_INPUTS.json').read_text());allitems={}
    for m in spec['models']:
        items=json.loads((pkg/(m['slug']+'_ITEMS.json')).read_text());check_items(items);allitems[m['slug']]=items
    # Check ALL models, probes and relevant input bytes before first model output.
    checks=[]
    records=[(r.get('relative_to_New_Paper',r.get('path')),r['sha256']) for r in spec['common_inputs']]
    for m in spec['models']:
        records += [(m['probe_path'],m['probe_sha256'])]+[(r['path'],r['sha256']) for r in m['snapshot_files']]
    for rel,h in dict(records).items():
        actual=sha(root/rel);assert actual==h,rel;checks.append({'path':rel,'sha256':actual})
    import torch,numpy as np,transformers
    from transformers import AutoModelForCausalLM
    assert torch.backends.mps.is_available(),'MPS required; no CPU fallback'
    out.mkdir();save(out/'INPUT_CHECKS.json',checks)
    save(out/'ENVIRONMENT.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'mps','dtype':'float32','attention':'sdpa','batch_size':1,'probe_projection':'CPU float32 sum(h*w)+b','source_grouping':'comment; author/article unknown'})
    shutil_files=['PROTOCOL.txt','FROZEN_COHORTS.json','MANIFEST.json']
    for name in shutil_files:(out/('FROZEN_'+name)).write_bytes((pkg/name).read_bytes())
    started=time.monotonic();deadline=started+6*3600;total=0
    try:
        torch.mps.set_per_process_memory_fraction(.8)
        for m in spec['models']:
            slug=m['slug'];items=allitems[slug];md=out/slug;md.mkdir();raw=md/'retained_features';raw.mkdir()
            snap=(root/m['snapshot_files'][0]['path']).parent
            probe=torch.load(root/m['probe_path'],map_location='cpu',weights_only=True)
            assert int(probe['layer'])==m['layer']
            w=probe['w'].detach().float().cpu().reshape(-1);b=float(probe['b'])
            assert torch.isfinite(w).all() and np.isfinite(b)
            model=AutoModelForCausalLM.from_pretrained(str(snap),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('mps').eval()
            assert model.config.num_hidden_layers==m['num_hidden_layers'] and model.config.hidden_size==len(w)
            assert next(model.parameters()).dtype==torch.float32
            durations=[]
            for index,it in enumerate(items):
                assert time.monotonic()<deadline,'Six-hour budget reached; preserve checkpoints'
                x=torch.tensor([it['ids']],dtype=torch.long,device='mps');torch.mps.synchronize();t0=time.monotonic()
                with torch.inference_mode():
                    result=model(input_ids=x,output_hidden_states=True,use_cache=False,logits_to_keep=1)
                    h=result.hidden_states[m['layer']][0,it['t']].detach().float().cpu().contiguous()
                torch.mps.synchronize();elapsed=time.monotonic()-t0;durations.append(elapsed)
                assert torch.isfinite(h).all(),'Nonfinite hidden vector; stop'
                z=float((h*w).sum()+b);own=z if it['own']=='EGY' else -z
                assert np.isfinite(own)
                fp=raw/(str(it['item'])+'.npy');temp=fp.with_suffix('.npy.tmp')
                with temp.open('wb') as f:np.save(f,h.numpy(),allow_pickle=False);f.flush();os.fsync(f.fileno())
                temp.replace(fp)
                record={**it,'egy_logit':z,'own_logit':own,'flipped':own<0,'feature_sha256':sha(fp),'feature_shape':list(h.shape),'seconds':elapsed}
                save(md/(str(it['item'])+'.json'),record);total+=1
                del result,h,x
                if index==3:
                    forecast=max(durations)*1.5*(len(items)-4);remaining=deadline-time.monotonic()
                    save(md/'FORECAST.json',{'seconds_remaining_this_model':forecast,'budget_remaining':remaining,'basis':'max first four longest-per-group timings times 1.5; not guarantee or estimate for later models'})
                    print(json.dumps({'model':slug,'forecast_minutes':forecast/60,'remaining_budget_minutes':remaining/60}),flush=True)
                    assert forecast<=remaining,'Forecast exceeds remaining budget; stop for review'
                if index%100==0:print(json.dumps({'model':slug,'completed':index+1,'total':len(items)}),flush=True)
            save(md/'COMPLETE.json',{'items':len(items),'collection_seconds':sum(durations),'raw_features_retained_locally':True})
            del model,probe,w;gc.collect();torch.mps.empty_cache()
        status={'status':'COLLECTION_COMPLETE_PENDING_INDEPENDENT_ANALYSIS','forwards':total,'reference_forwards':0,'training':False,'R1_2_closed':False}
    except Exception as e:
        save(out/'STOPPED.json',{'error':repr(e),'completed_forwards':total,'automatic_retry':False,'resume_authorized':False})
        raise
    save(out/'STATUS.json',status)
    # Core excludes feature arrays; each array is hashed in its corresponding item record.
    core=[p for p in sorted(out.rglob('*')) if p.is_file() and 'retained_features' not in p.parts]
    save(out/'RETURN_MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in core})
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in core+[out/'RETURN_MANIFEST.json']:z.write(p,out.name+'/'+str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'sha256':sha(archive),'status':status}),flush=True)

if __name__=='__main__':main()
