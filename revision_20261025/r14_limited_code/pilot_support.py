"""Pure numerical checks for the declared ALLaM FP32 pilot."""
import hashlib,json,re
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def parse(h):return tuple(map(int,re.fullmatch(r'L(\d+)H(\d+)',h).groups()))
def require_hash(p,h):
 if sha(p)!=h:raise RuntimeError('Hash mismatch: '+str(p))
def metrics_from_tokens(it,own,tokenlp,np):
 met=np.full((len(own),4),np.nan,np.float32);met[:,0]=own
 for j,role in enumerate(['all','own','matrix'],1):
  chunks=[tokenlp[:,a-it['t']-1:b-it['t']-1] for a,b,lab in it['cont'] if lab and (role=='all' or lab==it[role])]
  if chunks:met[:,j]=np.concatenate(chunks,axis=1).mean(1)
 return met
def summarize(records,np):
 first={(r['item'],r['condition']):r for r in records if r['mode']=='batch8_first'}
 rows=[];deltas=[]
 for r in records:
  if r['mode'] not in ['batch8_repeat','batch2']:continue
  p=first[(r['item'],r['condition'])]
  a=np.array([np.nan if x is None else x for x in p['metrics']]);b=np.array([np.nan if x is None else x for x in r['metrics']]);ok=np.isclose(b,a,atol=1e-4,rtol=1e-4,equal_nan=True)
  rows.append({'item':r['item'],'condition':r['condition'],'mode':r['mode'],'metrics_within_tolerance':bool(ok.all()),'selected_scores_within_tolerance':r['selected_scores_within_tolerance'],'metric_abs_diff':[None if np.isnan(x) else float(x) for x in abs(b-a)],'full_logits_hash_identical':r['logits_sha256']==p['logits_sha256'],'full_logits_max_abs_diff':r['full_logits_max_abs_diff_vs_first']})
 for item in sorted({r['item'] for r in records if r['mode']=='batch8_first'}):
  for mode in ['batch8_repeat','batch2']:
   selected={r['condition']:r for r in records if r['item']==item and r['mode']==mode}
   if not {'baseline','top5'}<=selected.keys():continue
   old=np.array([np.nan if x is None else x for x in first[(item,'top5')]['metrics']])-np.array([np.nan if x is None else x for x in first[(item,'baseline')]['metrics']])
   new=np.array([np.nan if x is None else x for x in selected['top5']['metrics']])-np.array([np.nan if x is None else x for x in selected['baseline']['metrics']])
   deltas.append({'item':item,'mode':mode,'delta_abs_diff':[None if np.isnan(x) else float(x) for x in abs(new-old)],'within_tolerance':bool(np.isclose(new,old,atol=1e-4,rtol=1e-4,equal_nan=True).all()),'strict_sign_reversals':[bool(x) for x in old*new<0],'old_zero_new_nonzero':[bool(x) for x in (old==0)&(new!=0)]})
 return {'scope':'FP32 internal stability with newly computed FP32 means; not historical reproduction','atol':1e-4,'rtol':1e-4,'comparisons':rows,'delta_comparisons':deltas,'all_metric_checks_pass':all(r['metrics_within_tolerance'] and r['selected_scores_within_tolerance'] for r in rows),'all_delta_checks_pass':all(r['within_tolerance'] for r in deltas),'sign_reversals':sum(sum(r['strict_sign_reversals']) for r in deltas)}
