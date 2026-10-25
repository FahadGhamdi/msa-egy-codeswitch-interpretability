"""Post-gate independent reporting: all rank differences and exact prefix repetition counts."""
from common import *
from collections import Counter
import numpy as np
import pandas as pd
import torch
assert read('STATUS.json')['status']=='COMPLETE'
old=pd.read_csv(WORK/'results/ORIGINAL_HEAD_RANKING.csv');new=pd.read_csv(WORK/'results/RESTORED_FULL_HEAD_RANKING.csv');old['rank_original']=np.arange(1,len(old)+1);new['rank_restored']=np.arange(1,len(new)+1)
comparison=old[['layer','head','rank_original','mean']].merge(new[['layer','head','rank_restored','delta']],on=['layer','head'],validate='one_to_one');comparison['rank_difference']=comparison.rank_restored-comparison.rank_original;comparison['mean_difference']=comparison.delta-comparison['mean'];comparison.to_csv(WORK/'results/ALL_HEAD_RANK_DIFFERENCES.csv',index=False)
pairs=[json.loads(s) for s in (WORK/'selection/pairs.jsonl').read_text().splitlines()][:150];pools=[json.loads(s) for s in (WORK/'selection/prefix_pool.jsonl').read_text().splitlines()];index={(x['variety'],x['pool_index']):x for x in pools};occurrences={}
for p in pairs:
 for v in ['EGY','MSA']:
  e=index[(v,p[v+'_pool_index'])];occurrences.setdefault(e['token_ids_sha256'],[]).append({'pair':p['pair'],'variety':v,'pool_index':e['pool_index'],'sid':e['sid']})
repeated=[{'prefix_sha256':k,'occurrences':v} for k,v in occurrences.items() if len(v)>1]
# Independently rebuild saved means from ordered committed per-tweet contributions, with no model.
samples=[json.loads(s) for s in (WORK/'selection/direction_sample.jsonl').read_text().splitlines()];counts={};mu_checks={}
for var in ['MSA','EGY']:
 acc=None;n=0
 for s in samples:
  if s['variety']!=var:continue
  with np.load(WORK/f"directions/unit_{s['unit']:03d}.npz",allow_pickle=False) as a:
   vec=a['sum'];assert vec.dtype==np.float32 and vec.shape==(NL+1,DIM) and np.isfinite(vec).all() and int(a['n_words'])==s['n_words']
  t=torch.from_numpy(vec);acc=t if acc is None else acc+t;n+=s['n_words']
 recomputed=(acc/n).numpy();stored=np.load(WORK/f'directions/mu_{var}.npy',allow_pickle=False);assert np.array_equal(recomputed,stored);mu_checks[var]={'count':n,'exact_saved_mu_match':True,'max_difference':float(np.abs(recomputed-stored).max())};counts[var]=n
out={'utc':now(),'all_head_rank_order_matches':bool((comparison.rank_difference==0).all()),'heads_with_rank_change':comparison[comparison.rank_difference!=0].to_dict('records'),'max_head_mean_difference':float(np.abs(comparison.mean_difference).max()),'prefix_occurrences_MSA_pairs':300,'distinct_prefix_token_sequences':len(occurrences),'repeated_selected_prefix_sequences':len(repeated),'repeated_selected_prefix_details':repeated,'MSA_target_repeated_types':{k:v for k,v in Counter(p['word'] for p in pairs).items() if v>1},'direction_mean_reconstruction':mu_checks,'no_new_inference':True,'timing_note':'This post-gate audit adds reports only and does not change frozen selection, thresholds, inference code or analysis plan.'}
save('manifest/ADDITIONAL_INDEPENDENT_AUDIT.json',out);log('additional_independent_audit',all_head_ranks_same=out['all_head_rank_order_matches'],duplicate_prefix_sequences=len(repeated),means_exact=True,code_sha256=sha(Path(__file__)))
print(dumps({k:v for k,v in out.items() if k!='repeated_selected_prefix_details'}))
