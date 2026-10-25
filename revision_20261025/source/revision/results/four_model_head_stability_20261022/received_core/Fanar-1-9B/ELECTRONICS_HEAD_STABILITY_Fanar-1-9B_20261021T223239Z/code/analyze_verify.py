from common import *
import ast,collections,itertools,traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():
 state=read('STATUS.json')
 if state['status']!='INFERENCE_COMPLETE_PENDING_ANALYSIS':log('analysis_skipped',status=state['status'],reason='Required inference gates not passed');return
 for n in ['EARLY_GATE','FULL_GATE','P3_GATE','LAYER_RANKING_GATE']:assert read('results/'+n+'.json')['passed']
 frozen=read('manifest/PROTOCOL_FROZEN.json')
 for p,d in frozen['files'].items():assert sha(WORK/p)==d
 # Independently verify grouping from the frozen raw prefix provenance.
 pairs=[json.loads(s) for s in (WORK/'selection/pairs.jsonl').read_text().splitlines()][:150];pool=[json.loads(s) for s in (WORK/'selection/prefix_pool.jsonl').read_text().splitlines()];by={(p['variety'],p['pool_index']):p for p in pool}
 for p in pairs:
  entries=[by[(v,p[v+'_pool_index'])] for v in ['EGY','MSA']];assert p['prefix_sids']==sorted(set(s for e in entries for s in e['alias_sids']));assert p['prefix_hashes']==sorted(set(e['token_ids_sha256'] for e in entries))
  for e in entries:assert digest(e['token_ids'])==e['token_ids_sha256'] and len(e['token_ids'])==12
 groups=read('selection/components.json')['group_by_pair'];folds=read('selection/FOLDS.json')['folds']
 for k,(tr,te) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=13).split(np.zeros((150,1)),groups=groups)):
  f=folds[k];assert tr.tolist()==f['train']['pair_ids'] and te.tolist()==f['test']['pair_ids'] and len(te)>=10
  for field in ['prefix_sids','prefix_hashes']:
   assert not set(x for i in tr for x in pairs[i][field])&set(x for i in te for x in pairs[i][field])
 # AST identity with local source ensures original mathematics and patch hooks were not changed.
 orig=ast.parse((ROOT/'exp02_causal_patching.py').read_text());copied=ast.parse((WORK/'code/original_helpers.py').read_text())
 for e in read('manifest/helper_origins.json'):
  a=next(n for n in orig.body if isinstance(n,ast.FunctionDef) and n.name==e['name']);b=next(n for n in copied.body if isinstance(n,ast.FunctionDef) and n.name==e['name']);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 raw=pd.read_csv(WORK/'results/HEAD_DELTAS.csv');layers=read('selection/LAYER_SELECTION.json');union=layers['union'];assert len(raw)==150*len(union)*NH and not raw.duplicated(['pair','layer','head']).any()
 assert set(map(tuple,raw[['pair','layer','head']].to_numpy()))==set(itertools.product(range(150),union,range(NH)))
 assert np.isfinite(raw[['s_corrupt','s_patched','delta']].to_numpy()).all();assert np.allclose(raw.delta,raw.s_patched-raw.s_corrupt,atol=1e-12,rtol=0)
 for var in ['MSA','EGY']:
  mu=np.load(WORK/f'directions/mu_{var}.npy',allow_pickle=False);assert mu.shape==(NL+1,DIM) and mu.dtype==np.float32 and np.isfinite(mu).all()
 # Verify serialized deltas independently reproduce all original P3 aggregates at fixed tolerance.
 orighead=pd.read_csv(EXPDIR/'P3_heads.csv');agg=raw.groupby(['layer','head']).delta.agg(['mean','std','count']).reset_index();joined=orighead.merge(agg,on=['layer','head'],suffixes=('_old','_new'),validate='one_to_one');assert len(joined)==len(layers['original_layers'])*NH
 for metric in ['mean','std']:assert np.all(np.abs(joined[metric+'_new']-joined[metric+'_old'])<=1e-4+1e-4*np.abs(joined[metric+'_old']))
 assert (joined.count_old==joined.count_new).all() and (joined.count_new==150).all()
 def ranking(df):return df.groupby(['layer','head']).delta.mean().reset_index().sort_values(['delta','layer','head'],ascending=[False,True,True]).reset_index(drop=True)
 def keys(df):return set((int(r.layer),int(r['head'])) for _,r in df.iterrows())
 def jacc(a,b):return len(a&b)/len(a|b)
 original_ranking=orighead.sort_values(['mean','layer','head'],ascending=[False,True,True]);original_top=keys(original_ranking.head(5));original_ranking.to_csv(WORK/'results/ORIGINAL_HEAD_RANKING.csv',index=False)
 restored=ranking(raw[raw.layer.isin(layers['original_layers'])]);restored.to_csv(WORK/'results/RESTORED_FULL_HEAD_RANKING.csv',index=False)
 summaries=[];selected_sets=[];layer_freq=collections.Counter();head_freq=collections.Counter();heldout=[]
 for f,ls in zip(folds,layers['folds']):
  k=f['fold'];candidate=ls['top_layers'];train=raw[raw.pair.isin(f['train']['pair_ids'])&raw.layer.isin(candidate)];test=raw[raw.pair.isin(f['test']['pair_ids'])&raw.layer.isin(candidate)];r=ranking(train);r['rank']=np.arange(1,len(r)+1);r.to_csv(WORK/f'results/fold_{k}_training_ranking.csv',index=False)
  selected=r.head(5);selected_keys=keys(selected);selected_sets.append(selected_keys);layer_freq.update(candidate);head_freq.update(selected_keys);selected_summary=[]
  for a in selected.to_dict('records'):
   d=test[(test.layer==a['layer'])&(test['head']==a['head'])];selected_summary.append({'layer':int(a['layer']),'head':int(a['head']),'train_mean_delta':a['delta'],'test_mean_delta':float(d.delta.mean()),'test_sd':float(d.delta.std(ddof=1)),'test_n':len(d)})
   for x in d.to_dict('records'):heldout.append(dict(fold=k,**x))
  chosen=pd.DataFrame([x for x in heldout if x['fold']==k]);assert len(chosen)==5*len(f['test']['pair_ids'])
  summaries.append({'fold':k,'top3_layers':candidate,'top5_heads':selected_summary,'train_pairs':len(f['train']['pair_ids']),'test_pairs':len(f['test']['pair_ids']),'train_components':len(f['train']['component_ids']),'test_components':len(f['test']['component_ids']),'original_top5_overlap_n':len(selected_keys&original_top),'original_top5_overlap_fraction':len(selected_keys&original_top)/5,'original_top5_jaccard':jacc(selected_keys,original_top),'test_mean_of_individual_selected_head_deltas':float(chosen.delta.mean()),'not_joint_ablation':True})
 pd.DataFrame(heldout).to_csv(WORK/'results/SELECTED_HEAD_TEST_DELTAS.csv',index=False)
 frequencies=[{'layer':L,'head':hd,'selected_folds':head_freq[(L,hd)],'unconditional_frequency':head_freq[(L,hd)]/5,'eligible_folds':layer_freq[L],'conditional_frequency_given_eligible':head_freq[(L,hd)]/layer_freq[L] if layer_freq[L] else None} for L,hd in itertools.product(union,range(NH))]
 pd.DataFrame(frequencies).to_csv(WORK/'results/HEAD_SELECTION_FREQUENCIES.csv',index=False)
 pairwise=[{'fold_a':a,'fold_b':b,'intersection':len(selected_sets[a]&selected_sets[b]),'jaccard':jacc(selected_sets[a],selected_sets[b])} for a,b in itertools.combinations(range(5),2)]
 save('results/STABILITY_ANALYSIS.json',{'folds':summaries,'layer_frequencies':[{'layer':L,'selected_folds':layer_freq[L],'frequency':layer_freq[L]/5} for L in union],'pairwise_fold_jaccards':pairwise,'original_vs_restored_top5_same':keys(restored.head(5))==original_top,'original_top5':sorted(original_top),'restored_top5':sorted(keys(restored.head(5))),'interpretation':'descriptive internal ranking sensitivity; single-head patch effects, no joint ablation, p-values, independent confirmation, causal mediation or new next-token outcome'})
 save('manifest/INDEPENDENT_VALIDATION.json',{'all_checks_passed':True,'raw_head_rows':len(raw),'shape_and_finiteness':True,'no_prefix_or_alias_sid_leakage':True,'folds_exact_regeneration':True,'pure_helpers_exact_AST':True,'P3_reaggregation_from_serialized_raw_matches':True,'mu_shape':[NL+1,DIM],'mu_dtype':'float32','original_vs_restored_top5_same':keys(restored.head(5))==original_top})
 state.update(status='COMPLETE',analysis_folds=5,head_rows=len(raw),head_layers=union,verified_utc=now());save('STATUS.json',state);log('analysis_validation_complete',status='COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception as e:
  (WORK/'logs/analysis_failure.txt').write_text(traceback.format_exc());state=read('STATUS.json');state.update(status='STOPPED',reason='Analysis/verification failure: '+str(e));save('STATUS.json',state);log('analysis_failure',status='STOPPED',reason=str(e));raise
