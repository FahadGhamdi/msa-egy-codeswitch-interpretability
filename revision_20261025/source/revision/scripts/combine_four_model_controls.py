from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,datetime,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
V=Path(__file__).resolve().parents[1];R=V.parents[1];O=V/'results/four_model_controls_20261022';O.mkdir(exist_ok=True)
spec=[('Qwen3-8B','qwen_features_analysis_20261022','qwen_features/ELECTRONICS_QWEN_FEATURES_20261021T204604Z'),('ALLaM-7B','ALLaM-7B-Instruct-preview_controls_20261022','three_model_features/ALLaM-7B-Instruct-preview'),('Gemma-2-9B','gemma-2-9b_controls_20261022','three_model_features/gemma-2-9b'),('Fanar-1-9B','Fanar-1-9B_controls_20261022','three_model_features/Fanar-1-9B')]
tables={n:[] for n in ['own_variety_metrics','own_variety_paired_differences','context_metrics','context_paired_differences','context_auroc','projection_diagnostics','variance_decomposition']};checks=[]
for model,d,features in spec:
 A=V/'results'/d;F=R/'local_evidence_received'/features/'features';P=pd.read_csv(A/'oof_predictions.csv');pars=np.load(A/'fitted_probe_parameters.npz',allow_pickle=False)
 X=np.load(F/'contextual.npy',allow_pickle=False);w=np.load(F/'saved_probe_w.npy',allow_pickle=False);component=np.full(len(P),np.nan)
 for k in range(5):
  for a in [0,1]:
   key=f'contextual_context_y{a}_f{k}';coef=(pars[key+'_coef'][0]/pars[key+'_sd']).astype(float);u=coef/np.linalg.norm(coef)
   ix=np.where((P.fold_id.to_numpy()==k)&(P.y.to_numpy()==a))[0]
   component[ix]=((X[ix].astype(float)-pars[key+'_mu'])@u)*np.dot(w,u)
 err=float(np.max(np.abs(component-P.context_component)));assert err<1e-10
 # pandas writes float32 scores using their shortest round-tripping decimal.
 # Restore the stored dtype before checking subtraction performed before CSV export.
 stored_score=P.saved_score.to_numpy(dtype=np.float32).astype(np.float64)
 residual_error=float(np.max(np.abs((stored_score-component)-P.residual_score)))
 assert residual_error<1e-10
 for name in tables:
  t=pd.read_csv(A/(name+'.csv'));t.insert(0,'model',model);tables[name].append(t)
  if 'valid_bootstraps' in t:assert (t.valid_bootstraps==10000).all() and (t.ci_low<=t.ci_high).all()
 ctx=tables['context_metrics'][-1]
 for q in ctx.itertuples():
  mask=(P.y==(1 if q.own=='EGY' else 0)) & (P.unseen_type_in_fold if q.unseen_types_only else True)
  y=P.loc[mask,'mat'].to_numpy();correct=(P.loc[mask,q.condition+'_context_score'].to_numpy()>0)==y
  val=(correct[y==0].mean()+correct[y==1].mean())/2;assert abs(val-q.estimate)<1e-14
 auc=tables['context_auroc'][-1]
 for q in auc.itertuples():
  mask=(P.y==(1 if q.own=='EGY' else 0)) & (P.unseen_type_in_fold if q.unseen_types_only else True)
  if q.fold!='pooled_OOF':mask &= P.fold_id==int(q.fold)
  if q.valid:assert abs(roc_auc_score(P.loc[mask,'mat'],P.loc[mask,q.condition+'_context_score'])-q.AUROC)<1e-14
 proj=tables['projection_diagnostics'][-1]
 for q in proj.itertuples():
  mask=(P.embedded==1)&(P.unseen_type_in_fold if q.unseen_types_only else True);T=P.loc[mask];y=T.y.to_numpy();sign=np.where(y==1,1,-1)
  ss=T.saved_score.to_numpy()*sign;rr=T.residual_score.to_numpy()*sign
  vals={}
  for a in [0,1]:
   m=y==a;vals[a]=[(ss[m]<0).mean(),(rr[m]<0).mean(),ss[m].mean(),rr[m].mean()]
  if q.own=='MSA_minus_EGY':
   gap0=vals[0][0]-vals[1][0];gap1=vals[0][1]-vals[1][1];val={'flip_gap_before':gap0,'flip_gap_after':gap1,'flip_gap_reduction':gap0-gap1}[q.metric]
  else:val=vals[1 if q.own=='EGY' else 0][['flip_before','flip_after','own_logit_before','own_logit_after'].index(q.metric)]
  assert abs(val-q.estimate)<1e-13
 check=json.loads((A/'analysis_verification.json').read_text());assert check['status']=='PASS' and check['original_point_metrics_exact']==16
 checks.append({'model':model,'projection_reconstruction_max_error':err,'residual_error_after_restoring_float32_score':residual_error,'csv_score_precision_note':'Restore original float32 dtype to recover its exact binary value from shortest decimal CSV. Summary means use decimal CSV values; difference below 1e-6, not bitwise equivalence.','context_point_checks':len(ctx),'context_auc_checks':int(auc.valid.sum()),'projection_point_checks':len(proj),'original_accuracies_exact':16,'fits':45,'input_audit':json.loads((A/'audit.json').read_text()),'analysis_path':str(A.relative_to(V))})
for name,ts in tables.items():pd.concat(ts,ignore_index=True).to_csv(O/(name+'.csv'),index=False)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'result':'PASS','models':checks,'original_point_accuracies_reproduced':64,'total_fitted_probes':180,'new_LLM_inference_in_cloud':False,'scope':'Review-time diagnostics on the existing exp11 recoverability cohort. Not new independent data; not the exp04 Table5 cohort/probes.','intervals':'Pointwise 95% tweet-cluster bootstrap intervals, 10000 resamples, fitted OOF predictions fixed; not simultaneous or confirmatory.'}
(O/'FOUR_MODEL_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='models'},indent=2))
