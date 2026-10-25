from pathlib import Path
import csv,json,os,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
audit=root/'revision/results/review_integration_20261024/Electronics_Selected_Curve_Independent_Review_20261024'
out=Path(os.environ['ELECTRONICS_REPRO_OUT'])/'curves';out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((audit/'ALL_MODELS_SELECTED_CURVE.csv').open()));models=['Qwen3-8B-Base','ALLaM-7B-Instruct-preview','gemma-2-9b','Fanar-1-9B'];names=['Qwen','ALLaM','Gemma','Fanar']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
figdir=out/'figures';figdir.mkdir(exist_ok=True)
def figure(metric,groups,labels,title,filename,ylabel,share=True):
 fig,axs=plt.subplots(2,2,figsize=(10.2,7.5),sharex=True,sharey=share,layout='constrained')
 for model,name,ax in zip(models,names,axs.flat):
  for g,label,color in zip(groups,labels,['#166a9d','#c65922']):
   subset=sorted([r for r in rows if r['model']==model and r['group']==g and (r['metric']==metric or metric=='D' and r['metric'].startswith('D_'))],key=lambda r:int(r['k']))
   x=np.array([int(r['k']) for r in subset]);y=np.array([float(r['delta_vs_baseline']) for r in subset]);lo=np.array([float(r['ci_low']) for r in subset]);hi=np.array([float(r['ci_high']) for r in subset]);ax.fill_between(x,lo,hi,color=color,alpha=.15,linewidth=0);ax.plot(x,y,color=color,label=label,lw=1.8,marker='o',ms=3)
  ax.axhline(0,color='#667085',lw=.8);ax.axvline(5,color='#98a2b3',lw=.7,ls=':');ax.set_title(name,fontweight='bold');ax.set_xticks([0,1,5,10,15,20]);ax.grid(axis='y',alpha=.17)
  ax.set_xlabel('Number of jointly replaced heads (k)');ax.set_ylabel(ylabel)
 handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
 fig.suptitle(title+'\nShading: pointwise 95% paired tweet-cluster bootstrap intervals',fontsize=12)
 fig.savefig(figdir/(filename+'.png'),dpi=180);fig.savefig(figdir/(filename+'.svg'));plt.close(fig)
figure('own_logit',['MSA_in_EGY','EGY_in_MSA'],['MSA in EGY context','EGY in MSA context'],'Selected-head cumulative intervention: switched words','switched_own_logit','Change in own-variety logit')
figure('flip_pp',['MSA_in_EGY','EGY_in_MSA'],['MSA in EGY context','EGY in MSA context'],'Selected-head cumulative intervention: flip rates','switched_flip','Change in flip rate (percentage points)')
figure('own_logit',['mono_MSA','mono_EGY'],['Monolingual MSA','Monolingual EGY'],'Monolingual controls: continuous readout effects','monolingual_own_logit','Change in own-variety logit')
figure('flip_pp',['mono_MSA','mono_EGY'],['Monolingual MSA','Monolingual EGY'],'Monolingual controls: flip-rate effects','monolingual_flip','Change in flip rate (percentage points)')
figure('cont_lp_all',['MSA_in_EGY','EGY_in_MSA'],['MSA in EGY context','EGY in MSA context'],'Secondary outcome: teacher-forced continuation','continuation','Change in mean token log probability')
figure('D',['MSA_in_EGY','EGY_in_MSA'],['MSA in EGY context','EGY in MSA context'],'Secondary outcome: context-oriented D (model-specific pools)','D_context','Change in D; compare within model',False)
