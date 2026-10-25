from common import *
import zipfile

def main():
 state=read('STATUS.json');before=read('manifest/inputs_before.json');after=inventory([Path(p) for p in before]);save('manifest/inputs_after.json',after);changed=[p for p in before if before[p]!=after.get(p)]
 save('manifest/source_preservation.json',{'original_files_checked':len(before),'unchanged':not changed,'changes':changed})
 if changed:state.update(status='STOPPED',reason='Original inputs changed');save('STATUS.json',state)
 gates={n:read('results/'+n+'.json') for n in ['EARLY_GATE','FULL_GATE','P3_GATE'] if (WORK/'results'/f'{n}.json').exists()}
 report=['تقرير '+SLUG+' لثبات اختيار الرؤوس — electronics-4637664','الحالة: '+state['status'],'السبب: '+str(state.get('reason') or state.get('substatus') or 'اكتملت جميع البوابات'),'المجلد: '+str(WORK),'المصدر: '+str(ROOT),'',dumps(state),'بوابات المقارنة:',dumps(gates),'ثبات جميع المدخلات الأصلية: '+str(not changed),'','إعادة البناء مشروطة بالشيفرة والبيانات وtokenizer والإعدادات المستعادة. لا تثبت استعادة معرّفات بوادئ تاريخية لم تُحفظ. لم تستخدم مصادر الحزمة بديلًا عن أصل المشروع. لم تُعد تمثيلات exp11 أو بوابة exp07 أو النماذج الأخرى.','اتجاه القراءة فرق المتوسطات الأصلي، مع 400 تغريدة لكل تنوع وبذرة مستقلة 13؛ ليس المسبار اللوجستي. أُبقي تداخل تغريدات الاتجاه مع مصادر البوادئ موثقًا، ولم يُعد تقدير الاتجاه داخل folds.','حُفظت تكرارات البوادئ وبدائل sid ودفق RNG. التقسيم إلى خمسة أقسام حسب المكونات المتصلة وGroupKFold(shuffle=True, random_state=13)؛ لا تداخل لمصدر أو بصمة بادئة بين تدريب واختبار القسم.','العتبة ثابتة: abs(new-old)<=1e-4+1e-4*abs(old). إعادة ترتيب الطبقات تعتمد تدريب كل fold فقط؛ لا تغيير للاتحاد بعد النتائج.','هذه حساسية داخل العينة الأصلية؛ ليست تأكيدًا مستقلاً، ولا تختبر الإزالة المشتركة أو الوساطة أو قوائم D أو تثبت الرسمية. حالة COMPLETE تخص هذا النموذج وهذا التكليف فقط.']
 if (WORK/'results/STABILITY_ANALYSIS.json').exists():
  a=read('results/STABILITY_ANALYSIS.json');report+=['','ملخص الثبات:']
  for f in a['folds']:report.append(f"القسم {f['fold']}: الطبقات {f['top3_layers']}؛ الرؤوس "+str([(x['layer'],x['head']) for x in f['top5_heads']])+f"؛ تداخل مع الأصل {f['original_top5_overlap_n']}/5؛ Jaccard={f['original_top5_jaccard']:.6f}؛ متوسط تأثير الرأس الفردي في الاختبار={f['test_mean_of_individual_selected_head_deltas']:.8f}.")
  js=[x['jaccard'] for x in a['pairwise_fold_jaccards']];report.append(f'Jaccard بين الأقسام: من {min(js):.6f} إلى {max(js):.6f}. الاختلافات محفوظة دون انتقاء أو عتبة نجاح علمي.');report.append('متوسط آثار الرؤوس الفردية ليس أثر إزالة الرؤوس الخمسة معًا ولا مجموع مساهمات سببية.')
 if (WORK/'selection/overlap_audit.json').exists():report+=['عدد تغريدات اتجاه القراءة المشتركة مع مصادر البوادئ: '+str(read('selection/overlap_audit.json')['readout_prefix_source_overlap_n'])]
 if (WORK/'RUNTIME_NOTES_AR.txt').exists():report+=[(WORK/'RUNTIME_NOTES_AR.txt').read_text()]
 (WORK/'HEAD_STABILITY_REPORT_AR.txt').write_text('\n'.join(report)+'\n')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=ROOT/f'ELECTRONICS_HEAD_STABILITY_{SLUG}_RETURN_{stamp}.zip';log('package_begin',status=state['status'],output=str(dest))
 payload=[p for p in sorted(WORK.rglob('*')) if p.is_file() and not p.is_symlink() and '/logs/tmp/' not in str(p) and p.suffix!='.tmp'];assert not any(p.suffix in {'.safetensors','.pt','.pth','.parquet','.zip'} for p in payload)
 manifest={'utc':now(),'paper':'electronics-4637664','status':state['status'],'files':[{'path':str(p.relative_to(WORK.parent)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload],'self_hash_rule':'RETURN_MANIFEST.json hash stored in external verification sidecar'}
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in payload:z.write(p,str(p.relative_to(WORK.parent)))
  z.writestr('RETURN_MANIFEST.json',dumps(manifest))
 dest.with_suffix('.workdir.json').write_text(dumps({'workdir':str(WORK),'status':state['status'],'archive':str(dest)}));print(str(dest),flush=True)
if __name__=='__main__':main()
