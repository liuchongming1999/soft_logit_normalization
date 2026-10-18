"""Offline reconstruction of manuscript medical tables from frozen outputs, without fitting."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,sys
import numpy as np
from scipy.stats import t
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from medmnist.info import INFO
from medmnist.evaluator import getAUC,getACC
from logit_diagnostics import geometry
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'generated');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
R=ROOT/'records';load=lambda p:json.loads(p.read_text());cache={};rows=[];configs={}
manifest_path=ROOT/'manifest.json'
if manifest_path.exists():
 for name,expected in load(manifest_path)['files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,('archive hash mismatch',name)

for c in load(R/'table4_cases.json'):
 path=ROOT/c['prediction_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==c['prediction_sha256']
 if str(path) not in cache:
  with np.load(path) as z:probs=z['probabilities'];y=z['labels'].reshape(-1);indices=z['indices']
  ds=c['dataset'];k=len(INFO[ds]['label']);assert probs.shape==(len(y),k) and len(y)==INFO[ds]['n_samples']['test'];assert np.array_equal(indices,np.arange(len(y)));assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-6)
  predicted=probs.argmax(1);cm=confusion_matrix(y,predicted,labels=list(range(k)))
  metrics=dict(acc=float(getACC(y,probs,INFO[ds]['task'])),auc=float(getAUC(y,probs,INFO[ds]['task'])),macro_f1=float(f1_score(y,predicted,average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,predicted)))
  if k==2:
   tn,fp,fn,tp=cm.ravel();metrics.update(sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
  cache[str(path)]=metrics
 metrics=cache[str(path)];assert abs(metrics['acc']-c['expected_acc'])<1e-12;assert abs(metrics['auc']-c['expected_auc'])<1e-12
 result=load(ROOT/c['result_file']);assert result['seed']==c['seed'] and result['epochs']==100;cfg=result['config']
 assert cfg['optimizer']=='Adam' and cfg['learning_rate']==.001 and cfg['weight_decay']==0 and cfg['batch_size']==128 and cfg['milestones']==[50,75]
 key=(c['dataset'],c['setting'],c['method']);selected=dict(dataset=c['dataset'],method=c['method'],gamma=cfg['gamma'],tau=cfg['tau'],focal_exponent=cfg.get('focal_exponent') or 2,beta=cfg.get('class_balanced_beta') or .999)
 if key in configs:assert configs[key]==selected
 configs[key]=selected;rows.append({k:c[k] for k in ['dataset','setting','method','seed']}|metrics)
summary=[]
for key in sorted(configs):
 ds,setting,method=key;group=[x for x in rows if (x['dataset'],x['setting'],x['method'])==key];assert sorted(x['seed'] for x in group)==list(range(5));entry=dict(dataset=ds,setting=setting,method=method,n=5)
 for metric in group[0]:
  if metric in ['dataset','setting','method','seed']:continue
  values=[x[metric] for x in group];entry[metric+'_mean']=statistics.mean(values);entry[metric+'_sd']=statistics.stdev(values)
 summary.append(entry)
def write_csv(name,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (a.output_dir/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
write_csv('medical_per_seed.csv',rows);write_csv('table4.csv',summary)
selected=[v for (ds,setting,method),v in configs.items() if setting=='selected'];(a.output_dir/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
# Assert manuscript archives match independently rebuilt values, including all task-specific metrics.
archived=load(R/'task_metrics.json')['summary']
for entry in summary:
 ref=next(x for x in archived if all(x[k]==entry[k] for k in ['dataset','setting','method']))
 for k,v in entry.items():
  if k.endswith('_mean') or k.endswith('_sd'):assert abs(v-ref[k])<1e-10,(entry,k,v,ref[k])
paired=[]
for ds in ['pneumoniamnist','bloodmnist','pathmnist']:
 for setting in ['default','selected']:
  for metric in ['acc','auc']:
   for baseline in ['ce','ln']:
    d=[]
    for seed in range(5):
     get=lambda m:next(x[metric] for x in rows if (x['dataset'],x['setting'],x['method'],x['seed'])==(ds,setting,m,seed))
     d.append(100*(get('sln')-get(baseline)))
    mean=statistics.mean(d);half=float(t.ppf(.975,4))*statistics.stdev(d)/np.sqrt(5);paired.append(dict(dataset=ds,setting=setting,metric=metric,baseline=baseline,mean=mean,ci95_low=mean-half,ci95_high=mean+half))
    ref=next(x for x in load(R/'paired_statistics.json')['rows'] if (x['dataset'],x['setting'],x['metric'])==(ds,setting,metric))['vs_'+baseline]
    assert abs(mean-ref['mean'])<1e-12 and np.allclose([mean-half,mean+half],ref['ci95'],atol=1e-12)
write_csv('table15_paired.csv',paired)
write_csv('table16_pneumonia.csv',[x for x in summary if x['dataset']=='pneumoniamnist']);write_csv('table17_multiclass.csv',[x for x in summary if x['dataset']!='pneumoniamnist'])
diag=[]
for c in load(R/'diagnostic_checkpoints.json'):
 row=dict(method=c['method'],seed=c['seed'],test_acc=c['metrics']['test']['acc'],test_auc=c['metrics']['test']['auc'])
 for split in ['train','val']:
  with np.load(ROOT/c[split+'_prediction_file']) as z:raw=z['raw_logits'];y=z['labels'].reshape(-1);assert np.array_equal(z['indices'],np.arange(len(y)))
  row[split+'_cdnv']=geometry(raw,y)['cdnv']
 diag.append(row)
means=[]
for method in ['ce','sln','ln','ln_refined']:
 group=[x for x in diag if x['method']==method];assert len(group)==5;row=dict(method=method,n=5)
 for metric in ['test_acc','test_auc','train_cdnv','val_cdnv']:
  values=[x[metric] for x in group];row[metric+'_mean']=statistics.mean(values);row[metric+'_sd']=statistics.stdev(values)
 ref=next(x for x in load(R/'best_checkpoint_geometry.json')['summary'] if x['method']==method)
 for split in ['train','val']:
  for stat in ['mean','sd']:assert abs(row[split+'_cdnv_'+stat]-ref['geometry'][split]['cdnv'][stat])<1e-12
 means.append(row)
write_csv('table18_diagnostics.csv',means)
# Human-readable LaTeX cells for the original combined medical comparison.
lines=[]
for ds in ['pneumoniamnist','bloodmnist','pathmnist']:
 allmeans=[x['acc_mean'] for x in summary if x['dataset']==ds];best=max(allmeans);lines.append('% '+ds)
 for m in ['ce','ln','sln','focal','cbce']:
  cells=[]
  for setting in ['default','selected']:
   x=next(x for x in summary if (x['dataset'],x['setting'],x['method'])==(ds,setting,m))
   for metric in ['auc','acc']:
    value=f"{100*x[metric+'_mean']:.2f}\\pm{100*x[metric+'_sd']:.2f}"
    if metric=='acc' and x['acc_mean']==best:value='\\mathbf{'+value+'}'
    cells.append('$'+value+'$')
  lines.append(m.upper()+' & '+' & '.join(cells)+r'\\')
(a.output_dir/'table4_cells.tex').write_text('\n'.join(lines)+'\n')
receipt=dict(table4_entries=len(rows),unique_table4_predictions=len(cache),table4_settings=len(summary),diagnostic_runs=len(diag),paired_comparisons=len(paired),original_metrics_match=True,selection_rule='locked validation AUC',test_used_for_selection=False)
(a.output_dir/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
