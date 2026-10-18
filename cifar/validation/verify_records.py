"""Rebuild both Table3 columns; audit all675 validation histories offline."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,statistics
import numpy as np
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=R/'generated');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text())
if (R/'manifest.json').exists():
 for n,h in load(R/'manifest.json')['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
runs=load(R/'records/screening_runs.json');assert len(runs)==675
with gzip.open(R/'records/training_histories.json.gz','rt') as f:histories=json.load(f)
byid={r['id']:r for r in runs};assert len(byid)==675
for h in histories:
 rows=h['rows'];assert len(rows)==160
 mx=max(float(x['val_acc']) for x in rows);epoch=int(next(x['epoch'] for x in rows if float(x['val_acc'])==mx));s=byid[h['id']]['summary'];assert abs(mx-s['best_val_acc'])<1e-12 and epoch==s['best_epoch']
reg=load(R/'records/regularizer_selected.json')['results'];base=load(R/'records/standalone_selected.json')['results'];allresults={**base,**reg};assert len(allresults)==15
besttest=load(R.parent/'results/cifar100_besttest.json')['finalized'];report=[];prediction_count=0
for method,x in sorted(allresults.items()):
 candidate={h:statistics.mean(r['summary']['best_val_acc'] for r in runs if r['config']['method']==method and r['config']['setting_id']==h) for h in range(9)}
 selected=x['config'].get('setting_id',x.get('setting_id'));highest=max(candidate.values());assert abs(candidate[selected]-highest)<1e-12,(method,selected,candidate)
 rr=x['per_seed'];assert sorted(r['seed'] for r in rr)==list(range(5));v=[r['test_acc'] for r in rr];assert abs(statistics.mean(v)-x['test_mean'])<1e-12 and abs(statistics.stdev(v)-x['test_sd'])<1e-12
 for r in rr:
  if 'prediction_file' in r:
   with np.load(R/r['prediction_file']) as z:assert np.array_equal(z['indices'],np.arange(10000));assert float((z['predicted']==z['labels']).mean())==r['test_acc']
   prediction_count+=1
 bt=besttest['r20_'+method];scores=[r['accuracy'] for r in bt['seeds']];assert abs(statistics.mean(scores)-bt['mean'])<1e-12 and abs(statistics.stdev(scores)-bt['sd'])<1e-12
 report.append(dict(method=method,best_test_mean_percent=100*bt['mean'],best_test_sd_percent=100*bt['sd'],validation_selected_mean_percent=100*x['test_mean'],validation_selected_sd_percent=100*x['test_sd']))
assert prediction_count==30
with (a.output_dir/'table3.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
(a.output_dir/'audit.json').write_text(json.dumps(dict(training_histories_verified=675,selected_methods=15,seeds_per_method=5,regularizer_prediction_files_verified=30,all_metrics_match=True),indent=2))
print(json.dumps(dict(training_histories_verified=675,selected_methods=15,regularizer_prediction_files_verified=30,all_metrics_match=True)))
