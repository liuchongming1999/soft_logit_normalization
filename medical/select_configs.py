"""Lock one loss configuration per dataset/method using screening validation AUC only."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups={}
for f in a.run_root.glob('*/result.json'):
 x=json.loads(f.read_text());c=x['config']
 if c.get('stage')!='screen':continue
 assert not x['test_evaluated'] and 'test' not in x['metrics']
 groups.setdefault((x['dataset'],x['method']),[]).append(x)
rows=[]
for (ds,m),group in sorted(groups.items()):
 expected=20 if m=='sln' else 5 if m in ['ce','ln'] else 3
 assert len(group)==expected,(ds,m,len(group),expected)
 best=sorted(group,key=lambda x:(-x['selection_validation_auc'],x['config'].get('focal_exponent') or 0,x['config'].get('class_balanced_beta') or 0,x['run_id']))[0];c=best['config']
 rows.append(dict(dataset=ds,method=m,gamma=c['gamma'],tau=c['tau'],focal_exponent=c.get('focal_exponent') or 2,beta=c.get('class_balanced_beta') or .999,screen_run=best['run_id'],validation_auc=best['selection_validation_auc']))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n');print('Locked',len(rows),'configurations using validation AUC; no test scores read.')
