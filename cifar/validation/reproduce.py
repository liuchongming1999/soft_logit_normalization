"""Generate portable copies of the original675 configurations; optionally train."""
from pathlib import Path
import argparse,json,os,shlex,subprocess,sys
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--methods',nargs='+');p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)));p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');a=p.parse_args()
for r in json.loads((R/'records/screening_runs.json').read_text()):
 c=r['config'].copy()
 if a.methods and c['method'] not in a.methods or c['seed'] not in a.seeds:continue
 old=c['out_dir'];out=a.output_root.resolve()/r['id'];out.mkdir(parents=True,exist_ok=True);c['out_dir']=str(out);c['data_root']=str(a.data_root.resolve());c['workers']=4
 backend=R/'source'/r['family']/('reference_train.py' if r['family']=='standalone' and 'neurocomputing_revision_c100_20260926_v2' in old else 'train.py')
 cfg=out/'portable_config.json';cfg.write_text(json.dumps(c,indent=2));cmd=[sys.executable,str(backend),'--config',str(cfg)];print('CUDA_VISIBLE_DEVICES='+shlex.quote(a.gpu)+' '+shlex.join(cmd),flush=True)
 if a.execute:subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu},check=True)
