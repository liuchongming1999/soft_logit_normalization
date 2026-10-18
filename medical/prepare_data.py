"""Download the official 28x28 archives and verify their published MD5 and split sizes."""
from pathlib import Path
import argparse,hashlib,json,sys,urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from medmnist import INFO
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--datasets',nargs='+',default=['pneumoniamnist','bloodmnist','pathmnist']);args=p.parse_args();args.data_root.mkdir(parents=True,exist_ok=True)
for name in args.datasets:
 info=INFO[name];target=args.data_root/(name+'.npz')
 digest=lambda f:hashlib.md5(f.read_bytes()).hexdigest()
 if not target.exists() or digest(target)!=info['MD5']:
  tmp=target.with_suffix('.partial');urllib.request.urlretrieve(info['url'],tmp)
  assert digest(tmp)==info['MD5'],'Official data checksum mismatch';tmp.replace(target)
 with np.load(target) as z:counts={s:len(z[s+'_images']) for s in ['train','val','test']}
 assert counts==info['n_samples']
 receipt=dict(dataset=name,url=info['url'],md5=info['MD5'],bytes=target.stat().st_size,n_samples=counts)
 (args.data_root/(name+'.ready.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(name,counts)
