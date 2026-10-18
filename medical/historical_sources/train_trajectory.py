"""Matched MedMNIST loss comparison; official ResNet18 and evaluation code.
Training data stay on the GPU to eliminate image-loader overhead. Test metrics
are evaluated only after validation-selected checkpoint selection is locked.
"""
import os,sys,json,time,random,hashlib,csv,argparse,platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from medmnist import INFO,Evaluator
from sln_losses import sln_logits
from models import ResNet18
from logit_diagnostics import record_epoch

def atomic_json(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_torch(path,obj):
 tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def state_hash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
ACTIVE_GAMMA=None
ACTIVE_TAU=None
def scores_for(raw,method):
 if method=='focal':return raw
 gamma=ACTIVE_GAMMA if ACTIVE_GAMMA is not None else {'ce':0.,'ln':1.,'sln':.7}[method]
 tau=ACTIVE_TAU if ACTIVE_TAU is not None else (1. if method=='ce' else .2)
 return sln_logits(raw,gamma=gamma,tau=tau)
def loss_for(scores,target,method):
 ce=F.cross_entropy(scores,target,reduction='none')
 if method=='focal':return ((1.0-torch.exp(-ce))**2*ce).mean()
 return ce.mean()
def selfcheck():
 z=torch.tensor([[0.3,-1.,2.],[1.3,0.1,-0.8]],dtype=torch.float64,requires_grad=True);y=torch.tensor([2,0])
 torch.testing.assert_close(loss_for(z,y,'ce'),F.cross_entropy(z,y))
 torch.testing.assert_close(scores_for(z,'ln'),z/(.2*(z.norm(dim=1,keepdim=True)+1e-8)))
 torch.testing.assert_close(scores_for(z,'sln'),z/(.2*(z.norm(dim=1,keepdim=True)+1e-8).pow(.7)))
 for m in ['ce','ln','sln','focal']:
  out=loss_for(scores_for(z,m),y,m);g=torch.autograd.grad(out,z,retain_graph=True)[0];assert torch.isfinite(g).all()
 return True
@torch.no_grad()
def predict(model,images,method,batch_size=128):
 model.eval();raw=[]
 for start in range(0,len(images),batch_size):raw.append(model(images[start:start+batch_size].contiguous(memory_format=torch.channels_last)).float())
 raw=torch.cat(raw);scores=scores_for(raw,method);return raw.cpu().numpy(),scores.softmax(dim=1).cpu().numpy()
def run(args):
 global ACTIVE_GAMMA,ACTIVE_TAU
 selfcheck()
 assert 0<=args.gamma<=1 and args.tau>0
 assert (args.method=='ce' and args.gamma==0) or (args.method=='ln' and args.gamma==1) or (args.method=='sln' and 0<args.gamma<1)
 ACTIVE_GAMMA=args.gamma;ACTIVE_TAU=args.tau;torch.set_num_threads(2);torch.set_num_interop_threads(1)
 seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():print('Already completed',args.run_id,flush=True);return
 info=INFO[args.dataset];nclasses=len(info['label']);nc=info['n_channels'];data_root=ROOT/'data'
 assert (data_root/(args.dataset+'.ready.json')).exists()
 receipt=json.loads((data_root/(args.dataset+'.ready.json')).read_text());t0=time.time()
 model=ResNet18(in_channels=nc,num_classes=nclasses);initial_hash=state_hash(model.state_dict());model=model.to('cuda',memory_format=torch.channels_last)
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.);scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[50,75],gamma=.1)
 train_order=torch.Generator().manual_seed(seed+1000003)
 data={};labels_cpu={};labels_gpu={};split_names=['train','val'] if args.stage=='screen' else ['train','val','test']
 with np.load(data_root/(args.dataset+'.npz')) as z:
  for split in split_names:
   images=z[split+'_images'];labels=z[split+'_labels'].reshape(-1).astype(np.int64);labels_cpu[split]=labels
   if images.ndim==3:images=images[:,:,:,None]
   cpu=torch.from_numpy(np.ascontiguousarray(images.transpose(0,3,1,2))).float().div_(255.).sub_(.5).div_(.5)
   # Check exact torchvision preprocessing on real images before moving them.
   if split=='train':
    import torchvision.transforms as T
    tr=T.Compose([T.ToTensor(),T.Normalize(mean=[.5],std=[.5])]);sample=images[0,:,:,0] if nc==1 else images[0]
    torch.testing.assert_close(cpu[0],tr(sample),rtol=0,atol=0)
   data[split]=cpu.to('cuda',memory_format=torch.channels_last);labels_gpu[split]=torch.from_numpy(labels).to('cuda')
 evaluators={split:Evaluator(args.dataset,split,root=str(data_root)) for split in split_names}
 config={'run_id':args.run_id,'dataset':args.dataset,'method':args.method,'seed':seed,'epochs':args.epochs,'smoke_only':args.epochs!=100,'backbone':'official MedMNIST ResNet18','n_classes':nclasses,'n_channels':nc,'parameter_count':sum(p.numel() for p in model.parameters()),'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'betas':[.9,.999],'eps':1e-8,'batch_size':128,'milestones':[50,75],'lr_decay':.1,'data_receipt':receipt,'initial_state_sha256':initial_hash,'order_seed':seed+1000003,'gamma':args.gamma,'tau':args.tau,'stage':args.stage,'normalization_epsilon':1e-8,'resume_checkpoint_source':args.resume_from,'focal_exponent':2. if args.method=='focal' else None,'label_smoothing':0.,'class_weight':None,'training_precision':'float32','cudnn_allow_tf32':True,'matmul_allow_tf32':False,'cudnn_benchmark':True,'memory_format':'channels_last','data_loading':'official uint8->ToTensor->Normalize exactly checked; tensors cached on GPU','torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'visible_device':os.getenv('CUDA_VISIBLE_DEVICES'),'python':platform.python_version(),'selection':'max validation AUC; earliest on ties','test_used_for_selection':False,'official_experiments_commit':'70b6b3a7ad7afddff1df2a3b735235830fbdb142','medmnist_api_commit':'805b74237fb5bd2db376fd94e50646b6a8905d9d'}
 atomic_json(out/'resolved_config.json',config)
 start_epoch=0;best_auc=-1.;best_epoch=0;best_acc=0.;timings=[]
 config['diagnostic_protocol']='raw logits, model.eval(), fixed full train/val; equal-class population covariance; epoch0 and every epoch'
 config['diagnostic_protocol_variant']='validation_selected_loss_parameters'
 atomic_json(out/'resolved_config.json',config)
 if args.resume_from and not (out/'last.pt').exists():
  import shutil
  source=Path(args.resume_from)
  assert source.is_file() and source.name=='last.pt'
  os.link(source,out/'last.pt');shutil.copy2(source.parent/'epochs.csv',out/'epochs.csv')
 if (out/'last.pt').exists():
  state=torch.load(out/'last.pt',map_location='cuda',weights_only=False);prior=state['best_checkpoint']['config'];assert prior['dataset']==args.dataset and prior['seed']==args.seed and prior['gamma']==args.gamma and prior['tau']==args.tau;atomic_torch(out/'best.pt',state['best_checkpoint']);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);train_order.set_state(state['order_rng'].cpu());start_epoch=state['epoch'];best_auc=state['best_auc'];best_acc=state['best_acc'];best_epoch=state['best_epoch'];timings=state.get('timings',[])
  # Remove epoch rows beyond the last durable training state before appending.
  if (out/'epochs.csv').exists():
   with (out/'epochs.csv').open() as f:rows=list(csv.DictReader(f));fields=rows[0].keys() if rows else ['epoch','lr','train_loss','val_auc','val_acc','epoch_seconds']
   with (out/'epochs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r for r in rows if int(r['epoch'])<=start_epoch)
 trajectory=out/'trajectory.jsonl'
 if start_epoch:
  assert trajectory.exists()
  rows=[json.loads(x) for x in trajectory.read_text().splitlines() if x.strip()]
  rows=[x for x in rows if x['epoch']<=start_epoch]
  trajectory.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 else:
  trajectory.write_text('')
  record_epoch(model,data,labels_cpu,args.method,out,0,predict,state_hash)
 mode='a' if start_epoch else 'w'
 with (out/'epochs.csv').open(mode,newline='') as cf:
  writer=csv.DictWriter(cf,fieldnames=['epoch','lr','train_loss','val_auc','val_acc','epoch_seconds'])
  if not start_epoch:writer.writeheader()
  for epoch in range(start_epoch,args.epochs):
   tick=time.monotonic();model.train();perm=torch.randperm(len(data['train']),generator=train_order).to('cuda');total_loss=torch.zeros((),device='cuda');lr=optimizer.param_groups[0]['lr']
   for idx in perm.split(128):
    xb=data['train'].index_select(0,idx).contiguous(memory_format=torch.channels_last);yb=labels_gpu['train'].index_select(0,idx)
    optimizer.zero_grad(set_to_none=True);raw=model(xb);scores=scores_for(raw,args.method);loss=loss_for(scores,yb,args.method);loss.backward();optimizer.step();total_loss+=loss.detach()*len(idx)
   torch.cuda.synchronize();train_loss=float(total_loss)/len(perm)
   assert np.isfinite(train_loss),('nonfinite train loss',args.run_id,epoch)
   _,val_probs=predict(model,data['val'],args.method);val_auc,val_acc=evaluators['val'].evaluate(val_probs)
   record_epoch(model,data,labels_cpu,args.method,out,epoch+1,predict,state_hash)
   scheduler.step()
   if val_auc>best_auc:
    best_auc=float(val_auc);best_acc=float(val_acc);best_epoch=epoch+1
    atomic_torch(out/'best.pt',{'model':model.state_dict(),'epoch':best_epoch,'val_auc':best_auc,'val_acc':best_acc,'config':config})
   elapsed=time.monotonic()-tick;timings.append(elapsed)
   row={'epoch':epoch+1,'lr':lr,'train_loss':train_loss,'val_auc':float(val_auc),'val_acc':float(val_acc),'epoch_seconds':elapsed};writer.writerow(row);cf.flush()
   atomic_json(out/'progress.json',{**row,'best_epoch':best_epoch,'best_val_auc':best_auc,'updated_at':time.time(),'pid':os.getpid(),'run_id':args.run_id})
   if (epoch+1)%10==0 or epoch==start_epoch:print(json.dumps({**row,'run':args.run_id,'best_epoch':best_epoch}),flush=True)
   if (epoch+1)%25==0 or epoch+1==args.epochs:
    atomic_torch(out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'order_rng':train_order.get_state(),'epoch':epoch+1,'best_auc':best_auc,'best_acc':best_acc,'best_epoch':best_epoch,'timings':timings,'best_checkpoint':torch.load(out/'best.pt',map_location='cpu',weights_only=False)})
 best=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best['model']);metrics={}
 for split in (['val'] if args.stage=='screen' else split_names):
  raw,probs=predict(model,data[split],args.method);auc,acc=evaluators[split].evaluate(probs,save_folder=str(out),run=args.run_id);metrics[split]={'auc':float(auc),'acc':float(acc)}
  np.savez_compressed(out/(split+'_predictions.npz'),raw_logits=raw,probabilities=probs,labels=labels_cpu[split],indices=np.arange(len(raw)))
 validation_recompute_delta=metrics['val']['auc']-best['val_auc']
 assert abs(validation_recompute_delta)<(1e-4 if args.resume_from else 1e-9),(metrics['val'],best['val_auc'])
 result={'run_id':args.run_id,'dataset':args.dataset,'method':args.method,'seed':seed,'epochs':args.epochs,'best_epoch':int(best['epoch']),'metrics':metrics,'initial_state_sha256':initial_hash,'parameter_count':config['parameter_count'],'total_wall_seconds':time.time()-t0,'epoch_seconds_median':float(np.median(timings)),'last10_epoch_seconds_median':float(np.median(timings[-10:])),'completed_at':time.time(),'smoke_only':args.epochs!=100,'selection':'validation_auc','selection_validation_auc':best['val_auc'],'validation_recompute_delta':validation_recompute_delta,'test_used_for_selection':False,'test_evaluated':args.stage!='screen','config':config}
 atomic_json(out/'result.json',result);print('COMPLETED '+json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',choices=['ce','ln','sln','focal'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--gamma',type=float,required=True);p.add_argument('--tau',type=float,required=True);p.add_argument('--stage',choices=['screen','confirm'],required=True);p.add_argument('--resume-from');run(p.parse_args())
