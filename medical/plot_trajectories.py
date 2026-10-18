"""Publication panels from all locked selected-parameter runs, epochs0..100."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'records'
A=json.loads((R/'diagnostic_results.json').read_text())
(Path(__file__).resolve().parent/'figures').mkdir(exist_ok=True)
methods=['ce','sln','ln','ln_refined']
colors={'ce':'#3274a1','sln':'#d95f02','ln':'#299471','ln_refined':'#9467bd'}
labels={'ce':r'CE $(0,1)$','sln':r'SLN $(0.9,1)$','ln':r'LN $(1,0.4)$','ln_refined':r'LN $(1,0.3)$'}
curves={}
for run in A['results']:
    rows=[json.loads(x) for x in (R/'trajectories'/run['run_id']/'trajectory.jsonl').read_text().splitlines()]
    for split in ['train','val']:
        values=sorted([x for x in rows if x['split']==split],key=lambda x:x['epoch'])
        assert [x['epoch'] for x in values]==list(range(101))
        curves[run['method'],run['seed'],split]=values
plt.rcParams.update({'font.family':'serif','font.size':8.2,'axes.titlesize':8.2,
    'axes.labelsize':8.2,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})

def panel(ax,split,metric,scale=1):
    for m in methods:
        a=np.array([[x[metric] for x in curves[m,seed,split]] for seed in range(5)])*scale
        for row in a:ax.plot(range(101),row,lw=.55,color=colors[m],alpha=.14)
        ax.plot(range(101),a.mean(0),lw=1.25,color=colors[m],label=labels[m])
    ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Epoch');ax.grid(alpha=.14,lw=.5)

fig,axes=plt.subplots(2,3,figsize=(6.85,4.55))
for i,(split,name) in enumerate([('train','Training'),('val','Validation')]):
    for j,(metric,ylab) in enumerate([('separation',r'$D$'),('cdnv',r'$\widehat V$'),('error','Error (%)')]):
        ax=axes[i,j];panel(ax,split,metric,100 if metric=='error' else 1)
        ax.set_title(f'({chr(97+i*3+j)}) {name}: '+['separation','CDNV','error'][j]);ax.set_ylabel(ylab)
        if metric=='error':ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(0,100);ax.set_yticks([0,.1,1,10,100]);ax.set_yticklabels(['0','0.1','1','10','100'])
        else:ax.set_yscale('log')
handles,names=axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(left=.085,right=.985,top=.95,bottom=.155,hspace=.5,wspace=.45)
fig.savefig(Path(__file__).resolve().parent/'figures'/'R06_medical_geometry.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(Path(__file__).resolve().parent/'figures'/'R06_medical_geometry.png',bbox_inches='tight',pad_inches=.04);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(6.85,4.1))
for i,(split,name) in enumerate([('train','Training'),('val','Validation')]):
    for j,(metric,ylab) in enumerate([('variance_margin',r'$v_\parallel$'),('variance_common',r'$v_\perp$')]):
        ax=axes[i,j];panel(ax,split,metric);ax.set_yscale('log');ax.set_ylabel(ylab)
        ax.set_title(f'({chr(97+i*2+j)}) {name}: '+['margin variance','common-logit variance'][j])
        ax.axvline(50,color='#777777',ls=':',lw=.7,alpha=.7)
fig.legend(handles,names,loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(left=.09,right=.985,top=.95,bottom=.165,hspace=.48,wspace=.32)
fig.savefig(Path(__file__).resolve().parent/'figures'/'R06_medical_variances.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(Path(__file__).resolve().parent/'figures'/'R06_medical_variances.png',bbox_inches='tight',pad_inches=.04)
print('Two publication figures rendered; all20 trajectories and all101epochs retained.')
