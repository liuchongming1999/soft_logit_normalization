"""Publication figures from the immutable, audited R04 numerical archives."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE=Path(__file__).resolve().parent
DATA=HERE
SUBMISSION=HERE/'figures'
SUBMISSION.mkdir(exist_ok=True)
ROWS=[json.loads(p.read_text()) for p in sorted((DATA/'results').glob('*.json'))]
COLORS={0.:'#2869a4',.4:'#7a63a8',.7:'#ce6236',1.:'#319475'}
NAMES={0.:r'CE ($\gamma=0$)',.4:r'SLN ($\gamma=0.4)',.7:r'SLN ($\gamma=0.7$)',1.:r'LN ($\gamma=1$)'}
USED={}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':8,
                     'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                     'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.65,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42})


def select(**kw):
    return [r for r in ROWS if all(r[k]==v for k,v in kw.items())]


def one(**kw):
    out=select(**kw);assert len(out)==1,(kw,len(out));return out[0]


def load(row):
    p=DATA/'results'/f'{row["id"]}.npz'
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==row['archive_sha256']
    USED[row['id']]=digest
    with np.load(p) as z:return {k:z[k] for k in z.files}


def decorate(ax,title,xlabel=r'Flow time $t$',log=False):
    ax.set_title(title,loc='left',pad=5)
    ax.set_xlabel(xlabel,labelpad=2)
    ax.grid(alpha=.17,linewidth=.5)
    if log:ax.set_yscale('log')


def legend(fig,gammas,extra=True):
    handles=[Line2D([0],[0],color=COLORS[g],label=NAMES[g]) for g in gammas]
    if extra:
        handles += [Line2D([0],[0],color='.25',label='Prediction'),
                    Line2D([0],[0],ls='none',marker='o',ms=3,mfc='none',mec='.25',label='Particle flow')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.065),ncol=len(handles),frameon=False,columnspacing=1.)


def save(fig,stem):
    for ext in ['pdf','png']:
        target=SUBMISSION/f'{stem}.{ext}' if ext=='pdf' else SUBMISSION/f'{stem}.png'
        fig.savefig(target,bbox_inches='tight',pad_inches=.045,dpi=190)
    plt.close(fig)


def main_figure():
    fig,axes=plt.subplots(2,3,figsize=(6.65,3.8),layout='constrained')
    titles=[r'(a) Mean separation $D/D_0$',r'(b) Radial variance $v_r/v_{r,0}$',
            r'(c) Contrast variance $v_t/v_{t,0}$',r'(d) Common variance $v_0/v_{0,0}$',
            r'(e) CDNV $V/V_0$',r'(f) Width and prediction error']
    for gamma in [0.,.7,1.]:
        d=load(one(group='baseline',k=10,gamma=gamma,tau=1.,rho=.01))
        for ax,j in zip(axes.flat,[0,2,3,4,1]):
            scale=d['analytic'][0,j]
            ax.plot(d['time'],d['analytic'][:,j]/scale,color=COLORS[gamma])
            at=np.arange(0,len(d['time']),10)
            ax.scatter(d['time'][at],d['measured'][at,j]/scale,facecolors='none',edgecolors=COLORS[gamma],s=13,linewidth=.65,zorder=3)
            ax.set_xlim(0,40)
        rs=sorted(select(group='baseline',k=10,gamma=gamma,tau=1.),key=lambda r:r['rho'])
        axes[1,2].loglog([r['rho'] for r in rs],
                        [max(r[f'variance_{b}_sup_relative_error'] for b in ['radial','contrast','common']) for r in rs],
                        '-o',ms=2.5,color=COLORS[gamma])
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        decorate(ax,title,log=i in [1,2,3,4])
    axes[1,2].set_xlabel(r'Initial relative radius $\rho_0/m_0$')
    axes[1,2].set_ylabel('Max. variance relative error',labelpad=2)
    legend(fig,[0.,.7,1.])
    save(fig,'R04_population_validation')


def binary():
    fig,axes=plt.subplots(2,2,figsize=(6.3,3.6),layout='constrained')
    indices=[0,2,4,1]
    titles=[r'(a) Mean separation $D/D_0$',r'(b) Radial variance $v_r/v_{r,0}$',
            r'(c) Common variance $v_0/v_{0,0}$',r'(d) CDNV $V/V_0$']
    for gamma in [0.,.4,.7,1.]:
        d=load(one(group='baseline',k=2,gamma=gamma,tau=1.,rho=.01))
        for ax,j in zip(axes.flat,indices):
            scale=d['analytic'][0,j]
            ax.plot(d['time'],d['analytic'][:,j]/scale,color=COLORS[gamma])
            at=np.arange(0,len(d['time']),10)
            ax.scatter(d['time'][at],d['measured'][at,j]/scale,facecolors='none',edgecolors=COLORS[gamma],s=12,linewidth=.6)
            ax.set_xlim(0,40)
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):decorate(ax,title,log=i>0)
    legend(fig,[0.,.4,.7,1.],extra=False)
    save(fig,'R04_binary_validation')


def width():
    fig,axes=plt.subplots(2,3,figsize=(6.65,3.7),layout='constrained')
    for col,k in enumerate([2,3,10]):
        for gamma in [0.,.4,.7,1.]:
            for tau in [.5,1.]:
                rs=sorted(select(group='baseline',k=k,gamma=gamma,tau=tau),key=lambda r:r['rho'])
                active=['radial','common']+(['contrast'] if k>2 else [])
                x=[r['rho'] for r in rs]
                style='-' if tau==1 else '--'
                axes[0,col].loglog(x,[r['cdnv_sup_relative_error'] for r in rs],style,color=COLORS[gamma],marker='o',ms=2)
                axes[1,col].loglog(x,[max(r[f'variance_{b}_sup_relative_error'] for b in active) for r in rs],style,color=COLORS[gamma],marker='o',ms=2)
        decorate(axes[0,col],f'({chr(97+col)}) CDNV, $K={k}$',r'$\rho_0/m_0$')
        decorate(axes[1,col],f'({chr(100+col)}) Variance blocks, $K={k}$',r'$\rho_0/m_0$')
        for ax in axes[:,col]:ax.set_ylabel('Max. relative error',labelpad=1)
    legend(fig,[0.,.4,.7,1.],extra=False)
    save(fig,'R04_width_errors')


def symmetry():
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.35),layout='constrained')
    for ax,k in zip(axes,[3,10]):
        for gamma in [0.,.7,1.]:
            rs=sorted(select(group='symmetry',k=k,gamma=gamma,rho=.1),key=lambda r:r['deformation'])
            b=one(group='baseline',k=k,gamma=gamma,tau=1.,rho=.1)
            ax.semilogy([0]+[r['deformation'] for r in rs],
                        [b['cdnv_sup_relative_error']]+[r['cdnv_sup_relative_error'] for r in rs],'-o',ms=3,color=COLORS[gamma])
            ax.semilogy([r['deformation'] for r in rs],[r['cdnv_matrix_sup_relative_error'] for r in rs],'--s',ms=3,color=COLORS[gamma])
        decorate(ax,f'({"a" if k==3 else "b"}) $K={k}$',r'Deformation $\delta$')
        ax.set_ylabel('Max. CDNV relative error')
    legend(fig,[0.,.7,1.],extra=False)
    save(fig,'R04_symmetry_errors')


def steps():
    fig,axes=plt.subplots(1,3,figsize=(6.65,2.1),layout='constrained')
    for col,(ax,k) in enumerate(zip(axes,[2,3,10])):
        for gamma in [0.,.7,1.]:
            ref=load(one(group='baseline',k=k,gamma=gamma,tau=1.,rho=.1))
            rs=sorted(select(group='euler',k=k,gamma=gamma),key=lambda r:r['dt'])
            errors=[]
            for r in rs:
                d=load(r)
                errors.append(np.max(np.abs(d['measured'][:,1]-ref['measured'][:,1])/ref['measured'][:,1]))
            ax.loglog([r['dt'] for r in rs],errors,'-o',ms=3,color=COLORS[gamma])
        decorate(ax,f'({chr(97+col)}) $K={k}$',r'Euler step $h$')
        ax.xaxis.set_major_locator(FixedLocator([.05,.1,.2,.4]))
        ax.xaxis.set_major_formatter(FixedFormatter(['0.05','0.1','0.2','0.4']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(.045,.45)
        ax.set_ylabel('Max. CDNV relative error',labelpad=1)
    legend(fig,[0.,.7,1.],extra=False)
    save(fig,'R04_step_convergence')


if __name__=='__main__':
    assert len(ROWS)==len(json.loads((DATA/'manifest.json').read_text())['cases'])
    main_figure();binary();width();symmetry();steps()
    (HERE/'figure_provenance.json').write_text(json.dumps(dict(source_archives=USED,
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SUBMISSION.glob('R04_*.pdf')}),indent=2))
    print('Five vector PDF figures generated from verified R04 archives.')
