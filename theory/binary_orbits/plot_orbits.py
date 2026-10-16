"""Plot exact level sets and the archived binary-logit particle trajectories."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
S=json.loads((HERE/'summary.json').read_text())
assert hashlib.sha256((HERE/'trajectories.npz').read_bytes()).hexdigest()==S['trajectory_sha256']
D=np.load(HERE/'trajectories.npz')
COLORS={0:'#2465a5',.4:'#9563ad',.7:'#d76334',1:'#29927c'}
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def select(gamma, tau=1., theta=1.2, radius=1., label=1):
    found=[i for i,c in enumerate(S['cases']) if c['gamma']==gamma and
           c['tau']==tau and c['theta']==theta and c['radius']==radius and c['label']==label]
    assert len(found)==1
    return D['cartesian'][:,found[0],:]


def invariant(path, gamma):
    return np.linalg.norm(path,axis=-1)**gamma*np.abs(path[...,1])**(1-gamma)


def main():
    fig,axs=plt.subplots(2,2,figsize=(10.5,9),layout='constrained')
    for ax,g in zip(axs.flat,[0.,.4,.7,1.]):
        for theta in [.4,1.2,2.2]:
            for sign in [-1,1]:
                path=select(g,theta=sign*theta)
                c=invariant(path[:1],g)[0]
                angles=np.linspace(.018,np.pi-.018,1600)
                rr=c/np.sin(angles)**(1-g)
                xx=rr*np.cos(angles); yy=sign*rr*np.sin(angles)
                mask=(xx>-1.2)&(xx<3.6)&(np.abs(yy)<1.7)
                ax.plot(np.where(mask,xx,np.nan),np.where(mask,yy,np.nan),
                        color='#b8c1cb',ls='--',lw=1.1,zorder=1)
                ax.plot(path[:,0],path[:,1],color=COLORS[g],lw=1.6,zorder=2)
                ax.scatter(path[::30,0],path[::30,1],facecolors='white',edgecolors=COLORS[g],s=18,zorder=3)
                ax.scatter(path[0,0],path[0,1],color='black',s=16,zorder=4)
                k=50
                ax.annotate('',path[k+10],path[k],arrowprops={'arrowstyle':'->','color':COLORS[g],'lw':1.4})
        ax.axvline(0,color='#d4dae1',lw=.7);ax.axhline(0,color='#d4dae1',lw=.7)
        ax.set(xlim=(-1.2,3.6),ylim=(-1.7,1.7),xlabel=r'Margin coordinate $u_\parallel$',
               ylabel=r'Common-logit coordinate $u_\perp$',aspect='equal')
        label={0:'CE: common coordinate fixed',.4:'SLN: mixed orbit',.7:'SLN: mixed orbit',1:'LN: radius fixed'}[g]
        ax.set_title(r'$\gamma='+str(g)+r'$ — '+label,fontsize=11)
    fig.legend(handles=[Line2D([],[],color='#9ba8b6',ls='--',label='Exact level set'),
                        Line2D([],[],color='#526574',marker='o',mfc='white',label='RK4 trajectory / saved points'),
                        Line2D([],[],color='black',marker='o',ls='',label='Initial state')],
               loc='outside upper center',ncols=3,fontsize=10)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'binary_orbits.pdf',bbox_inches='tight')
    fig.savefig(out/'binary_orbits.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
