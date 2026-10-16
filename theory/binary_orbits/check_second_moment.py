"""Reuse the stored off-axis trajectories to check exact symmetric moments."""
from pathlib import Path
import json
import numpy as np

p=Path(__file__).resolve().parent
s=json.loads((p/'summary.json').read_text())
d=np.load(p/'trajectories.npz')
a=d['cartesian']
maximum=0.; mean_error=0.; groups=0
for gamma in [.2,.4,.7,.9]:
    for tau in [.5,1.,2.]:
        for radius in [1.,2.]:
            for label in [1,2]:
                indices=[i for i,c in enumerate(s['cases'])
                         if (c['gamma'],c['tau'],c['radius'],c['label'])
                         ==(gamma,tau,radius,label)]
                assert len(indices)==6
                cloud=a[:,indices,:]
                r=np.linalg.norm(cloud,axis=-1)
                actual=np.mean(cloud[:,:,1]**2,axis=1)
                predicted=np.mean(cloud[0,:,1]**2*(r[0]/r)**(2*gamma/(1-gamma)),axis=1)
                maximum=max(maximum,float(np.max(np.abs(actual/predicted-1))))
                mean_error=max(mean_error,float(np.max(np.abs(np.mean(cloud[:,:,1],axis=1)))))
                groups+=1
assert maximum<1e-8 and mean_error<1e-14
result=dict(symmetric_population_groups=groups,states_per_group=len(d['times']),
            points_per_group=6,max_relative_second_moment_transport_error=maximum,
            max_absolute_common_coordinate_mean=mean_error,reused_existing_trajectories=True)
(p/'second_moment_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
