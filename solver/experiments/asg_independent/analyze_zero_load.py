"""Post hoc analytic diagnosis; does not change labels, stopping rules or timings."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent/'results'
manifest=json.loads((ROOT/'manifest.json').read_text())
g=next(v for v in manifest['geometries'] if v['geometry_id']=='asg2_ramp_g040')
p=g['parameters']
assert p['direction']==[0,1,0] and p['levels']==2 and g['normalization']==[0.,1.]
# The sampled layer profile depends only on y. For engineering shear gamma_xz=1,
# sigma_xz=sigma_zx=mu0*rho(y), all other stresses vanish. Its divergence and
# the interface traction jump are exactly zero. The exact fluctuation is zero.
n=64;y=np.arange(n,dtype=np.float64)/n
coordinate=np.mod(y+p['phases'][0]/(2*np.pi),1)**p['coordinate_warp']
q=(np.minimum(np.floor(coordinate*p['levels']),p['levels']-1)/(p['levels']-1))**p['profile_power']
rows=[]
for contrast in [100,1000,10000]:
    sid=f"{g['geometry_id']}_c{contrast}_n64"
    rho=np.ascontiguousarray(np.broadcast_to(np.exp(np.log(contrast)*q)[None,:,None],(n,n,n)))
    digest=hashlib.sha256(str((rho.shape,rho.dtype.str)).encode()+rho.tobytes()).hexdigest()
    value=json.loads((ROOT/'evaluation_loads/records'/f'{sid}_load4.json').read_text())
    assert digest==value['rho_sha256']
    assert value['load']==[0.,0.,0.,0.,1.,0.]
    expected=np.array([0.,0.,0.,0.,rho.mean(),0.])
    errors=[float(np.linalg.norm(np.array(v['average_stress'])-expected)/np.linalg.norm(expected)) for v in value['records']]
    rows.append(dict(sample_id=sid,rho_sha256=digest,exact_discrete_average_stress=expected.tolist(),
        rhs_norm=value['records'][0]['rhs_norm'],max_absolute_residual=max(v['rhs_norm']*v['relative_residual'] for v in value['records']),
        max_relative_stress_error_against_analytic= max(errors),failed_timed_solves=sum(not v['converged'] for v in value['records'])))
result=dict(purpose=__doc__,geometry_id=g['geometry_id'],layer_axis='y',engineering_shear='xz',
    exact_fluctuation='zero',identity='div(rho(y) * (e_x tensor e_z + e_z tensor e_x)) = 0; mu0=1',
    timing_status='All 42 original failed certifications retained. Analytic diagnosis does not relabel them as successes.',rows=rows)
dest=ROOT/'analysis/zero_load.json';dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(result,indent=2))
