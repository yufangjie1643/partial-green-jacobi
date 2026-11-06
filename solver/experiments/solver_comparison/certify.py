"""Recheck current contrast solutions using a newly built physical material pool.

Usage: python certify.py --case woven|sphere|grosmont --grid 192
Excluded from all timing and memory comparisons.
"""
import argparse
import json
import sys
import numpy as np
import torch
import study as s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('woven','sphere','grosmont'),required=True)
    p.add_argument('--grid',type=int,default=192)
    args=p.parse_args()
    out=s.RUN_DIR/'results'/f'certify_{args.case}_{args.grid}.json'
    if out.exists(): raise FileExistsError(out)
    b,mf=s.configure('latest')
    adapt,digest,metadata=s.inputs(b,mf,args.case,args.grid)
    data,table=adapt()
    op=b.build(data,table,'direct','mixed',b.Stages())
    sys.path.insert(0,str(s.OPERATOR))
    import operator_adapter as oa
    _,hm=oa.prior.material_modules()
    pool,index,_=hm.build_hybrid_material(data['phase_id'],table,orientation=data.get('orientation'))
    physical=oa.prior.HybridFIOperator(pool,index,data['grid_shape'],op._modal_voxel)
    rows=[];columns=[]
    for j,macro in enumerate(np.eye(6)):
        rhs=op.project_zero_mean(op.compute_rhs(macro))
        u,call=b.solve(op,rhs)
        original_rhs=physical.project_zero_mean(physical.compute_rhs(macro))
        norm=torch.linalg.vector_norm(original_rhs).clamp_min(1e-300)
        residual=float(torch.linalg.vector_norm(original_rhs-physical.apply(u))/norm)
        rhs_error=float(torch.linalg.vector_norm(rhs-original_rhs)/norm)
        stress=op.compute_avg_stress(u,macro)
        original_stress=physical.compute_avg_stress(u,macro)
        stress_error=float(np.linalg.norm(stress-original_stress)/np.linalg.norm(original_stress))
        assert residual<=1e-8 and rhs_error<1e-13 and stress_error<1e-13
        rows.append(dict(load=j,physical_operator_true_residual=residual,rhs_relative=rhs_error,stress_relative=stress_error,solve=call))
        columns.append(stress[0])
        del rhs,u,original_rhs
    b.gpu_guard()
    result=dict(passed=True,grid=args.grid,case=args.case,input_hash=digest,input=metadata,
                rows=rows,CH=np.column_stack(columns).tolist(),sources=s.sources(),
                scope='Independent physical coefficient pool and unchanged full-domain hybrid action. Untimed correctness run.')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('CERTIFIED',args.case,args.grid,max(r['physical_operator_true_residual'] for r in rows),flush=True)


if __name__=='__main__':main()
