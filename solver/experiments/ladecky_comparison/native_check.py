"""Independent native muSpectre 0.26.3 CPU check; no GPU timing claim."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from itertools import product
import numpy as np
import muSpectre as ms


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    source=json.loads((args.out/'atomic_validation/small_validation.json').read_text())
    rows=[]
    for n in (16,32):
        expected=next(r for r in source['comparisons'] if r['case']=='sphere' and r['grid']==n)
        phase=np.load(args.out/f'atomic_validation/inputs/sphere_{n}.npy')
        ref=np.array(expected['reference']);lam,mu=ref[0,1],ref[3,3]
        tensor=np.zeros((3,3,3,3))
        for i,j,k,l in product(range(3),repeat=4):
            tensor[i,j,k,l]=lam*(i==j)*(k==l)+mu*((i==k)*(j==l)+(i==l)*(j==k))
        reference=tensor.reshape(9,9,order='F')
        CH=np.empty((6,6));counts=[]
        for col in range(6):
            cell=ms.cell.CellData.make([n]*3,[float(n)]*3)
            cell.nb_quad_pts=8
            offsets=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                              [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.int32)
            signs=2*offsets-1;shape=[]
            # Generic public constructor; the packaged Python binding does
            # not expose the C++ trilinear_hexahedron factory.
            for point in product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3):
                gradient=np.empty((3,8))
                for j in range(3):
                    gradient[j]=signs[:,j]/4
                    for k in range(3):
                        if j!=k:gradient[j]*=1+signs[:,k]*point[k]
                shape.append([gradient])
            stencil=ms.FEMStencil(8,1,8,1,shape,[(np.zeros(8,dtype=np.int32),offsets)],[1/8]*8,cell)
            disc=ms.Discretisation(stencil)
            soft=ms.material.MaterialLinearElastic1_3d.make(cell,'soft',3500.,.35)
            hard=ms.material.MaterialLinearElastic1_3d.make(cell,'hard',70000.,.2)
            for pix_id,coords in enumerate(cell.pixels):
                x,y,z=coords
                (hard if phase[z,y,x] else soft).add_pixel(pix_id)
            cg=ms.solvers.KrylovSolverPCG(1e-10,10000,ms.Verbosity.Silent)
            solver=ms.solvers.SolverFEMNewtonPCG(disc,cg,ms.Verbosity.Silent,1e-10,1e-10,20)
            solver.formulation=ms.Formulation.small_strain
            solver.initialise_cell();solver.evaluate_stress_tangent();solver.set_reference_material(reference)
            macro=np.zeros((3,3));i,j=[(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)][col]
            macro[i,j]=1. if i==j else .5;macro[j,i]=macro[i,j]
            result=solver.solve_load_increment(macro)
            mean=np.asarray(result.stress).mean(axis=1).reshape(3,3,order='F')
            CH[:,col]=[mean[0,0],mean[1,1],mean[2,2],mean[1,2],mean[0,2],mean[0,1]]
            counts.append(int(cg.counter))
            print('NATIVE_LOAD',n,col,counts[-1],flush=True)
        target=np.array(expected['results']['literature']['CH'])
        error=float(np.linalg.norm(CH-target)/np.linalg.norm(target))
        assert error<1e-8,(n,error,CH,target)
        rows.append(dict(grid=n,case='sphere',CH=CH.tolist(),CH_relative_error=error,iterations=counts,
                         reference=ref.tolist(),native_CG_tolerance=1e-10))
    result=dict(passed=True,software='native muSpectre',version=importlib.metadata.version('muSpectre'),
                backend='CPU',comparisons=rows,
                note='Independent small-grid verification; stricter native tolerance 1e-10; same reference tensor and physical input; public generic FEMStencil constructor with full HEX8 gradients and equal Gauss weights.')
    (args.out/'native_validation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
