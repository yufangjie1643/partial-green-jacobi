"""Post-design six-strain diagnostic; excluded from optimization timings."""
import json
import csv
import numpy as np
import torch
from scenarios import HERE, ROOT, CASES, Candidates, energy_extension, pilot

def main():
    records=[]
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();b.loads=np.eye(6);b.affine.cache_clear();n=64
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ext=energy_extension();affine=b.affine(n)
            for case,spec in CASES.items():
                path=HERE/'results/full'/f'{case}_n64_s20261106_0.5_r0.npz'
                a=np.load(path)['a'];op=b.build(a);b.select(op,'0.5')
                solutions=[];certificates=[]
                for load in b.loads:
                    u,info=b.solve(op,op.compute_rhs(load),rtol=1e-10)
                    assert info['converged'];solutions.append(u);certificates.append(info)
                U=torch.cat(solutions,dim=0)
                diagonal=(ext.evaluate(U,b.packed0,affine,n)*op._material_field.reshape(1,-1)).mean(1)
                C=torch.diag(diagonal)
                for i in range(6):
                    for j in range(i+1,6):
                        energy=ext.evaluate(U[i:i+1]+U[j:j+1],b.packed0,affine[i:i+1]+affine[j:j+1],n)
                        q=(energy*op._material_field.reshape(1,-1)).mean()
                        C[i,j]=C[j,i]=.5*(q-diagonal[i]-diagonal[j])
                C=C.cpu().numpy()
                w,v=np.linalg.eigh(b.C0);inverse=(v/w**.5)@v.T
                eigen=np.linalg.eigvalsh(inverse@C@inverse)
                assert eigen.min()>=1e-4*(1-1e-6)
                response=np.einsum('li,ij,lj->l',spec['loads'],C,spec['loads'])
                solid=np.einsum('li,ij,lj->l',spec['loads'],b.C0,spec['loads'])
                J=float(np.dot(spec['weights'],response/solid))
                optimized=json.loads(path.with_suffix('.json').read_text())['final']['J']
                error=abs(J-optimized)/abs(optimized);assert error<1e-7
                records.append(dict(case=case,n=n,C_effective=C.tolist(),normalized_eigenvalues=eigen.tolist(),
                    bulk_normalized=float(C[:3,:3].sum()/18),axial_zz_normalized=float(C[2,2]/(10/3)),
                    shear_yz_normalized=float(C[3,3]),shear_xz_normalized=float(C[4,4]),shear_xy_normalized=float(C[5,5]),
                    min_normalized_stiffness=float(eigen.min()),max_normalized_stiffness=float(eigen.max()),
                    objective_reconstruction_relative_error=error,loads=certificates))
                print(case,records[-1]['normalized_eigenvalues'],flush=True)
    pilot.save(HERE/'results/constitutive.json',dict(passed=True,post_design_diagnostic=True,
        used_to_select_cases=False,included_in_optimization_timing=False,cases=records))
    fields=['case','n','bulk_normalized','axial_zz_normalized','shear_yz_normalized','shear_xz_normalized','shear_xy_normalized','min_normalized_stiffness','max_normalized_stiffness']
    with (ROOT/'manuscripts/periodic_cell_partial_scaling_20261104/data/scenarios_constitutive.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)

if __name__=='__main__':main()
