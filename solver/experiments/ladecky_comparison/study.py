"""Matched all-FP64 literature-form matrix-free and optimized solver comparison."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType
from project_paths import ROOT, source_path


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def setup():
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'experiments/solver_comparison'))
    comp=load('ladecky_comparison',ROOT/'experiments/solver_comparison/study.py')
    b=comp.cs.s.load('fi',configure=False)
    sys.path.insert(0,str(source_path('hybrid_operator')))
    import operator_adapter as oa
    mf,hm=oa.prior.material_modules()
    sys.path.insert(0,str(source_path('spectral')))
    import precision_adapter
    precision_adapter.install(comp.cs,'fp64')
    sys.path.insert(0,str(source_path('ladecky')))
    from ladecky_operator import GaussFIOperator
    from reference_closed_form import design_closed
    b._variant='optimized';b._assembly='gather';b._block=128

    def build(data,materials,route,precision,stages):
        grid=tuple(data['grid_shape']);voxel=data['domain_size'][0]/np.array(grid[::-1])
        phase=data['phase_id'];orientation=data.get('orientation')
        if b._variant=='optimized':
            C,index,storage=hm.build_hybrid_material(phase,materials,orientation=orientation)
            stages.end('material')
            op=oa.prior.HybridFIOperator(C,index,grid,voxel,dense=False)
        else:
            C=mf.build_material_field(phase,materials,orientation=orientation)
            storage=dict(mode='elementwise',coefficient_bytes=C.numel()*C.element_size())
            stages.end('material')
            op=GaussFIOperator(C,grid,voxel,assembly=b._assembly,block=b._block)
        b._storage=storage;stages.end('operator',op)
        initial=op._compute_reference_medium('zeman',verbose=False)[0]
        phases=[mf.material_stiffness(materials[int(k)]) for k in np.unique(phase)]
        reference,design=design_closed(phases,initial,'fi')
        reference=reference/reference[3,3]
        op._compute_reference_medium=MethodType(lambda self,*a,**kw:reference[None].copy(),op)
        comp.cs.s.f.formula.enable_fi_reference(op)
        b._reference_design=dict(reference=reference.tolist(),design=design)
        stages.end('reference',op)
        if b._variant=='optimized':oa.attach(op,data,materials,'split_delta',64)
        b._operator_metadata=op._operator_metadata
        stages.end('representation',op)
        return op

    b.build=build
    # Both solvers receive the same FP64 FFT executor and fused FP64 PCG.
    # This strengthens the comparator and isolates operator/material costs.
    comp.cs.a.configure(b,'fi','combined')
    b.TOL=1e-8
    return b,comp,mf


def inputs(b,comp,mf,case,n,out):
    import numpy as np
    if case=='sphere':
        from cases import prepare
        from legacy_inputs import prepare_inputs
        path=(ROOT/'experiments/solver_comparison/inputs/sphere_192.npy' if n==192
              else prepare('sphere',n,out/'inputs'))
        phase,data,geometry,material,digest,audit=prepare_inputs(path)
        table={label:dict(kind='isotropic',E=m['E'],nu=m['nu'])
               for label,m in zip((0,255),audit['source_metadata']['phase_materials'])}
        return dict(grid_shape=data['grid_shape'],domain_size=data['domain_size'],phase_id=phase[None]),table,digest,geometry
    adapt,digest,meta=comp.inputs(b,mf,case,n)
    data,materials=adapt()
    return data,materials,digest,meta


def event_time(fn,value):
    import torch
    for _ in range(3):fn(value)
    samples=[]
    for _ in range(5):
        a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):fn(value)
        z.record();z.synchronize();samples.append(a.elapsed_time(z)/10)
    return dict(samples_ms=samples,median_ms=statistics.median(samples))


def fingerprint(comp):
    sources=comp.sources()
    for directory in (source_path('ladecky'),Path(__file__).parent):
        for path in directory.glob('*'):
            if path.suffix in ('.py','.cu','.md'):
                sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def worker(a):
    import numpy as np
    import torch
    b,comp,mf=setup();b._variant=a.variant;b._assembly=a.assembly;b._block=a.block
    data,mat,digest,meta=inputs(b,comp,mf,a.case,a.grid,a.out)
    def task():
        start=time.perf_counter();result,_=b.workload(data,mat,'direct','fp64')
        return result,time.perf_counter()-start
    b.gpu_guard();task();b.gpu_guard()
    print('MEASUREMENT_BEGIN',flush=True)
    measured,elapsed=task()
    print('MEASUREMENT_END',flush=True)
    maximum=max(v for row in measured['calls'] for v in row['true_residuals'])
    assert maximum<=1e-8 and np.isfinite(measured['CH']).all()
    b.cleanup();op=b.build(data,mat,'direct','fp64',b.Stages())
    torch.manual_seed(193)
    u=torch.randn((1,op.mesh.n_dofs),device='cuda',dtype=torch.float64)
    micro=dict(Ku=event_time(op.apply,u),preconditioner=event_time(op.apply_precond,u))
    assert op._formula_precision=='fp64'
    native_meta=dict(op._contest_metadata)
    assert native_meta['FFT_precision']=='FP64/complex128'
    report=dict(complete=True,variant=a.variant,case=a.case,grid=a.grid,repeat=a.repeat,
        assembly=a.assembly,block=a.block,input_hash=digest,input=meta,
        measurement=measured,workflow_seconds=elapsed,max_true_residual=maximum,
        microbenchmark=micro,reference_design=b._reference_design,
        storage=b._storage,operator=b._operator_metadata,fft=native_meta,
        sources=fingerprint(comp),environment=dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),
        protocol='Same FP64 physical arithmetic, complex128 FFT, fused FP64 PCG, normalized phase-based reference, six unit engineering strains, true residual <=1e-8. Same-size warmup excluded; full setup, loading, solve, certification, stress recovery and cleanup included. Literature-form independent CUDA realization, not native muSpectre.')
    save(a.out/'results'/f'{a.case}_{a.grid}_{a.variant}_r{a.repeat}.json',report)
    print('PASS',a.case,a.grid,a.variant,elapsed,maximum,flush=True)


def independent_cpu(C,grid,voxel,u,macro=None):
    """Separate dense B-matrix Gauss reference, no modal/source CUDA functions."""
    import numpy as np
    from itertools import product
    nz,ny,nx=grid;n=nz*ny*nx
    offsets=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    z,y,x=np.indices(grid);nodes=[]
    for dx,dy,dz in offsets:nodes.append((((z+dz)%nz*ny+(y+dy)%ny)*nx+(x+dx)%nx).ravel())
    nodes=np.stack(nodes,axis=1);dofs=(nodes[:,:,None]*3+np.arange(3)).reshape(n,24)
    uel=u.reshape(-1)[dofs];forces=np.zeros((n,24));stress=np.zeros((n,6))
    for xi,eta,zeta in product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3):
        B=np.zeros((6,24))
        for a,(sx,sy,sz) in enumerate(2*offsets-1):
            dx=sx*(1+sy*eta)*(1+sz*zeta)/(4*voxel[0])
            dy=sy*(1+sx*xi)*(1+sz*zeta)/(4*voxel[1])
            dz=sz*(1+sx*xi)*(1+sy*eta)/(4*voxel[2])
            B[0,3*a]=dx;B[1,3*a+1]=dy;B[2,3*a+2]=dz
            B[3,3*a+1]=dz;B[3,3*a+2]=dy
            B[4,3*a]=dz;B[4,3*a+2]=dx
            B[5,3*a]=dy;B[5,3*a+1]=dx
        eps=uel@B.T
        if macro is not None:eps+=macro
        sig=np.einsum('nij,nj->ni',C,eps)
        forces+=sig@B*np.prod(voxel)/8;stress+=sig/8
    answer=np.zeros(n*3);np.add.at(answer,dofs.ravel(),forces.ravel())
    return answer[None],stress.mean(0)


def relative(x,y):
    import numpy as np
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-300))


def validate(a):
    import numpy as np
    import torch
    from itertools import product
    b,comp,mf=setup()
    from ladecky_operator import GaussFIOperator
    import operator_adapter as oa
    rng=np.random.default_rng(5025);rows=[]
    pairs=[(i,j) for i in range(6) for j in range(i,6)]
    for grid in [(3,4,5),(4,4,4)]:
        n=int(np.prod(grid));voxel=(.7,1.1,1.4)
        A=rng.normal(size=(n,6,6));C=A@A.transpose(0,2,1)+np.eye(6)[None]*2
        packed=np.stack([C[:,i,j] for i,j in pairs],axis=1)
        gpu=torch.tensor(packed,device='cuda');u=rng.normal(size=(1,n*3))
        v=torch.tensor(u,device='cuda');macro=rng.normal(size=6)
        expected,stress=independent_cpu(C,grid,voxel,u,macro)
        ku,_=independent_cpu(C,grid,voxel,u)
        load,_=independent_cpu(C,grid,voxel,np.zeros_like(u),macro)
        modal=oa.prior.HybridFIOperator(gpu,torch.empty(0,device='cuda',dtype=torch.int32),grid,voxel,dense=True)
        for assembly in ('atomic','gather'):
            op=GaussFIOperator(gpu,grid,voxel,assembly=assembly)
            row=dict(grid=list(grid),assembly=assembly,
                Ku_error=relative(op.apply(v).cpu().numpy(),ku),
                rhs_error=relative(op.compute_rhs(macro).cpu().numpy(),-load),
                stress_error=relative(op.compute_avg_stress(v,macro)[0],stress),
                modal_error=relative(op.apply(v).cpu().numpy(),modal.apply(v).cpu().numpy()))
            assert max(row[k] for k in ('Ku_error','rhs_error','stress_error','modal_error'))<1e-12,row
            rows.append(row)
        # FFT reference vs explicit impulse columns and dense frequency inverse.
        ref=mf.material_stiffness(dict(kind='isotropic',E=5.,nu=.27))
        cp=np.broadcast_to(ref,(n,6,6));kernel=np.empty((3,3,*grid))
        for j in range(3):
            impulse=np.zeros((1,3*n));impulse[0,j]=1
            force,_=independent_cpu(cp,grid,voxel,impulse)
            kernel[:,j]=force.reshape(*grid,3).transpose(3,0,1,2)
        symbol=np.fft.fftn(kernel,axes=(-3,-2,-1)).transpose(2,3,4,0,1)
        inverse=np.zeros_like(symbol);mask=np.ones(grid,bool);mask[0,0,0]=False
        inverse[mask]=np.linalg.inv(symbol[mask])
        rf=u.reshape(*grid,3);rh=np.fft.fftn(rf,axes=(0,1,2))
        expected=np.fft.ifftn(np.einsum('...ij,...j->...i',inverse,rh),axes=(0,1,2)).real.reshape(1,-1)
        op._compute_reference_medium=MethodType(lambda self,*args,**kw:ref[None].copy(),op)
        comp.cs.s.f.formula.enable_fi_reference(op)
        pre_error=relative(op.apply_precond(v).cpu().numpy(),expected)
        assert pre_error<2e-12,pre_error
        rows.append(dict(grid=list(grid),preconditioner_dense_inverse_error=pre_error,includes_nyquist=True))
    save(a.out/'operator_validation.json',dict(passed=True,checks=rows))
    # The same inputs, reference and optimized execution are used on 16/32 cubes.
    comparisons=[]
    for n in (16,32):
        for case in ('sphere','woven'):
            data,mat,digest,meta=inputs(b,comp,mf,case,n,a.out)
            outcomes={};fields={};references={}
            for variant in ('literature','optimized'):
                b._variant=variant;b._assembly=a.assembly;b.cleanup()
                op=b.build(data,mat,'direct','fp64',b.Stages())
                references[variant]=b._reference_design['reference']
                ch=np.empty((6,6));us=[];rhs_list=[];calls=[]
                for j,E in enumerate(np.eye(6)):
                    rhs=op.project_zero_mean(op.compute_rhs(E));u,call=b.solve(op,rhs)
                    ch[:,j]=op.compute_avg_stress(u,E)[0]
                    us.append(u.cpu().numpy());rhs_list.append(rhs.cpu().numpy());calls.append(call)
                outcomes[variant]=dict(CH=ch.tolist(),calls=calls)
                fields[variant]=(np.stack(us),np.stack(rhs_list))
                del op,rhs,u;b.cleanup()
            assert references['literature']==references['optimized']
            ce=relative(np.array(outcomes['optimized']['CH']),np.array(outcomes['literature']['CH']))
            ue=relative(fields['optimized'][0],fields['literature'][0])
            re=relative(fields['optimized'][1],fields['literature'][1])
            assert ce<1e-8 and ue<1e-6 and re<1e-12,(case,n,ce,ue,re)
            row=dict(case=case,grid=n,input_hash=digest,reference=references['literature'],
                     CH_relative_error=ce,displacement_relative_error=ue,rhs_relative_error=re,results=outcomes)
            np.savez_compressed(a.out/f'fields_{case}_{n}.npz',literature_U=fields['literature'][0],
                optimized_U=fields['optimized'][0],literature_rhs=fields['literature'][1],
                optimized_rhs=fields['optimized'][1])
            comparisons.append(row)
            print('SMALL_PASS',case,n,ce,ue,re,flush=True)
    save(a.out/'small_validation.json',dict(passed=True,comparisons=comparisons,sources=fingerprint(comp)))


def tune(a):
    import numpy as np
    import torch
    b,comp,mf=setup();records=[]
    for case in ('sphere','woven'):
        data,mat,_,_=inputs(b,comp,mf,case,128,a.out)
        for assembly in ('atomic','gather'):
            for block in (64,128,256):
                b._variant='literature';b._assembly=assembly;b._block=block;b.cleanup()
                op=b.build(data,mat,'direct','fp64',b.Stages())
                torch.manual_seed(77)
                u=torch.randn((1,op.mesh.n_dofs),device='cuda',dtype=torch.float64)
                timing=event_time(op.apply,u)
                records.append(dict(case=case,grid=128,assembly=assembly,block=block,**timing))
                print('TUNE',records[-1],flush=True)
                del op,u;b.cleanup()
    candidates=[]
    for assembly in ('atomic','gather'):
        for block in (64,128,256):
            values=[r['median_ms'] for r in records if r['assembly']==assembly and r['block']==block]
            candidates.append(dict(assembly=assembly,block=block,score=float(np.prod(values)**.5)))
    best=min(candidates,key=lambda x:x['score'])
    save(a.out/'tuning.json',dict(records=records,selected=best,
         rule='Smallest geometric mean Ku time over both 128^3 cases; frozen before 192^3 measurement'))


def launch(a):
    runner=load('ladecky_nvml',ROOT/'experiments/solver_comparison/run_jobs.py')
    runner.RUN_DIR=a.out;nvml=runner.NVML()
    small=json.loads((a.out/'small_validation.json').read_text());assert small['passed']
    selection=json.loads((a.out/'tuning.json').read_text())['selected']
    if selection['assembly']=='atomic':
        assert json.loads((a.out/'atomic_validation/small_validation.json').read_text())['passed']
    jobs=[]
    for r in range(3):
        for case in ('sphere','woven'):
            for variant in (('literature','optimized') if r%2==0 else ('optimized','literature')):
                name=f'{case}_192_{variant}_r{r}'
                cmd=[str(ROOT/'run'),'ladecky-study','--phase','worker','--case',case,'--grid','192',
                     '--variant',variant,'--repeat',str(r),'--assembly',selection['assembly'],
                     '--block',str(selection['block']),'--out',str(a.out)]
                jobs.append(dict(name=name,arguments=[str(Path(__file__))],command=cmd))
    save(a.out/'plan.json',jobs);done=[]
    for job in jobs:
        done.append(runner.execute(job,nvml,.02))
        (a.out/'status.json').write_text(json.dumps(dict(completed=done,total=len(jobs)),indent=2))
        print('DONE',job['name'],flush=True)
    save(a.out/'complete.json',dict(complete=True,jobs=done))


def summarize(a):
    import numpy as np
    import csv
    assert json.loads((a.out/'complete.json').read_text())['complete']
    rows=[];comparisons=[]
    for case in ('sphere','woven'):
        records={}
        for variant in ('literature','optimized'):
            records[variant]=[json.loads((a.out/'results'/f'{case}_192_{variant}_r{r}.json').read_text()) for r in range(3)]
        all_records=records['literature']+records['optimized']
        assert len({r['input_hash'] for r in all_records})==1
        assert all(r['reference_design']['reference']==all_records[0]['reference_design']['reference'] for r in all_records)
        assert all(r['fft']['FFT_precision']=='FP64/complex128' and r['max_true_residual']<=1e-8 for r in all_records)
        reference=np.array(records['literature'][0]['measurement']['CH'])
        for variant,data in records.items():
            times=[r['workflow_seconds'] for r in data]
            telemetry=[json.loads((a.out/'logs'/f'{case}_192_{variant}_r{r}_telemetry.json').read_text()) for r in range(3)]
            assert all(t['returncode']==0 and t['failure'] is None and t['measurement_sample_count']>0 for t in telemetry)
            peaks=[t['peak_process_bytes_measured']/2**30 for t in telemetry]
            iterations=[[c['iterations'] for c in r['measurement']['calls']] for r in data]
            ch_errors=[relative(np.array(r['measurement']['CH']),reference) for r in data]
            row=dict(case=case,variant=variant,time_median_s=statistics.median(times),
                time_min_s=min(times),time_max_s=max(times),time_samples_s=times,
                nvml_peak_GiB=max(peaks),nvml_samples_GiB=peaks,
                iteration_totals=[sum(it) for it in iterations],iterations=iterations,
                max_true_residual=max(r['max_true_residual'] for r in data),
                max_CH_relative_error=max(ch_errors),
                Ku_median_ms=statistics.median(r['microbenchmark']['Ku']['median_ms'] for r in data),
                FFT_median_ms=statistics.median(r['microbenchmark']['preconditioner']['median_ms'] for r in data))
            for group,predicate in [('solve',lambda s:s.startswith('solve')),
                                    ('rhs',lambda s:s.startswith('rhs')),
                                    ('post',lambda s:s.startswith('post')),
                                    ('setup',lambda s:not s.startswith(('rhs','solve','post')))]:
                row[group+'_median_s']=statistics.median(sum(s['seconds'] for s in r['measurement']['stages'] if predicate(s['name'])) for r in data)
            rows.append(row)
        old,new=rows[-2:]
        assert new['max_CH_relative_error']<1e-8
        comparisons.append(dict(case=case,time_reduction_percent=100*(1-new['time_median_s']/old['time_median_s']),
            memory_reduction_percent=100*(1-new['nvml_peak_GiB']/old['nvml_peak_GiB']),
            speedup=old['time_median_s']/new['time_median_s'],
            operator_speedup=old['Ku_median_ms']/new['Ku_median_ms'],
            equal_iterations=old['iterations']==new['iterations'],
            max_CH_relative_error=new['max_CH_relative_error']))
    save(a.out/'summary.json',dict(rows=rows,comparisons=comparisons))
    columns=['case','variant','time_median_s','time_min_s','time_max_s','nvml_peak_GiB',
             'max_true_residual','max_CH_relative_error','Ku_median_ms','FFT_median_ms',
             'setup_median_s','rhs_median_s','solve_median_s','post_median_s']
    with (a.out/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=rows,comparisons=comparisons),indent=2))


def native(a):
    import subprocess
    subprocess.run([str(a.out/'native_env/bin/python'),str(Path(__file__).with_name('native_check.py')),
                    '--out',str(a.out)],check=True)


def native_setup(a):
    import runpy
    runpy.run_path(str(Path(__file__).with_name('native_setup.py')),run_name='__main__')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validate','tune','worker','launch','summarize','native','native_setup'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=('literature','optimized'),default='literature')
    p.add_argument('--assembly',choices=('gather','atomic'),default='gather')
    p.add_argument('--block',type=int,default=128)
    p.add_argument('--grid',type=int,default=192)
    p.add_argument('--case',choices=('sphere','woven'),default='sphere')
    p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.is_relative_to(ROOT/'runs'):p.error('Use project runs directory')
    a.out.mkdir(parents=True,exist_ok=True);os.environ['VOXEL_RUN_DIR']=str(a.out)
    globals()[a.phase](a)


if __name__=='__main__':main()
