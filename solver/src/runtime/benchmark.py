"""Matched rerun: frozen physics, explicit common residual tolerance."""
from pathlib import Path
import os,sys,ast,argparse,gc,copy,hashlib,json,time
ROOT=Path(__file__).resolve().parent
TOL=float(os.environ['FEAD_TOL'])
assert 0<TOL<1
sys.path.insert(0,str(ROOT/'repro/extension_20261006'))
from material_control import *
from mixed_preconditioner import enable_reference
from femlib.batched.c3d8r_hourglass_operator import BatchedC3D8RHourglassTwoStageOperator as HG
from femlib.geometry import LOCAL_OFFSET
from femlib.c3d8r import hourglass_basis
from torch.utils.cpp_extension import load

# Extract the unchanged archived geometry function, without executing its CLI.
gs=ROOT/'repro/extension_20261006/run_resolution.py'
tree=ast.parse(gs.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='geometry')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(gs),'exec'))
OUT=ROOT/'evidence/hg_representation'
ss=(OUT/'assembly.py').read_text()
exec(compile(ss,str(OUT/'assembly.py'),'exec'))

def cleanup():
    gc.collect();torch.cuda.empty_cache();synchronize()

def inputs(N,case):
    data,geom=geometry(N);material=copy.deepcopy(STRONG)
    if case=='near':material=copy.deepcopy(NEAR)
    elif case.startswith('contrast'):
        factor=float(case.removeprefix('contrast'))
        for k in ['E11','E22','E33','G12','G13','G23']:material['yarn'][k]*=factor
    elif case=='aligned':
        mask=data['yarn_id']>=0;data['orient1'][mask]=[1,0,0];data['orient2'][mask]=[0,1,0]
    elif case!='strong':raise ValueError(case)
    digest=hashlib.sha256()
    for key in ['yarn_id','orient1','orient2']:digest.update(np.ascontiguousarray(data[key]).tobytes())
    return data,geom,material,digest.hexdigest()

class Stages:
    def __init__(self):self.rows=[];self.begin()
    def begin(self):synchronize();torch.cuda.reset_peak_memory_stats();self.t=time.perf_counter()
    def end(self,name,op=None):
        synchronize();r=dict(name=name,seconds=time.perf_counter()-self.t,live=torch.cuda.memory_allocated(),peak=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        if op is not None:
            r['arrays']={k:dict(bytes=v.numel()*v.element_size(),shape=list(v.shape),dtype=str(v.dtype),storage=v.untyped_storage().data_ptr()) for k,v in vars(op).items() if isinstance(v,torch.Tensor)}
        self.rows.append(r);self.begin()

def build(data,material,route,precision,stages):
    C=construct(data,0,1,material,True,True);stages.end('material')
    voxel=data['domain_size'][0]/np.array(data['grid_shape'][::-1])
    op=make_solver(C,data['grid_shape'],voxel,fft=False).op;del C
    stages.end('operator',op)
    enable_reference(op,2,precision);stages.end('reference',op)
    if route=='tap14':op.S_hg=assemble(op,14);op.__class__=HGStencil
    elif route in ('orientation','matched21'):
        from orientation_adapter import attach
        attach(op,data,material,route)
    elif route!='direct':raise ValueError(route)
    stages.end('representation',op)
    return op

def trace_solver(op,rhs):
    """Instrument the frozen solver at a host check; no changes to updates/stopping."""
    import inspect
    from femlib.batched import pcg
    source=inspect.getsource(pcg.solve_pcg_batched)
    source=source.replace('        if check_now:\n','        if check_now:\n            _record(it + 1, u, rhs, op)\n')
    events=[];start=time.perf_counter()
    def record(it,u,rhs,op):
        synchronize();elapsed=time.perf_counter()-start
        rr=torch.linalg.vector_norm(rhs-op.apply(u),dim=1)/torch.linalg.vector_norm(rhs,dim=1).clamp_min(1e-300)
        events.append(dict(iteration=it,elapsed_seconds=elapsed,true_residual=float(rr.max())))
    context=dict(vars(pcg));context['_record']=record
    exec(compile(source,'instrumented_frozen_pcg','exec'),context)
    return context['solve_pcg_batched'],events

def solve(op,rhs,trace=False):
    norm=torch.linalg.vector_norm(rhs,dim=1)
    if float(norm.max())==0:return torch.zeros_like(rhs),dict(iterations=0,restarts=0,true_residuals=[0.],trace=[])
    fn,events=trace_solver(op,rhs) if trace else (solve_pcg_batched,[])
    u=None;total=0;restarts=0
    while total<10000:
        u,count,_=fn(op,rhs,tol=TOL,max_iter=10000-total,x0=u,update_mode='compiled',convergence_check_stride=4,verify_final_residual=True)
        total+=int(count);u=op.project_zero_mean(u)
        rr=torch.linalg.vector_norm(rhs-op.apply(u),dim=1)/norm
        if not bool(torch.isfinite(rr).all()):raise RuntimeError('Nonfinite true residual')
        if float(rr.max())<=TOL:return u,dict(iterations=total,restarts=restarts,true_residuals=rr.cpu().tolist(),trace=events)
        if not count:break
        restarts+=1
    raise RuntimeError(f'Unconverged at {total}: {rr.tolist()}')

def workload(data,material,route,precision,trace=False,fields=False):
    cleanup();st=Stages();start=time.perf_counter();op=build(data,material,route,precision,st)
    ch=np.empty((6,6));calls=[];slices={};field_stats={}
    for j in range(6):
        e=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(e));st.end(f'rhs{j}',op)
        u,r=solve(op,rhs,trace=trace and j==0);st.end(f'solve{j}',op)
        ch[:,j]=op.compute_avg_stress(u,e)[0];calls.append(r);st.end(f'post{j}',op)
        if fields and j in (0,5):
            strain,stress=op.recover_center_fields(u,e)
            energy=.5*(strain*stress).sum(-1);N=data['grid_shape'][0]
            y=torch.as_tensor(data['yarn_id'].reshape(1,-1)>=0,device='cuda')
            field_stats[str(j)]={'mean_energy':float(energy.mean()),'yarn_mean_energy':float(energy[y].mean()),'matrix_mean_energy':float(energy[~y].mean()),'stress_l2':float(torch.linalg.vector_norm(stress))}
            slices[f'energy_load{j}']=energy.reshape(N,N,N)[N//2].cpu().numpy()
            slices[f'stress11_load{j}']=stress.reshape(N,N,N,6)[N//2,:,:,0].cpu().numpy()
            del strain,stress,energy,y
            st.end(f'fields{j}',op)
        del rhs,u
    synchronize();seconds=time.perf_counter()-start
    result=dict(seconds=seconds,CH=ch.tolist(),calls=calls,stages=st.rows,field_statistics=field_stats,
        peak_allocated_bytes=max(r['peak'] for r in st.rows),peak_reserved_bytes=max(r['reserved'] for r in st.rows))
    del op;cleanup();return result,slices

def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,required=True);p.add_argument('--route',choices=['direct','tap14','orientation','matched21'],default='direct')
    p.add_argument('--precision',choices=['fp64','mixed'],default='fp64');p.add_argument('--case',default='strong');p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--trace',action='store_true');p.add_argument('--fields',action='store_true');p.add_argument('--tag',default='main');a=p.parse_args()
    stem=f'{a.tag}_{a.case}_{a.grid}_{a.route}_{a.precision}_{a.repeat}';out=ROOT/'results'/(stem+'.json')
    if out.exists():raise FileExistsError(out)
    gpu_guard();torch.set_num_threads(1)
    if a.route=='tap14':extensions()
    if a.route in ('orientation','matched21'):
        from orientation_adapter import extension
        extension()
    data,geom,mat,digest=inputs(a.grid,a.case)
    record=dict(arguments=vars(a),tolerance=TOL,residual_definition='norm(f-Ku,2)/norm(f,2), recomputed in FP64',geometry=geom,material=mat,input_hash=digest,source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.rglob('*')) if f.suffix in ('.py','.cu','.cpp','.h')},
        environment=dict(gpu=gpu_guard(),torch=torch.__version__,cuda=torch.version.cuda),
        protocol=f'one complete same-size warmup; fresh process per repetition; six loads; true FP64 residual <={TOL}; synchronized phase boundaries included; geometry generation/disk/JIT excluded; traces and fields are separate diagnostics')
    try:
        record['attempt_phase']='warmup'
        workload(data,mat,a.route,a.precision)
        record['attempt_phase']='measurement'
        gpu_guard();r,slices=workload(data,mat,a.route,a.precision,a.trace,a.fields)
        record.update(status='passed',measurement=r)
        if slices:np.savez_compressed(ROOT/'results'/(stem+'_fields.npz'),**slices)
    except torch.OutOfMemoryError as e:record.update(status='OOM',error=str(e),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:
        record.update(status='failed',error=repr(e));save_json(out,record);raise
    save_json(out,record);print('RESULT',stem,record['status'],record.get('measurement',{}).get('seconds'),flush=True)
if __name__=='__main__':main()
