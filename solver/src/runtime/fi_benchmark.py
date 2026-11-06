"""Same experimental inputs/PCG, full 2x2x2 integration in both FI routes."""
from pathlib import Path
import os,sys,ast
ROOT=Path(__file__).resolve().parent
os.environ['FEAD_SOURCE']=str(ROOT/'fi/source')
sys.path.insert(0,str(ROOT/'fi'))
# The shared input module imports this HG-only helper although FI never calls
# it. Supply its unchanged frozen definition without changing any FI operator.
sys.path.insert(0,str(ROOT/'fi/source'))
import femlib.batched.c3d8r_hourglass_operator as hg_module
if not hasattr(hg_module,'_packed_matvec'):
    source=ROOT/'repro/source/femlib/batched/c3d8r_hourglass_operator.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_packed_matvec')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),vars(hg_module))
import benchmark as b
import atomic_adapter,memory_adapter
from fi_reference import enable_reference as original_reference

def build(data,material,route,precision,stages):
    C=b.construct(data,0,1,material,True,True);stages.end('material')
    voxel=data['domain_size'][0]/b.np.array(data['grid_shape'][::-1])
    def reference(op,*args,**kwargs):
        stages.end('operator',op)
        original_reference(op,precision,'upper6');stages.end('reference',op)
    atomic_adapter.enable_reference=reference;memory_adapter.enable_reference=reference
    solver=atomic_adapter.make(C,data['grid_shape'],'atomic128' if route=='direct' else 'lifetime',voxel)
    op=solver.op;del C,solver;stages.end('representation',op)
    return op
b.build=build
if __name__=='__main__':b.main()
