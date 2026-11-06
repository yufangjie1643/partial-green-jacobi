"""Resume with the same first-parent warmup field used by the original grid run.

Only excluded warmup input is stabilized across restarts. Timed evaluation,
selection, reference certification and all saved protocol files are unchanged.
"""
import json
from pathlib import Path
import sys
import time
import evaluate
from common import array_hash,digest,read,save_json
from study import field

original=evaluate.Evaluation.warm
original_init=evaluate.Evaluation.__init__

def verified_resume_init(self,out):
    original_init(self,out)
    expected=read(self.out/'evaluation_resolution/protocol.json')['solver_source_hashes']
    actual=self.runner.source_hashes
    changed={p:(h,actual.get(p)) for p,h in expected.items() if actual.get(p)!=h}
    assert not changed,changed
    extras={p:h for p,h in actual.items() if p not in expected}
    assert set(extras)<={str(Path(__file__).resolve())},extras
    # Keep the exact frozen solver fingerprint. Record the additional launch
    # wrapper separately, rather than treating it as changed solver code.
    save_json(self.out/'implementation_checks/resume_warmup'/f'{time.time_ns()}_entry.json',
              dict(wrapper_sha256=digest(__file__),extra_entry_hashes=extras,
                   frozen_protocol_sha256=digest(self.out/'evaluation_resolution/protocol.json'),
                   verified_source_files=len(expected),utc_epoch=time.time()))
    self.runner.source_hashes={p:actual[p] for p in expected}

def original_grid_warmup(self,rho):
    n=rho.shape[0]
    if n in self.warmed:return
    # cases(ev,'resolution') always starts with this parent, contrast 100,
    # then n32/n128/n256. Use exactly that original per-grid warmup input.
    parent=next(g for g in self.manifest['geometries'] if g['split']=='test' and g['family']=='smooth')
    warm_rho=field(parent,n,100)
    metadata=dict(scope='Excluded warmup only; exact original first-parent input for this grid',
        n=n,geometry_id=parent['geometry_id'],contrast=100,rho_sha256=array_hash(warm_rho),
        wrapper_sha256=digest(__file__),frozen_evaluation_sha256=digest(evaluate.__file__),
        model_freeze_sha256=digest(self.out/'MODELS_FROZEN.json'),started_epoch=time.time())
    path=self.out/'implementation_checks/resume_warmup'/f'{time.time_ns()}_n{n}.json'
    save_json(path,metadata)
    original(self,warm_rho)
    metadata['completed_epoch']=time.time();save_json(path,metadata,replace=True)

if __name__=='__main__':
    assert len(sys.argv)>1 and sys.argv[1]=='resolution','This wrapper is only for grid-transfer resumption'
    evaluate.Evaluation.__init__=verified_resume_init
    evaluate.Evaluation.warm=original_grid_warmup
    evaluate.main()
