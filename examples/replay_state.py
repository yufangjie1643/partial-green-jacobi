"""Compare preconditioners on one released Study II state and warm start."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'exp/asg_density_selector_20261105'))
import numpy as np
import torch
from run import PaperBackend, SelectorResident, pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('state', type=Path)
    p.add_argument('--eta', type=float, required=True)
    p.add_argument('--objective', choices=['shear','bulk','auxetic'], default='shear')
    p.add_argument('--load-mode', choices=['serial','parallel'], default='parallel')
    p.add_argument('--methods', nargs='+', default=['gj','eta','compact8'])
    p.add_argument('--output', type=Path, default=ROOT/'outputs/replay.json')
    a = p.parse_args()
    for method in a.methods:
        if method not in ['green','gj','asg','eta','contrast','compact8','compact8q','gray']:
            try:
                if not 0 <= float(method) <= 1: raise ValueError()
            except ValueError:
                p.error('Unknown method or exponent outside [0,1]: '+method)
        if method == 'asg': p.error('Frozen ASG belongs to Study I; use examples/shear_design.py.')
    if a.eta not in [1e-4,1e-6,1e-9]: p.error('The calibrated eta table supports 1e-4, 1e-6 and 1e-9.')
    if a.output.exists(): p.error('Output exists; choose a new path.')
    state = np.load(a.state, allow_pickle=False)
    n = state['a'].shape[0]
    with pilot.Guard(), torch.inference_mode():
        backend = PaperBackend()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            r = SelectorResident(backend,n,a.objective,.2,rmin=3.2,emin=a.eta,load_mode=a.load_mode)
            r.capture()
            r.material(torch.as_tensor(state['a'], device='cuda'))
            r.outer.ph.copy_(torch.as_tensor(state['rho'], device='cuda'))
            u0 = torch.as_tensor(state['u0'], device='cuda')
            records = []
            for method in a.methods:
                # Warm each method, then reset the identical archived initial guess.
                r.select(method); r.solutions.copy_(u0); r.solve()
                r.solutions.copy_(u0); pilot.sync(); start=time.perf_counter()
                r.select(method); loads, checks=r.solve(); elapsed=pilot.sync()-start
                records.append(dict(method=method,alpha=float(r.alpha),analysis_s=elapsed,
                    loads=loads,host_checks=checks,C=r.effective_tensor().cpu().tolist()))
                print(method, elapsed, 's', flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(state=a.state.name,eta=a.eta,load_mode=a.load_mode,
        timing='Selection, preconditioner construction, solve and true residual certification; one timed repetition.',
        records=records),indent=2)+'\n')


if __name__ == '__main__': main()
