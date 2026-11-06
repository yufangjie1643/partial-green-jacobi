"""Run the Study I periodic shear design from a fresh checkout."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'exp/asg_gpu_resident_20261103'))
import numpy as np
import torch
from resident import Candidates, Resident, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--methods', nargs='+', default=['0.5'],
                        choices=['green','gj','asg','compact8','0.5','0.69'])
    parser.add_argument('--load-modes', nargs='+', default=['serial'], choices=['serial','parallel'])
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/shear')
    args = parser.parse_args()
    if args.n < 4 or args.n % 2 or not 1 <= args.steps <= 180:
        parser.error('Use an even n >= 4 and 1 <= steps <= 180.')
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU and CUDA-enabled PyTorch are required; use scripts/reproduce_tables.py for CPU analysis.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty to preserve prior results.')
    with pilot.Guard(), torch.inference_mode():
        backend = Candidates()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results = []
            for mode in args.load_modes:
                resident = Resident(backend, args.n, load_mode=mode)
                resident.capture()
                for method in args.methods:
                    resident.outer.set_initial(pilot.initial(args.n, args.seed))
                    resident.solutions.zero_()
                    rows = []
                    stable = 0
                    for k in range(args.steps):
                        beta = pilot.CONFIG['beta_stages'][min(k//25, 3)]
                        start = pilot.sync()
                        resident.outer.evaluate(beta)
                        resident.material(resident.outer.a)
                        resident.select(method)
                        infos, checks = resident.solve()
                        moduli, energy = resident.energy()
                        resident.outer.step(energy)
                        objective = float(moduli.mean())
                        change = float(resident.outer.change)
                        relative = abs(objective-rows[-1]['objective'])/abs(objective) if rows else 1.
                        stable = stable+1 if beta == 8 and k >= 95 and change < .005 and relative < 1e-4 else 0
                        rows.append(dict(step=k, beta=beta, alpha=float(resident.alpha),
                            objective=objective, volume=float(resident.outer.ph.mean()), change=change,
                            step_s=pilot.sync()-start, loads=infos, host_checks=checks))
                        if stable >= 5 or k == args.steps-1:
                            break
                        resident.outer.advance()
                    prefix = args.output/f'{method}_{mode}'
                    np.savez_compressed(prefix.with_name(prefix.name+'.npz'), x=resident.outer.x.cpu().numpy(),
                        physical=resident.outer.ph.cpu().numpy())
                    result = dict(method=method, load_mode=mode, n=args.n, seed=args.seed,
                        optimization_converged=stable >= 5, rows=rows,
                        timing_note='Demonstration timings include first-use compilation; use registered experiment drivers for benchmarks.')
                    # with_suffix would discard decimal exponents in filenames.
                    prefix.with_name(prefix.name+'.json').write_text(json.dumps(result, indent=2)+'\n')
                    results.append(result)
                    print(f'{method:8} {mode:8} J={objective:.10g} volume={rows[-1]["volume"]:.8g} solves={3*len(rows)}', flush=True)
                del resident
            (args.output/'environment.json').write_text(json.dumps(dict(
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                python=sys.version, command=sys.argv), indent=2)+'\n')
    return results


if __name__ == '__main__':
    main()
