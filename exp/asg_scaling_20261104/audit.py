"""Audit completed numerical records, timing closure and frozen dependencies."""
from pathlib import Path
import hashlib,json,math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    assert json.loads((OUT/'BATCH_COMPLETE.json').read_text())['completed']
    frozen=json.loads((OUT/'frozen_sources.json').read_text())
    for path,digest in frozen.items():assert sha(ROOT/path)==digest,path
    old=ROOT/'exp/asg_gpu_resident_20261103'
    archived=json.loads((old/'COMPLETION.json').read_text())['sha256']
    for path,digest in archived.items():assert sha(old/path)==digest,path
    fulls=sorted((OUT/'full').glob('*.json'));replays=sorted((OUT/'replay').glob('*.json'))
    assert len(fulls)==14 and len(replays)==12
    full_loads=replay_loads=0;max_ratio=0.;max_closure=0.;max_J_error=0.;min_host=math.inf
    errors=[]
    stages=['filter_s','material_s','selection_precond_s','solve_s','response_s','outer_update_s','host_and_reporting_s']
    for path in fulls:
        d=json.loads(path.read_text());assert len(d['rows'])==d['steps']
        closure=abs(d['setup_s']+sum(r['step_s'] for r in d['rows'])-d['total_s'])
        assert closure<1e-9
        if d.get('validation'):errors.append(d['validation'])
        for r in d['rows']:
            closure=abs(sum(r[k] for k in stages)-r['step_s']);max_closure=max(max_closure,closure)
            assert closure<1e-9
            min_host=min(min_host,r['host_and_reporting_s'])
            for load in r['loads']:
                assert load['converged'] and load['absolute_residual']<=load['threshold']
                max_ratio=max(max_ratio,load['absolute_residual']/load['threshold']);full_loads+=1
    assert min_host>=-1e-6,min_host
    for path in replays:
        d=json.loads(path.read_text());assert d['reference']['converged'] and len(d['rows'])==18 and len(d['outer'])==3
        assert sha(Path(d['source']))==d['source_sha256']
        for r in d['rows']:
            max_J_error=max(max_J_error,r['relative_J_error']);assert r['relative_J_error']<1e-7
            assert abs(sum(r[k] for k in ['material_s','selection_precond_s','solve_s','response_s'])-r['total_s'])<1e-9
            for load in r['loads']:
                assert load['converged'] and load['absolute_residual']<=load['threshold']
                max_ratio=max(max_ratio,load['absolute_residual']/load['threshold']);replay_loads+=1
        assert max(r['max_density_difference'] for r in d['outer'])<2e-8
    for n in (32,64,128):
        for mode in ('serial','parallel'):
            d=json.loads((OUT/f'memory_n{n}_{mode}.json').read_text())
            assert 0<d['peak_allocated_bytes']<=d['peak_reserved_bytes']
            assert d['sampled_device_peak_bytes']>d['peak_allocated_bytes']
    summary=dict(validation_passed=True,frozen_sources_verified=len(frozen),previous_records_verified=len(archived),
        new_full_optimization_runs=len(fulls),new_full_load_solves=full_loads,replay_three_load_cases=216,replay_load_solves=replay_loads,
        maximum_residual_over_threshold=max_ratio,maximum_replay_relative_J_error=max_J_error,
        maximum_stage_closure_error_s=max_closure,minimum_host_and_reporting_s=min_host,
        worst_full_consistency={k:max(e[k] for e in errors) for k in errors[0]},
        scope='64^3: A/B, serial/parallel, three repeats. 128^3: A, serial/parallel, one run each. Original outer stopping rules retained; 128^3 is not a grid-convergence study.',
        sha256={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md') and p.name!='COMPLETION.json'})
    (HERE/'COMPLETION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
