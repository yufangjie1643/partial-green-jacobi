"""Plan, generate and audit 900 grouped material fields for an 8^3-input CNN."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm
from common import (FAMILIES, SPLITS, CONTRASTS, SEED, array_hash, digest, encode_numpy,
                    filters, lock, object_hash, output_path, read, save_json, save_npz, select)

VERSION = 'global-alpha8-v1'
# Primitive integer directions preserve periodicity on the unit cube.
DIRECTIONS = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,-1,0), (1,0,1),
              (1,0,-1), (0,1,1), (0,1,-1), (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)]


def lhs(rng, count, dimension):
    return np.stack([(rng.permutation(count)+rng.random(count))/count for _ in range(dimension)], 1)


def make_manifest():
    geometries, samples = [], []
    for fi, family in enumerate(FAMILIES):
        index = 0
        for si, (split, count) in enumerate(SPLITS.items()):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, fi, si]))
            coordinates = lhs(rng, count, 17)
            for u in coordinates:
                weights = .2 + u[6:9]
                weights /= weights.sum()
                params = dict(frequencies=(1+np.floor(3*u[:3]).astype(int)).tolist(),
                              phases=(2*np.pi*u[3:6]).tolist(), weights=weights.tolist(),
                              volume_fraction=float(.15+.7*u[9]),
                              transition_width_parameter=float(np.exp(np.log(.015)+u[10]*np.log(.12/.015))),
                              mixture=float(.2+.65*u[11]),
                              cosine_power=float(.5+3.5*u[12]),
                              direction=list(DIRECTIONS[int(13*u[13])]),
                              levels=int((2,4,8,16)[int(4*u[14])]),
                              coordinate_warp=float(.6+1.8*u[15]),
                              profile_power=float(.6+1.8*u[16]))
                gid = f'{family}_g{index:03d}'
                # Only used non-phase parameters enter the translation-group signature.
                keys = {'smooth': ('frequencies','weights'), 'cosine': ('frequencies','cosine_power'),
                        'ramp': ('direction','levels','coordinate_warp','profile_power'),
                        'sharp': ('frequencies','weights','volume_fraction'),
                        'transition': ('frequencies','weights','volume_fraction','transition_width_parameter'),
                        'mixed': ('frequencies','weights','volume_fraction','mixture')}[family]
                shape_signature = object_hash([family, {k:params[k] for k in keys}])
                geometries.append(dict(geometry_id=gid, family=family, split=split, parameters=params,
                                       shape_signature=shape_signature))
                for contrast in CONTRASTS:
                    samples.append(dict(sample_id=f'{gid}_c{contrast}_n64', geometry_id=gid,
                                        family=family, split=split, n=64, contrast=contrast))
                index += 1
    c0 = np.zeros((6,6)); c0[:3,:3] = 4/3
    c0[np.arange(3),np.arange(3)] += 2
    c0[np.arange(3,6),np.arange(3,6)] = 1
    result = dict(version=VERSION, seed=SEED, original_grid=[64]*3, cnn_input_shape=[1,8,8,8],
                  rho_dtype='float64', input_dtype='float32', contrasts=list(CONTRASTS),
                  encoding='FP32 log(rho), 8^3 equal-block mean, divide by fixed log(10000)',
                  boundary='periodic unit cube', C0=c0.tolist(),
                  macro_strain=[1.,1.,0.,0.,0.,float(np.sqrt(2))],
                  geometry_counts_per_family=SPLITS, geometries=geometries, samples=samples,
                  source_hashes={p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name('common.py'))})
    validate_manifest(result)
    return result


def validate_manifest(m):
    assert len(m['geometries']) == 300 and len(m['samples']) == 900
    assert Counter(r['split'] for r in m['samples']) == {'train':630, 'validation':90, 'test':180}
    geometries = {r['geometry_id']:r for r in m['geometries']}
    assert len(geometries) == 300 and len({r['sample_id'] for r in m['samples']}) == 900
    assert len({r['shape_signature'] for r in geometries.values()}) == 300
    for family in FAMILIES:
        assert Counter(g['split'] for g in geometries.values() if g['family']==family) == SPLITS
    for r in m['samples']:
        g = geometries[r['geometry_id']]
        assert (g['split'],g['family']) == (r['split'],r['family'])
        assert r['n'] == 64 and r['contrast'] in CONTRASTS
    for gid in geometries:
        assert sorted(r['contrast'] for r in m['samples'] if r['geometry_id']==gid) == list(CONTRASTS)


def shape_field(g, n=64):
    p = g['parameters']; family = g['family']
    t = np.arange(n, dtype=np.float64)/n
    xyz = (t[None,None,:], t[None,:,None], t[:,None,None])
    waves = [np.sin(2*np.pi*p['frequencies'][i]*v+p['phases'][i]) for i,v in enumerate(xyz)]
    smooth = sum(w*v for w,v in zip(p['weights'],waves))
    if family == 'smooth':
        q = smooth
    elif family == 'cosine':
        q = ((.5+.5*waves[0])*(.5+.5*waves[1])*(.5+.5*waves[2]))**p['cosine_power']
    elif family == 'ramp':
        coordinate = sum(a*v for a,v in zip(p['direction'],xyz))
        coordinate = np.mod(coordinate+p['phases'][0]/(2*np.pi), 1)**p['coordinate_warp']
        q = (np.minimum(np.floor(coordinate*p['levels']),p['levels']-1)/(p['levels']-1))**p['profile_power']
    elif family in ('sharp','transition'):
        threshold = np.quantile(smooth, 1-p['volume_fraction'])
        level = smooth-threshold
        q = (level>0).astype(float) if family == 'sharp' else .5*(1+np.tanh(level/(2*np.pi*p['transition_width_parameter'])))
    elif family == 'mixed':
        product = waves[0]*waves[1]*waves[2]
        phase = product>np.quantile(product, 1-p['volume_fraction'])
        q = p['mixture']*phase+(1-p['mixture'])*.5*(smooth+1)
    else:
        raise ValueError(family)
    q = np.broadcast_to(q, (n,n,n)).copy()
    span = float(q.max()-q.min())
    if not np.isfinite(span) or span <= 1e-12:
        raise ValueError('Degenerate material geometry '+g['geometry_id'])
    return np.ascontiguousarray((q-q.min())/span)


def generate_one(out, m, row):
    g = next(g for g in m['geometries'] if g['geometry_id']==row['geometry_id'])
    q = shape_field(g, row['n'])
    rho = np.exp(np.log(row['contrast'])*q)
    x = encode_numpy(rho)
    path = out/'fields'/f"{row['sample_id']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if not (np.array_equal(old['rho'],rho) and np.array_equal(old['x'],x)):
                raise RuntimeError('Existing material differs; use a new run directory: '+str(path))
    else:
        save_npz(path, rho=rho, x=x)
    meta = dict(**row, manifest_sha256=digest(out/'manifest.json'), rho_sha256=array_hash(rho),
                input_sha256=array_hash(x), shape_sha256=array_hash(q), archive_sha256=digest(path),
                rho_min=float(rho.min()), rho_max=float(rho.max()), actual_contrast=float(rho.max()/rho.min()),
                input_shape=list(x.shape))
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        if read(sidecar) != meta:
            raise RuntimeError('Metadata mismatch '+str(sidecar))
    else:
        save_json(sidecar, meta)
    return meta


def load_field(out, m, row):
    path = out/'fields'/f"{row['sample_id']}.npz"
    meta = read(path.with_suffix('.json'))
    assert meta['sample_id']==row['sample_id'] and meta['manifest_sha256']==digest(out/'manifest.json')
    assert meta['archive_sha256']==digest(path)
    with np.load(path, allow_pickle=False) as d:
        rho, x = d['rho'].copy(), d['x'].copy()
    assert rho.shape==(64,64,64) and rho.dtype==np.float64
    assert x.shape==(1,8,8,8) and x.dtype==np.float32
    assert array_hash(rho)==meta['rho_sha256'] and array_hash(x)==meta['input_sha256']
    assert np.isfinite(rho).all() and rho.min()>0 and np.isfinite(x).all()
    assert np.array_equal(x,encode_numpy(rho))
    assert abs(rho.max()/rho.min()/row['contrast']-1)<1e-12
    return rho, x, meta


def field_task(out, m, row, mode):
    if mode == 'generate':
        return row, generate_one(out, m, row)
    if not (out/'fields'/f"{row['sample_id']}.json").exists():
        return row, None
    return row, load_field(out, m, row)[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan','generate','audit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true', help='Audit fails unless all selected fields exist')
    parser.add_argument('--workers', type=int, default=1, help='CPU generation/audit processes')
    filters(parser)
    args = parser.parse_args(); out = output_path(args.out)
    if args.workers < 1:
        parser.error('workers must be positive')
    with lock(out):
        expected = make_manifest()
        if not (out/'manifest.json').exists():
            if args.mode != 'plan':
                raise RuntimeError('Run plan first')
            save_json(out/'manifest.json', expected)
        m = read(out/'manifest.json')
        if m != expected:
            raise RuntimeError('Manifest or generator source changed; use a new run directory')
        rows = select(m,args)
        if args.mode == 'plan':
            print('PLAN: 300 geometries, 900 fields; train=630 validation=90 test=180',flush=True)
            return
        complete, missing, signatures = [], [], {}
        # Importable module functions also work under the project's runpy launcher.
        from dataset import field_task as task
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(task,out,m,row,args.mode) for row in rows]
            with tqdm(total=len(rows), desc=args.mode, unit='field', dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    row, meta = future.result()
                    if meta is None:
                        missing.append(row['sample_id'])
                    else:
                        previous = signatures.get(meta['shape_sha256'])
                        if previous is not None and previous != row['geometry_id']:
                            raise RuntimeError('Duplicate discretized geometry in different groups')
                        signatures[meta['shape_sha256']] = row['geometry_id']
                        complete.append(row['sample_id'])
                    bar.set_postfix_str(row['sample_id'], refresh=False)
                    bar.update()
        status = dict(selected=len(rows), complete=len(complete), missing=missing,
                      workers=args.workers,
                      complete_by_split=dict(Counter(r['split'] for r in rows if r['sample_id'] in complete)))
        save_json(out/f'{args.mode}_status.json', status, replace=True)
        if args.require_all and missing:
            raise RuntimeError(f'{len(missing)} material fields are missing')
        print(status,flush=True)


if __name__ == '__main__':
    main()
