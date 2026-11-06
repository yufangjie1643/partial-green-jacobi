"""Immutable experiment artifacts and input representation for global alpha."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

FAMILIES = ('smooth', 'cosine', 'ramp', 'sharp', 'transition', 'mixed')
SPLITS = {'train': 35, 'validation': 5, 'test': 10}
CONTRASTS = (100, 1000, 10000)
SEED = 20261028
LOG_SCALE = float(np.log(10000.))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, x.dtype.str)).encode() + x.tobytes()).hexdigest()


def atomic(path, writer, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_json(path, obj, replace=False):
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    atomic(path, lambda f: f.write(data), replace)


def save_npz(path, **arrays):
    atomic(path, lambda f: np.savez(f, **arrays))


@contextlib.contextmanager
def lock(out):
    # Kernel releases the lock even after a killed process; no stale-PID bypass.
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.writer.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another writer is using '+str(out)) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def output_path(path):
    from project_paths import ROOT
    out = Path(path).resolve()
    if not out.is_relative_to((ROOT/'runs').resolve()) or out == (ROOT/'runs').resolve():
        raise ValueError('Outputs must be under ${HOME}/code/voxel-mechanics/runs/<experiment>/')
    return out


def encode_numpy(rho):
    n = rho.shape[0]
    if rho.shape != (n, n, n) or n % 8:
        raise ValueError('A cubic grid divisible by 8 is required')
    q = np.log(rho.astype(np.float32))
    return (q.reshape(8, n//8, 8, n//8, 8, n//8).mean((1, 3, 5))
            / np.float32(LOG_SCALE)).reshape(1, 8, 8, 8).astype(np.float32)


def encode_torch(rho):
    n = rho.shape[0]
    q = rho.float().log()
    return (q.reshape(8, n//8, 8, n//8, 8, n//8).mean((1, 3, 5)) / LOG_SCALE).reshape(1, 1, 8, 8, 8)


def select(manifest, args):
    rows = manifest['samples']
    if args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if args.family:
        rows = [r for r in rows if r['family'] == args.family]
    if args.sample_id:
        ids = set(args.sample_id)
        rows = [r for r in rows if r['sample_id'] in ids]
        if {r['sample_id'] for r in rows} != ids:
            raise ValueError('Unknown sample ID or incompatible filters')
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('limit must be positive')
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No samples selected')
    return rows


def filters(parser):
    parser.add_argument('--split', choices=('all', *SPLITS), default='all')
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--sample-id', action='append', help='Repeat to select multiple exact IDs')
    parser.add_argument('--limit', type=int, help='Limit number of fields, never the 101 alpha candidates')
