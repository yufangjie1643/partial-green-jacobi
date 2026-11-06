"""Download a versioned data asset, verify every checksum, and safely extract it."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def destination(name, root=ROOT):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.parts[0] not in ['exp','data']:
        raise ValueError('Unsafe archive path: '+name)
    target=(root/str(path)).resolve()
    target.relative_to(root.resolve())
    return target


def extract(archive, root=ROOT):
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if any(not m.isfile() for m in members):
            raise ValueError('Only regular files are accepted.')
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries.')
        manifests=[m for m in members if m.name.startswith('data/manifests/') and m.name.endswith('.json')]
        if len(manifests)!=1:raise ValueError('Expected one per-file manifest.')
        expected={r['path']:r for r in json.load(tar.extractfile(manifests[0]))['files']}
        if set(names)!=(set(expected)|{manifests[0].name}):raise ValueError('Manifest coverage mismatch.')
        # Validate contents and existing destinations before writing any file.
        for m in members:
            target=destination(m.name,root)
            digest=hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()
            if m.name in expected and (digest!=expected[m.name]['sha256'] or m.size!=expected[m.name]['bytes']):
                raise ValueError('Content checksum mismatch: '+m.name)
            if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=digest:
                raise FileExistsError('Existing file differs; use a clean checkout: '+str(target))
        for m in members:
            target=destination(m.name,root)
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.partial')
            with tar.extractfile(m) as source,temporary.open('wb') as output:
                shutil.copyfileobj(source,output)
            temporary.replace(target)
    return len(members)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('assets',nargs='*',help='Asset names without .tar.gz, or all')
    p.add_argument('--list',action='store_true')
    p.add_argument('--archive-dir',type=Path,default=ROOT/'downloads')
    args=p.parse_args();metadata=json.loads((ROOT/'data/releases.json').read_text())
    assets={a['name'].removesuffix('.tar.gz'):a for a in metadata['assets']}
    if args.list or not args.assets:
        for name,a in assets.items():print(f"{name:25} {a['bytes']/2**20:8.1f} MiB, {a['files']} files, {a['npz_files']} array archives")
        return
    names=list(assets) if args.assets==['all'] else args.assets
    if any(n not in assets for n in names):p.error('Unknown asset; use --list.')
    args.archive_dir.mkdir(parents=True,exist_ok=True)
    for name in names:
        a=assets[name];path=args.archive_dir/a['name']
        if not path.exists():
            part=path.with_name(path.name+'.partial')
            print('Downloading',a['url'],flush=True)
            request=urllib.request.Request(a['url'],headers={'User-Agent':'partial-green-jacobi/0.1.0'})
            with urllib.request.urlopen(request,timeout=60) as src,part.open('wb') as dst:
                shutil.copyfileobj(src,dst,length=2**20)
            part.replace(path)
        if path.stat().st_size!=a['bytes'] or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=a['sha256']:
            raise ValueError('Asset checksum mismatch: '+str(path))
        print('Verified and extracted',extract(path),'files from',path.name,flush=True)


if __name__=='__main__':main()
