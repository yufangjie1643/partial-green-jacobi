"""Repair missing Python conversion includes; no numerical source changes."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request

root=Path(__file__).resolve().parents[2]
out=root/'runs/ladecky_comparison_20261025'
target=root/'baselines/muspectre_0_26_3'
if not target.exists():
    metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/muspectre/0.26.3/json'))
    item=next(f for f in metadata['urls'] if f['packagetype']=='sdist')
    archive=out/'muspectre-0.26.3.tar.gz'
    archive.write_bytes(urllib.request.urlopen(item['url']).read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['digests']['sha256']
    unpack=out/'native_source';unpack.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:stream.extractall(unpack,filter='data')
    candidates=[p for p in unpack.iterdir() if p.is_dir()]
    assert len(candidates)==1
    shutil.copytree(candidates[0],target)
path=target/'language_bindings/python/bind_py_fem_discretisation.cc'
before=path.read_text()
if '#include <pybind11/stl.h>' not in before:
    after=before.replace('#include <pybind11/pybind11.h>',
                         '#include <pybind11/pybind11.h>\n#include <pybind11/eigen.h>\n#include <pybind11/stl.h>')
    path.write_text(after)
    (out/'native_binding_compat.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile='original/bind_py_fem_discretisation.cc',
        tofile='patched/bind_py_fem_discretisation.cc')))
    (out/'native_binding_compat.json').write_text(json.dumps(dict(
        source='PyPI muSpectre 0.26.3 source distribution',file=str(path),
        original_sha256=hashlib.sha256(before.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(after.encode()).hexdigest(),
        change='Add missing pybind11 Eigen and STL converters for public generic FEMStencil constructor; no solver/gradient/material changes',
        compiler_flag='-Wno-template-body (GCC 15 compatibility)'),indent=2))
env=dict(os.environ,CXXFLAGS='-Wno-template-body')
with (out/'native_install_binding.log').open('w') as log:
    # The vendored pybind11 2.10.2 predates NumPy 2 support.
    subprocess.run(['${HOME}/.local/bin/uv','pip','install','--python',str(out/'native_env/bin/python'),
                    'numpy==1.26.4','scipy==1.14.1'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['${HOME}/.local/bin/uv','pip','install','--python',str(out/'native_env/bin/python'),
                    '--reinstall-package','muspectre',str(target),'--config-settings=compile-args=-j2'],
                   env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
