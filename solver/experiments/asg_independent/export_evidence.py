"""Export records, weights and source; omit regenerable bulky coefficient arrays."""
from pathlib import Path
import tarfile
import shutil
import hashlib
import json

root=Path('${HOME}/code/voxel-mechanics/runs/asg_independent_20261030')
shutil.copytree(root.parents[1]/'experiments/asg_independent',root/'source/experiments/asg_independent',dirs_exist_ok=True)
law=root.parents[1]/'experiments/global_alpha_compact_law/law.py'
expected=json.loads((root/'evaluation_test/protocol.json').read_text())['selector_hashes']['compact_law']
assert hashlib.sha256(law.read_bytes()).hexdigest()==expected
dest=root/'source/experiments/global_alpha_compact_law/law.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(law,dest)
for name in ['run','run.py','project_paths.py']:
    source=root.parents[1]/name
    if source.exists():shutil.copy2(source,root/'source'/name)
target=root/'evidence_snapshot.tar.gz'
paths=[p for p in root.rglob('*') if p.is_file() and p!=target and p.suffix not in ['.npz','.pyc','.gz','.tmp']
       and '__pycache__' not in p.parts and 'rejected_interference' not in p.parts and p.name!='.lock']
with tarfile.open(target,'w:gz',compresslevel=3) as archive:
    for path in paths:archive.add(path,arcname=path.relative_to(root),recursive=False)
print(target,len(paths),'files',target.stat().st_size,'bytes')
