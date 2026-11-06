"""Use the CPU-training interval for a validation-only GPU memory preflight."""
import json
from pathlib import Path
import subprocess
import time

root=Path('${HOME}/code/voxel-mechanics')
out=root/'runs/asg_independent_20261030'
while True:
    status=json.loads((out/'pipeline_status.json').read_text())
    if status.get('experiment_stage')=='train' and status['stage']=='running':break
    if status['stage'] in ['error','complete'] or status.get('experiment_stage') in ['calibrate','test_labels','test']:raise SystemExit('CPU training window passed; preflight was not launched')
    time.sleep(3)
with (out/'logs/resolution_preflight.log').open('w') as log:
    command=['${HOME}/uv-torch-py312/bin/python',str(root/'experiments/asg_independent/check_resolution.py')]
    import os
    env=dict(os.environ,PYTHONPATH=str(root))
    result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
print('Resolution preflight return code:',result.returncode,flush=True)
