"""Stop only the verified ASG controller and its child to reorder independent stages."""
import os
from pathlib import Path
import signal
import time

pid=865923
path=Path(f'/proc/{pid}')
cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert '${HOME}/code/voxel-mechanics/experiments/asg_independent/finish_pipeline.py' in cmd,cmd
# Freeze the controller before inspecting children so it cannot spawn a new one.
os.kill(pid,signal.SIGSTOP)
children=[int(v) for v in (path/f'task/{pid}/children').read_text().split()]
targets=[]
try:
    for child in children:
        try:child_cmd=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:continue
        assert ('${HOME}/code/voxel-mechanics/' in child_cmd and 'asg-independent' in child_cmd) or 'nvidia-smi' in child_cmd,child_cmd
        targets.append((child,child_cmd))
    for child,child_cmd in targets:
        try:os.kill(child,signal.SIGTERM)
        except ProcessLookupError:pass
    os.kill(pid,signal.SIGTERM)
finally:
    try:os.kill(pid,signal.SIGCONT)
    except ProcessLookupError:pass
time.sleep(2)
for child,child_cmd in targets:
    p=Path(f'/proc/{child}/cmdline')
    if p.exists() and p.read_bytes().replace(b'\0',b' ').decode()==child_cmd:
        os.kill(child,signal.SIGKILL)
print('Stopped only verified ASG controller and children:',pid,targets)
