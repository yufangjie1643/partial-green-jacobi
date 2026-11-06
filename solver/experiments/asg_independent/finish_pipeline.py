"""Continue the already-authorized ASG experiment after its training labels finish."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('${HOME}/code/voxel-mechanics')
OUT=ROOT/'runs/asg_independent_20261030'
LOG=OUT/'logs';LOG.mkdir(exist_ok=True)

def status(stage,**kwargs):
    data=dict(stage=stage,utc_epoch=time.time(),**kwargs)
    p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
    print(json.dumps(data),flush=True)

def gpu_pids():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(v.strip()) for v in out.splitlines() if v.strip().isdigit()}

stages=[('training_labels','asg-independent','label',['--split','train'],OUT/'labels_train_complete.json',True),
        ('validation_labels','asg-independent','label',['--split','validation'],OUT/'labels_validation_complete.json',True),
        ('train','asg-independent-train','train',[],OUT/'TRAINING_COMPLETE.json',False),
        ('calibrate','asg-independent-train','calibrate',[],OUT/'MODELS_FROZEN.json',True),
        ('test_labels','asg-independent','label',['--split','test'],OUT/'labels_test_complete.json',True),
        ('test','asg-independent-evaluate','test',[],OUT/'evaluation_test/COMPLETE.json',True),
        ('loads','asg-independent-evaluate','loads',[],OUT/'evaluation_loads/COMPLETE.json',True),
        ('contrast','asg-independent-evaluate','contrast',[],OUT/'evaluation_contrast/COMPLETE.json',True),
        ('application','asg-independent-evaluate','application',[],OUT/'application/COMPLETE.json',True),
        ('resolution','asg-independent-resume-resolution','resolution',[],OUT/'evaluation_resolution/COMPLETE.json',True)]

for stage,command,mode,extra,marker,gpu in stages:
    if marker.exists():continue
    while True:
        while gpu and gpu_pids():status('waiting_exclusive_gpu',next_stage=stage);time.sleep(15)
        status('running',experiment_stage=stage)
        conflict=None
        with (LOG/f'{stage}.log').open('a') as stream:
            process=subprocess.Popen([str(ROOT/'run'),command,mode,'--out',str(OUT),*extra],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if gpu:
                    other=gpu_pids()-{process.pid}
                    if other:
                        conflict=time.time();status('gpu_interference_rejected',experiment_stage=stage,other_pids=sorted(other))
                        process.terminate()
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        break
                time.sleep(.25)
            rc=process.wait()
        tail=(LOG/f'{stage}.log').read_text(errors='replace')[-3000:]
        if conflict is not None or 'Other GPU compute processes are active:' in tail and rc:
            # Discard only our recent completed records as well as the interrupted
            # in-memory block, conservatively covering the monitoring interval.
            since=(conflict or time.time())-2
            directories=[OUT/'labels'] if 'labels' in stage else ([OUT/'calibration'] if stage=='calibrate' else [OUT/f'evaluation_{stage}'/'records',OUT/'application'/'records'] if stage=='application' else [OUT/f'evaluation_{stage}'/'records'])
            quarantine=OUT/'rejected_interference'/str(time.time_ns());quarantine.mkdir(parents=True)
            for directory in directories:
                for path in directory.glob('*.json'):
                    if path.stat().st_mtime>=since:
                        assert path.resolve().is_relative_to(OUT.resolve())
                        path.rename(quarantine/path.name)
            if marker.exists():marker.rename(quarantine/marker.name)
            time.sleep(5);continue
        if rc:
            status('error',experiment_stage=stage,exit_code=rc,log=str(LOG/f'{stage}.log'))
            raise SystemExit(rc)
        break
status('complete')
