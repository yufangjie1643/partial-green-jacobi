"""One GPU process at a time, retrying only externally interfered timings."""
from pathlib import Path
import subprocess,sys,time,json
HERE=Path(__file__).resolve().parent
OUT=HERE/'results';(OUT/'logs').mkdir(parents=True,exist_ok=True)
commands=[['full','--n','64','--mode','parallel'],['full','--n','64','--mode','serial'],
          ['full','--n','128','--mode','parallel','--seeds','20261104','--repetitions','1'],
          ['full','--n','128','--mode','serial','--seeds','20261104','--repetitions','1']]
commands += [['replay','--n',str(n)] for n in (32,64,128)]
commands += [['memory','--n',str(n),'--mode',mode] for n in (32,64,128) for mode in ('serial','parallel')]
for number,command in enumerate(commands):
    attempt=0
    while True:
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:
            print('Waiting for GPU to become free.',flush=True);time.sleep(10);continue
        attempt+=1;log=OUT/'logs'/f'{number:02d}_attempt{attempt}.log'
        print('START',*command,flush=True)
        with log.open('w') as file:
            process=subprocess.Popen([sys.executable,str(HERE/'study.py'),*command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                file.write(line);file.flush();print(line,end='',flush=True)
            code=process.wait()
        if code==0:break
        content=log.read_text()
        if 'GPU interference' not in content and 'GPU occupied' not in content:raise SystemExit(f'Experiment failed; see {log}')
        print('Rejected interfered attempt; waiting before retry.',flush=True);time.sleep(10)
    print('DONE',*command,flush=True)
(OUT/'BATCH_COMPLETE.json').write_text(json.dumps(dict(commands=commands,completed=True),indent=2)+'\n')
