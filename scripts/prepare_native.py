"""Fetch pinned external sources and optionally create a local CUDA header overlay."""
from pathlib import Path
import argparse
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'exp/asg_mutopopt_20261104'
PINS={'muTopOpt':'7290b28fedf6ec9510993d68ddb0692f5720fc3c',
      'muGrid':'2faa75b2590d327fbdd846c21cfec959ed381cf5'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-overlay',type=Path,help='Installed CUDA root, e.g. /usr/local/cuda; only needed for the recorded rsqrt declaration incompatibility.')
    a=p.parse_args()
    vendor=HERE/'vendor';vendor.mkdir(exist_ok=True)
    for name,commit in PINS.items():
        dest=vendor/name
        if not dest.exists():
            subprocess.run(['git','clone','--filter=blob:none','https://github.com/muSpectre/'+name+'.git',str(dest)],check=True)
            subprocess.run(['git','checkout',commit],cwd=dest,check=True)
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()
        if actual!=commit:raise RuntimeError('Existing dependency is not the pinned revision: '+name)
    patch=HERE/'results/mugrid_cuda131.patch'
    test=subprocess.run(['git','apply','--check',str(patch)],cwd=vendor/'muGrid',capture_output=True)
    if test.returncode==0:subprocess.run(['git','apply',str(patch)],cwd=vendor/'muGrid',check=True)
    else:subprocess.run(['git','apply','--reverse','--check',str(patch)],cwd=vendor/'muGrid',check=True)
    if a.cuda_overlay:
        include=(a.cuda_overlay/'include').resolve()
        overlay=HERE/'compat'
        if overlay.exists():raise FileExistsError('Overlay already exists; inspect it before replacing.')
        overlay.mkdir();(overlay/'crt').mkdir()
        for source in include.iterdir():
            if source.name!='crt':(overlay/source.name).symlink_to(source,target_is_directory=source.is_dir())
        for source in (include/'crt').iterdir():
            target=overlay/'crt'/source.name
            if source.name!='math_functions.h':target.symlink_to(source,target_is_directory=source.is_dir())
            else:
                text=source.read_text()
                text,count=re.subn(r'(\b(?:rsqrt|rsqrtf)\((?:double|float) x\))\s*;',r'\1 noexcept;',text)
                if count!=2:raise RuntimeError('Installed header differs; expected two declarations. No general patch is inferred.')
                target.write_text(text)
        print('Created local overlay:',overlay)
    print('Pinned sources prepared. Build commands: docs/NATIVE_SOLVER.md')


if __name__=='__main__':main()
