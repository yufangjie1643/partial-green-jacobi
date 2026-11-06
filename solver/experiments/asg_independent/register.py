"""Add only the ASG experiment command to the canonical launcher."""
from pathlib import Path
root=Path('${HOME}/code/voxel-mechanics')
p=root/'run.py'
text=p.read_text()
for name,file in [('asg-independent','study.py'),('asg-independent-train','train.py'),('asg-independent-evaluate','evaluate.py'),('asg-independent-resume-resolution','resume_resolution.py')]:
    line=f'    "{name}": ROOT / "experiments/asg_independent/{file}",\n'
    if line not in text:
        assert f'"{name}"' not in text
        assert text.count('COMMANDS = {\n')==1
        text=text.replace('COMMANDS = {\n','COMMANDS = {\n'+line)
p.write_text(text)
print('asg-independent registered')
