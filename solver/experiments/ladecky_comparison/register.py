"""Add the isolated comparison to the project's supported command entry."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
path=root/'run.py'
text=path.read_text()
line='    "ladecky-study": ROOT / "experiments/ladecky_comparison/study.py",\n'
if line not in text:
    assert text.count('COMMANDS = {\n')==1
    archive=root/'runs/ladecky_comparison_20261025/run_before.py'
    if not archive.exists():archive.write_text(text)
    path.write_text(text.replace('COMMANDS = {\n','COMMANDS = {\n'+line))
print('registered ladecky-study')
