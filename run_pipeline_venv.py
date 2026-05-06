import shutil
import os
import sys
from pathlib import Path
import subprocess

ROOT = Path.cwd()
src_input = ROOT / 'input' / 'Job-Application-Form.png'
dest = ROOT / 'input' / 'form.png'

os.makedirs(ROOT / 'input', exist_ok=True)
if src_input.exists():
    shutil.copy(src_input, dest)
    print('Copied', src_input, '->', dest)
else:
    print('Source file not found:', src_input)

PY = sys.executable
print('Running pipeline with', PY)
ret = subprocess.run([PY, '-m', 'src.main'])
print('returncode', ret.returncode)

print('Output dir contents:', os.listdir(ROOT / 'output'))
