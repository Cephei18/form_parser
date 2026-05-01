import shutil
import os
from pathlib import Path

ROOT = Path.cwd()
src_input = ROOT / 'input' / 'Job-Application-Form.png'
dest = ROOT / 'input' / 'form.png'

os.makedirs(ROOT / 'input', exist_ok=True)
if src_input.exists():
    shutil.copy(src_input, dest)
    print('Copied', src_input, '->', dest)
else:
    print('Source file not found:', src_input)

print('Running pipeline via import...')
try:
    from src.main import run_default_pipeline

    run_default_pipeline()
except Exception as exc:
    print('Pipeline error:', exc)
    raise
