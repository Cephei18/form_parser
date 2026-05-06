import subprocess
import sys

python = sys.executable
print('Using', python)
subprocess.check_call([python, '-m', 'pip', 'install', 'opencv-python'])
print('pip install finished')
