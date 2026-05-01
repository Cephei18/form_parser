import sys, subprocess
print('Installing into current Python:', sys.executable)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
print('Installed')
