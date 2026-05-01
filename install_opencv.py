import subprocess
python = r'd:\form_parser\venv\Scripts\python.exe'
print('Using', python)
subprocess.check_call([python, '-m', 'pip', 'install', 'opencv-python'])
print('pip install finished')
