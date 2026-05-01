import requests, sys, os

API='http://127.0.0.1:8000/process-form'
file_path = os.path.join('output','output.pdf')
if not os.path.exists(file_path):
    print('File not found:', file_path)
    sys.exit(2)

with open(file_path,'rb') as f:
    files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
    try:
        r = requests.post(API, files=files, timeout=300)
        print('Status:', r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text[:1000])
    except Exception as e:
        print('ERR', e)
