import http.client, sys

try:
    conn = http.client.HTTPConnection('127.0.0.1',8000,timeout=5)
    conn.request('GET','/')
    r = conn.getresponse()
    print(r.status)
    data = r.read()
    print(data.decode('utf-8')[:2000])
except Exception as e:
    print('ERR', e)
    sys.exit(2)
