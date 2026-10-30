"""Self-test for the server side. No camera, phone, network or cloud account needed.

Checks the access key, CORS (a frontend on another address), /health, the phone lens option, "look around"
over the camera socket, the fresh/old fields, and that `python api.py --server` starts and behaves.
Your real spatial_memory.db and current_position.json are never touched (temporary files).
Run:  python selftest_server.py
"""
import datetime
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api
import tracker
from tracker import connect, new_session, write_position

failures = []
KEY = 'k' * 24
ORIGIN = 'https://frontend.example.com'


def check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f' ({detail})' if detail else ''))
    if not ok:
        failures.append(name)


def rot(yaw_deg):
    a = math.radians(yaw_deg)
    f = np.array([math.sin(a), 0.0, math.cos(a)])
    r = np.array([math.cos(a), 0.0, -math.sin(a)])
    return np.array([r, np.cross(f, r), f])


tmp = tempfile.mkdtemp()
tracker.DB_PATH = os.path.join(tmp, 'server.db')
tracker.POSITION_FILE = os.path.join(tmp, 'position.json')
tracker.LIVE_SCENE_FILE = os.path.join(tmp, 'live_scene.json')
import lookaround  # noqa: E402  (after the paths above are set)
lookaround.LAST_SCAN_FILE = os.path.join(tmp, 'last_scan.json')

conn = connect()
sid = new_session(conn)
now = datetime.datetime.now()
for label, seconds_ago in (('chair', 3), ('couch', 900)):
    conn.execute('INSERT INTO detections (label, confidence, x, y, z, pos_source, session_id, timestamp) '
                 'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                 (label, 0.9, 0.0, 0.0, 5.0, 'map', sid, (now - datetime.timedelta(seconds=seconds_ago)).isoformat()))
conn.commit()
conn.close()
write_position(sid, np.eye(3), np.zeros((3, 1)), 'tracking')

api.TOKEN = KEY
api.configure_cors([ORIGIN])                       # before the first request, as the start-up code does


class FakePipeline(api.PhonePipeline):
    """The real message handling and look-around, with the camera program replaced by a stand-in."""

    def start(self, first_frame):
        self.source = api.FrameQueueSource()
        self.state = {'status': 'tracking'}
        self.learner.reset()
        self.scan.reset()

        def drain():
            while True:
                ok, _ = self.source.read()
                if not ok:
                    return

        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()


jpeg = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))[1].tobytes()
real_pipe = api.PIPE
H = {'x-access-key': KEY}

# ------------------------------------------------------------------------------------ access key
print('--- access key, health and CORS')
client = TestClient(api.app)
r = client.get('/health')
check('/health needs no key and shows nothing private', r.status_code == 200 and r.json() == {'ok': True, 'pipeline': 'idle'},
      r.text)
check('a page without the key is refused', client.get('/status').status_code == 403)
check('a wrong key is refused', TestClient(api.app).get('/status?key=nope').status_code == 403)
check('a key with odd characters is refused, not a crash', TestClient(api.app).get('/status?key=%D0%BA%D0%BB').status_code == 403)
check('a key one character short is refused', TestClient(api.app).get('/status?key=' + KEY[:-1]).status_code == 403)
check('the key in the address works', TestClient(api.app).get('/status?key=' + KEY).status_code == 200)
check('the key in an X-Access-Key header works (for a separate frontend)',
      TestClient(api.app).get('/status', headers=H).status_code == 200)
c2 = TestClient(api.app)
c2.get('/status?key=' + KEY)
check('after the first visit the page keeps working through the cookie', c2.get('/status').status_code == 200)

pre = client.options('/navigate', headers={'Origin': ORIGIN, 'Access-Control-Request-Method': 'GET',
                                           'Access-Control-Request-Headers': 'x-access-key'})
check('a browser preflight from the frontend passes without a key',
      pre.status_code == 200 and pre.headers.get('access-control-allow-origin') == ORIGIN
      and 'x-access-key' in pre.headers.get('access-control-allow-headers', '').lower(), str(dict(pre.headers)))
bad = client.options('/navigate', headers={'Origin': 'https://evil.example.com', 'Access-Control-Request-Method': 'GET'})
check('a preflight from another site is refused', 'access-control-allow-origin' not in bad.headers, str(dict(bad.headers)))
ok = client.get('/navigate?query=what+is+ahead', headers={**H, 'Origin': ORIGIN})
check('the frontend can read an answer', ok.status_code == 200 and ok.headers.get('access-control-allow-origin') == ORIGIN)
denied = client.get('/status', headers={'Origin': ORIGIN})
check('even a refused request carries the CORS header, so the frontend sees "403" and not a vague network error',
      denied.status_code == 403 and denied.headers.get('access-control-allow-origin') == ORIGIN)
check('a refused request from another site gets no CORS header',
      'access-control-allow-origin' not in client.get('/status', headers={'Origin': 'https://evil.example.com'}).headers)

# ------------------------------------------------------------------------------ fresh and old fields
print('--- fresh and old in the data')
rows = {d['label']: d for d in client.get('/detections?limit=10', headers=H).json()}
check('each detection says how long ago it was seen', isinstance(rows['chair']['age_seconds'], float)
      and rows['couch']['age_seconds'] > 800, str({k: round(v['age_seconds']) for k, v in rows.items()}))
check('recent is fresh and old is not', rows['chair']['fresh'] is True and rows['couch']['fresh'] is False)

# ----------------------------------------------------------------------------------- the phone lens
print('--- phone lens')
launched = {}
release = threading.Event()
fake_live_run = type(sys)('live_run')


def fake_run(**kw):
    launched['K'] = kw['K']
    release.wait(5)


fake_live_run.run = fake_run
saved_module = sys.modules.get('live_run')
sys.modules['live_run'] = fake_live_run
frame = np.zeros((480, 640, 3), np.uint8)
try:
    p = api.PhonePipeline()
    p.start(frame)
    time.sleep(0.2)
    check('a normal lens uses the usual focal length', abs(launched['K'][0, 0] - 0.8 * 640) < 1e-6, str(launched['K'][0, 0]))
    check('the focal length is not changed while the camera is running', True if p.running else False)
    p.handle_text(json.dumps({'type': 'camera', 'lens': 'ultrawide'}))
    check('a lens message while running is ignored', p.lens == 'wide')
    release.set()
    p.stop()
    p2 = api.PhonePipeline()
    p2.handle_text(json.dumps({'type': 'camera', 'lens': 'ultrawide'}))
    check('the lens message before the first frame is accepted', p2.lens == 'ultrawide')
    release.clear()
    p2.start(frame)
    time.sleep(0.2)
    check('an ultra-wide lens uses a shorter focal length', abs(launched['K'][0, 0] - 0.38 * 640) < 1e-6, str(launched['K'][0, 0]))
    release.set()
    p2.stop()
    p3 = api.PhonePipeline()
    for junk in ('telephoto', '', None, 5, ['ultrawide'], {'a': 1}):
        p3.handle_text(json.dumps({'type': 'camera', 'lens': junk}))
    check('an unknown or odd lens value is ignored', p3.lens == 'wide')
    p3.handle_text('not json')
    p3.handle_text(json.dumps({'type': 'scan', 'action': 'nonsense'}))
    check('odd messages are ignored without a crash', True)
finally:
    if saved_module is not None:
        sys.modules['live_run'] = saved_module
    else:
        del sys.modules['live_run']

# ---------------------------------------------------------------------------- look around, end to end
print('--- look around over the camera socket')
text0 = client.get('/navigate?query=look+around', headers=H).json()
check('"look around" with no phone connected says to start the camera', text0['found'] is False
      and 'Start the phone camera first' in text0['text'], str(text0))

fake = FakePipeline()
api.PIPE = fake
events = []
try:
    with TestClient(api.app) as tc, tc.websocket_connect('/ws/camera?key=' + KEY) as ws:
        ws.send_text(json.dumps({'type': 'camera', 'lens': 'ultrawide'}))
        write_position(sid, np.eye(3), np.zeros((3, 1)), 'tracking')
        ws.send_bytes(jpeg)
        events += ws.receive_json()['events']
        check('the lens chosen on the page reaches the server', fake.lens == 'ultrawide', fake.lens)

        answer = tc.get('/navigate?query=look+around', headers=H).json()
        check('"look around" by voice starts the scan', answer['found'] is True and answer['text'] == 'Look around started.'
              and answer['spoken'] == '', str(answer))
        again = tc.get('/navigate?query=look+around', headers=H).json()
        check('asking twice does not start a second scan', again['found'] is False, str(again))

        yaw = 0.0
        while yaw < 400 and fake.scan.active:
            yaw += 3
            write_position(sid, rot(yaw), np.zeros((3, 1)), 'tracking')
            ws.send_bytes(jpeg)
            events += ws.receive_json()['events']
        texts = [e['text'] for e in events]
        check('the phone is told how to turn, with a spoken line', {'text': 'Turn slowly all the way around on the spot.',
                                                                     'speak': True} in events, str(texts))
        check('progress reaches the phone but is not spoken', any(e['text'].startswith('Look around: 50%') and e['speak'] is False
                                                                   for e in events), str(texts))
        check('the phone is told when the scan is done', texts[-1] == 'Look around done. I have seen all around you.',
              str(texts[-3:]))
        printed = tc.get('/navigate?query=what+is+ahead', headers=H).json()['text']
        check('the printed answers say how much the last look-around saw',
              re.search(r'Last look-around just now saw (9[7-9]|100)% of the circle\.', printed) is not None, printed)

        ws.send_text(json.dumps({'type': 'scan', 'action': 'start'}))
        ws.send_bytes(jpeg)
        events = ws.receive_json()['events']
        check('the page button starts a scan too (a "scan" message)', fake.scan.active and
              any('Turn slowly' in e['text'] for e in events), str(events))
        ws.send_text(json.dumps({'type': 'scan', 'action': 'cancel'}))
        ws.send_bytes(jpeg)
        events = ws.receive_json()['events']
        check('and it can be cancelled', not fake.scan.active and any('cancelled' in e['text'] for e in events), str(events))
finally:
    api.PIPE = real_pipe

with TestClient(api.app) as tc:
    try:
        with tc.websocket_connect('/ws/camera') as ws:
            ws.receive_json()
        refused = False
    except Exception:
        refused = True
check('the camera socket refuses a connection without the key', refused)

# ------------------------------------------------------------------------------------------ the page
print('--- the page')
page = api.PAGE.replace('__CAMERA__', 'true')
check('the page has the lens choice and the look-around button', 'id="lens"' in page and 'id="scanbtn"' in page
      and 'Ultra-wide' in page and 'Seen</th>' in page)
check('the normal lens stays the default', re.search(r'<select id="lens"><option value="wide">', page) is not None)
check('the page sends the lens before the first frame', "type:'camera', lens: lens" in page)
script = re.findall(r'<script>(.*?)</script>', page, re.S)[0]
node = shutil.which('node')
if node:
    path = os.path.join(tmp, 'page.js')
    with open(path, 'w') as f:
        f.write(script)
    result = subprocess.run([node, '--check', path], capture_output=True, text=True)
    check('the page JavaScript has no syntax errors', result.returncode == 0, result.stderr.strip()[:200])
else:
    print('  [skip] node is not installed, so the page JavaScript was not syntax-checked')

# ------------------------------------------------------------------------------- python api.py --server
print('--- python api.py --server')
here = os.path.dirname(os.path.abspath(api.__file__))


def run_server(env_extra, wait=0.0):
    env = {k: v for k, v in os.environ.items() if k not in ('ACCESS_KEY', 'PORT', 'ALLOWED_ORIGINS', 'SN_DATA_DIR')}
    env.update(env_extra)
    return subprocess.Popen([sys.executable, os.path.join(here, 'api.py'), '--server'], cwd=tmp, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


proc = run_server({})
out, _ = proc.communicate(timeout=30)
check('--server refuses to start without ACCESS_KEY', proc.returncode not in (0, None) and 'ACCESS_KEY' in out, out.strip()[-120:])
proc = run_server({'ACCESS_KEY': 'short'})
out, _ = proc.communicate(timeout=30)
check('--server refuses a short ACCESS_KEY', proc.returncode not in (0, None) and 'ACCESS_KEY' in out, out.strip()[-120:])

s = socket.socket()
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()
data_dir = os.path.join(tmp, 'data')
secret = 'S3cret-' + 'x' * 20
proc = run_server({'ACCESS_KEY': secret, 'PORT': str(port), 'ALLOWED_ORIGINS': ORIGIN, 'SN_DATA_DIR': data_dir})
base = f'http://127.0.0.1:{port}'


def fetch(path, headers=None, method='GET'):
    req = urllib.request.Request(base + path, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode()


try:
    up = False
    for _ in range(60):
        try:
            up = fetch('/health')[0] == 200
            break
        except OSError:
            time.sleep(0.5)
    check('the server starts and answers /health on the port the host gives it', up)
    if up:
        check('the server refuses requests without the key', fetch('/sessions')[0] == 403)
        code, _, body = fetch('/sessions', {'x-access-key': secret})
        check('and answers with the key in the header', code == 200 and body == '[]', f'{code} {body[:60]}')
        check('its database was created in SN_DATA_DIR', os.path.exists(os.path.join(data_dir, 'spatial_memory.db')))
        code, hdrs, _ = fetch('/navigate?query=x', {'Origin': ORIGIN, 'Access-Control-Request-Method': 'GET'}, 'OPTIONS')
        check('the preflight from ALLOWED_ORIGINS works on the running server',
              code == 200 and hdrs.get('access-control-allow-origin') == ORIGIN, f'{code} {hdrs}')
        code, _, body = fetch('/navigate?query=what+is+ahead', {'x-access-key': secret})
        check('a question gets an answer when the camera is not running', code == 200 and json.loads(body)['found'] is False, body[:100])
finally:
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
check('the access key is never written to the server log', secret not in out)

print('PASS' if not failures else f'FAIL: {len(failures)} check(s) failed: ' + '; '.join(failures))
sys.exit(1 if failures else 0)
