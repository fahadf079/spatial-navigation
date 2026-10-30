"""Checks a RUNNING server (Render, or any address) from the outside, the way your frontend and phone use it.

    python test_deploy.py https://your-service.onrender.com --key YOUR_ACCESS_KEY --origin https://your-frontend.example
    python test_deploy.py http://127.0.0.1:10000 --key YOUR_ACCESS_KEY            (a local --server run)

The key can also come from the ACCESS_KEY environment variable. It is never printed.
It sends 30 test pictures over the camera socket, which makes the server start one (empty) session;
use --no-camera to skip that. Needs: pip install websockets   (opencv and numpy are already installed).
Exit code 0 means every check passed.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

failures = []


def check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f' ({detail})' if detail else ''))
    if not ok:
        failures.append(name)


def fetch(base, path, headers=None, method='GET'):
    req = urllib.request.Request(base + path, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read().decode('utf-8', 'replace')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('url', help='the server address, for example https://your-service.onrender.com')
    parser.add_argument('--key', default=os.environ.get('ACCESS_KEY', ''), help='the ACCESS_KEY of the server')
    parser.add_argument('--origin', default='', help='your frontend\'s address, to test CORS')
    parser.add_argument('--no-camera', action='store_true', help='skip the camera socket test')
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    base = args.url.rstrip('/')
    if not args.key:
        sys.exit('Give the access key with --key or the ACCESS_KEY environment variable.')
    H = {'x-access-key': args.key}

    print('--- reachability')
    start = time.time()
    status = None
    while time.time() - start < 120:                   # a sleeping or just-deployed service can take a minute
        try:
            status, _, body = fetch(base, '/health')
            if status == 200:
                break
        except OSError:
            pass
        time.sleep(3)
    check('/health answers 200', status == 200, f'after {time.time() - start:.0f} s')
    if status != 200:
        sys.exit('The server is not reachable, so the other checks were skipped.')
    check('/health says ok', json.loads(body).get('ok') is True, body[:80])
    check('the address uses https (the phone camera needs it)', base.startswith('https://') or 'localhost' in base
          or '127.0.0.1' in base, base.split('/')[2])

    print('--- access key')
    check('no key: refused', fetch(base, '/detections?limit=1')[0] == 403)
    check('wrong key: refused', fetch(base, '/detections?limit=1', {'x-access-key': 'wrong-' + args.key})[0] == 403)
    code, _, body = fetch(base, '/detections?limit=1', H)
    check('right key (header): answers', code == 200, f'{code}')
    code, _, _ = fetch(base, '/detections?limit=1&key=' + args.key)
    check('right key (in the address): answers', code == 200, f'{code}')
    code, _, body = fetch(base, '/navigate?query=what+is+ahead', H)
    check('a navigation question gets a JSON answer', code == 200 and 'text' in json.loads(body), body[:100])

    if args.origin:
        print('--- CORS (your frontend)')
        code, hdrs, _ = fetch(base, '/navigate?query=x', {'Origin': args.origin, 'Access-Control-Request-Method': 'GET',
                                                          'Access-Control-Request-Headers': 'x-access-key'}, 'OPTIONS')
        check('the preflight from your frontend passes', code == 200 and hdrs.get('access-control-allow-origin') == args.origin,
              f"{code}, allow-origin={hdrs.get('access-control-allow-origin')}; set ALLOWED_ORIGINS if this fails")
        code, hdrs, _ = fetch(base, '/navigate?query=x', {'Origin': 'https://not-allowed.example', 'Access-Control-Request-Method': 'GET'}, 'OPTIONS')
        check('another site is refused', 'access-control-allow-origin' not in hdrs)
        code, hdrs, _ = fetch(base, '/detections?limit=1', {**H, 'Origin': args.origin})
        check('an answer to your frontend carries the CORS header', hdrs.get('access-control-allow-origin') == args.origin)

    if not args.no_camera:
        print('--- camera socket (30 test pictures)')
        try:
            import cv2
            import numpy as np
            from websockets.sync.client import connect
        except ImportError as exc:
            check('the test tools are installed', False, f'{exc}; run: pip install websockets')
        else:
            ws_base = base.replace('https://', 'wss://').replace('http://', 'ws://')
            try:
                with connect(ws_base + '/ws/camera', open_timeout=30) as ws_no_key:
                    ws_no_key.recv(timeout=10)                 # a refused socket closes at once: this raises
                    refused = False
            except Exception:
                refused = True
            check('the camera socket refuses a connection without the key', refused)
            rng = np.random.default_rng(1)
            picture = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 3)
            jpeg = cv2.imencode('.jpg', picture)[1].tobytes()
            times, replies, error = [], [], None
            try:
                with connect(ws_base + '/ws/camera?key=' + args.key, open_timeout=30, max_size=2 ** 22) as ws:
                    ws.send(json.dumps({'type': 'camera', 'lens': 'wide'}))
                    for _ in range(args.frames):
                        t0 = time.time()
                        ws.send(jpeg)
                        replies.append(json.loads(ws.recv(timeout=60)))
                        times.append(time.time() - t0)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            check('the socket accepts the key and answers every picture', error is None and len(replies) == args.frames, error or '')
            if replies:
                check('each answer has the tracking state and events', all('state' in r and 'events' in r for r in replies))
                check('the server did not report an error', not any(r.get('error') for r in replies),
                      str([r.get('error') for r in replies if r.get('error')][:1]))
                med, p95 = statistics.median(times), sorted(times)[int(0.95 * (len(times) - 1))]
                print(f'  [info] round trip per picture: median {med * 1000:.0f} ms, slowest 5% {p95 * 1000:.0f} ms '
                      '(the phone adds its own camera and network time)')
                check('a picture is answered in under 3 seconds on average', statistics.mean(times) < 3.0,
                      f'mean {statistics.mean(times):.2f} s')
            code, _, body = fetch(base, '/health')
            check('the server is still healthy after the pictures', code == 200)

    print('PASS' if not failures else f'FAIL: {len(failures)} check(s) failed: ' + '; '.join(failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
