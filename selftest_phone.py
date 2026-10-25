"""Self-test for speech safety, limited speech, the tracking-lost behaviour, learning mode and the phone server.

No camera, phone or microphone is needed, and your real spatial_memory.db is never touched.
Run:  python selftest_phone.py        (about 30 seconds; the Windows speech engine is not used)
Optional parts are skipped with a note if their tools are missing (fastapi/httpx, node).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

import guidance
import motion
import sound
import tracker
from tracker import new_session, write_position

RESULTS = []
SKIPPED = []


def check(name, passed, detail=''):
    RESULTS.append(bool(passed))
    print(f"  [{'ok' if passed else 'FAIL'}] {name}" + (f' ({detail})' if detail else ''))
    return passed


# --------------------------------------------------------------------------- speech safety
def sound_checks():
    print('--- speech can never hang the program')
    py = sys.executable
    marker_dir = tempfile.mkdtemp()

    def writer(marker, seconds, child=False):
        """A fake 'speech engine': waits, then writes a marker file (so a survivor would be noticed).
        With child=True the waiting is done by a second process, like PowerShell's helpers."""
        body = f"import time; time.sleep({seconds}); open(r'{marker}', 'w').write('x')"
        if child:
            return [py, '-c', f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {body!r}]); p.wait()"]
        return [py, '-c', body]

    saved = (sound._speak_command, sound._paused_until, list(sound._timeouts))
    try:
        sound._paused_until = 0.0
        sound._timeouts.clear()
        marker = os.path.join(marker_dir, 'm1')
        sound._speak_command = lambda text: (writer(marker, 3, child=True), None)
        t = time.time()
        result = sound.speak_text('a line that never finishes', timeout=1.0)
        took = time.time() - t
        check('a stuck speech process is stopped at the time limit', result is False and took < 3.0, f'{took:.1f}s')
        time.sleep(3.5)
        check('the helper process it started is killed too', not os.path.exists(marker))

        sound._paused_until = 0.0
        sound._timeouts.clear()
        sound._speak_command = lambda text: ([py, '-c', 'pass'], None)
        t = time.time()
        check('a normal line speaks and returns quickly', sound.speak_text('short line', timeout=5) is True
              and time.time() - t < 3.0)

        sound._paused_until = 0.0
        sound._timeouts.clear()
        sound._speak_command = lambda text: ([py, '-c', 'import time; time.sleep(5)'], None)
        for _ in range(sound.BREAKER_TIMEOUTS):
            sound.speak_text('x', timeout=0.3)
        t = time.time()
        paused = sound.speech_paused() and sound.speak_text('while paused', timeout=5) is False
        check('after repeated time-outs speech pauses, so warnings are never stuck behind it',
              paused and time.time() - t < 0.5)
        sound._paused_until = 0.0
        sound._timeouts.clear()

        sound._speak_command = lambda text: ([py, '-c', 'import time; time.sleep(5)'], None)
        threading.Timer(0.6, __import__('_thread').interrupt_main).start()        # like pressing Ctrl+C
        t = time.time()
        try:
            sound.speak_text('interrupted', timeout=20)
            interrupted = False
        except KeyboardInterrupt:
            interrupted = True
        check('Ctrl+C while speaking stops it within a second', interrupted and time.time() - t < 2.5,
              f'{time.time() - t:.1f}s')

        sound._paused_until = 0.0
        sound._timeouts.clear()
        runs = iter([[py, '-c', 'import time; time.sleep(5)'], [py, '-c', 'pass']])
        sound._speak_command = lambda text: (next(runs), None)
        real_timeout = sound.speech_timeout
        sound.speech_timeout = lambda text: 0.5
        speaker = sound.Speaker()
        speaker.say('first line hangs', gap=0)
        time.sleep(2.0)
        speaker.say('second line works', gap=0)
        time.sleep(2.0)
        sound.speech_timeout = real_timeout
        check('the background speaker carries on after a time-out', speaker.works is True)
        check('a longer line gets a longer (but capped) time limit',
              sound.speech_timeout('hi') < sound.speech_timeout('word ' * 12) <= sound.MAX_SPEAK_SECONDS
              and sound.speech_timeout('word ' * 200) == sound.MAX_SPEAK_SECONDS)
    finally:
        sound._speak_command, sound._paused_until = saved[0], saved[1]
        sound._timeouts[:] = saved[2]


# --------------------------------------------------------------------------- limited speech and tracking lost
def set_position_age(seconds_old):
    """Make the position file look `seconds_old` seconds old (as if live_run.py had stopped)."""
    import datetime
    with open(tracker.POSITION_FILE) as f:
        data = json.load(f)
    data['timestamp'] = (datetime.datetime.now() - datetime.timedelta(seconds=seconds_old)).isoformat()
    with open(tracker.POSITION_FILE, 'w') as f:
        json.dump(data, f)


def guidance_checks():
    print('--- limited speech, no stale answers')
    old = tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE
    tmp = tempfile.mkdtemp()
    tracker.DB_PATH = os.path.join(tmp, 'phone.db')
    tracker.POSITION_FILE = os.path.join(tmp, 'position.json')
    tracker.LIVE_SCENE_FILE = os.path.join(tmp, 'live_scene.json')
    try:
        conn = tracker.connect()
        sid = new_session(conn)
        conn.execute('UPDATE sessions SET units_per_meter = 5.0 WHERE id = ?', (sid,))
        for label, x, z in [('chair', 0.0, 4.0), ('laptop', 1.5, 10.0)]:
            conn.execute("INSERT INTO detections (label, confidence, image_source, timestamp, x, y, z, session_id, "
                         "pos_source) VALUES (?, 0.8, 'selftest', '2026-01-01T00:00:00', ?, 0, ?, ?, 'map')",
                         (label, x, z, sid))
        conn.commit()
        eye, zero = np.eye(3), np.zeros((3, 1))
        write_position(sid, eye, zero, 'tracking')

        a = guidance.describe('what is ahead', conn)
        check('"what is ahead" speaks one short line: the object and its distance',
              a['spoken'] == 'Chair, 80 cm.', repr(a['spoken']))
        check('the full answer is still printed and kept', 'chair' in a['text'].lower() and len(a['spoken_full']) > len(a['spoken']))
        d = guidance.describe('take me to the laptop', conn)
        check('directions speak only the object and distance', d['spoken'] == 'Laptop, 2.0 m.', repr(d['spoken']))
        check('the printed directions still have the turn and the obstacle',
              'Direction' in d['text'] and 'Obstacle' in d['text'])
        write_position(sid, np.diag([-1.0, 1.0, -1.0]), zero, 'tracking')
        check('nothing ahead is said plainly, not as "clear"',
              guidance.describe('what is ahead', conn)['spoken'] == 'Nothing known ahead.')
        write_position(sid, eye, zero, 'tracking')

        check('an obstacle warning is one short line: the object and its distance, no direction',
              guidance.warning_line('chair', 0.8) == 'Chair ahead, 0.8 meters'
              and guidance.warning_line('Ahmed, your brother', 0.65) == 'Ahmed, your brother ahead, 0.7 meters'
              and 'left' not in guidance.warning_line('chair', 0.8))
        r = guidance.describe('remember this person as Ahmed, my friend', conn)
        check('people replies are one short line', 0 < len(r['spoken'].split()) <= 14 and r['spoken'] != 'OK.', repr(r['spoken']))
        check('one_line keeps "OK. ..." together and cuts very long sentences',
              guidance.one_line('OK. Point the camera at Ahmed for a few seconds. I will tell you.') ==
              'OK. Point the camera at Ahmed for a few seconds.'
              and len(guidance.one_line('word ' * 40).split()) <= 15)
        guidance.LIMITED_SPEECH = False
        try:
            full = guidance.describe('what is ahead', conn)
            check('with limited speech off, the whole answer is spoken', full['spoken'] == full['spoken_full'])
        finally:
            guidance.LIMITED_SPEECH = True

        for status, line in (('lost', 'Tracking lost.'), ('initializing', 'Tracking is not ready.')):
            write_position(sid, eye, zero, status)
            a = guidance.describe('what is ahead', conn)
            d = guidance.describe('take me to the laptop', conn)
            check(f"tracking '{status}': no stale answer for \"what is ahead\"",
                  a['spoken'] == line and not a['found'] and 'chair' not in a['text'].lower(), repr(a['spoken']))
            check(f"tracking '{status}': no stale directions",
                  d['spoken'] == line and 'Distance' not in d['text'] and not d['found'])
        write_position(sid, eye, zero, 'tracking')

        set_position_age(10)
        a = guidance.describe('what is ahead', conn)
        d = guidance.describe('take me to the laptop', conn)
        check('an old position (the camera program has stopped) is never used',
              a['spoken'] == 'Camera is not running.' and d['spoken'] == 'Camera is not running.' and not a['found'])
        os.remove(tracker.POSITION_FILE)
        check('no position file at all says the camera is not running',
              guidance.describe('what is ahead', conn)['spoken'] == 'Camera is not running.')
        write_position(sid, eye, zero, 'tracking')
        check('a fresh position works again', guidance.describe('what is ahead', conn)['found'])
        conn.close()
    finally:
        tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE = old


# --------------------------------------------------------------------------- learning mode
def simulate_walk(rng, cadence=1.8, step_len=0.7, steps=8, scale=4.5, amp=3.0, rest_before=2.0, rest_after=3.0,
                  imu_hz=60, pos_hz=12, jolts=0, track_noise=0.03, client_zero_ms=50000.0, server_offset=1000.0,
                  latency=0.04, sway=0.4, lost_window=None, new_session_at=None):
    """A synthetic phone walk: accelerometer samples (any phone orientation) and a camera track."""
    total = rest_before + steps / cadence + rest_after
    t = np.arange(0, total, 1.0 / imu_hz) + rng.normal(0, 0.002, int(total * imu_hz) + 1)[:len(np.arange(0, total, 1.0 / imu_hz))]
    tau = t - rest_before
    walking = (tau >= 0) & (tau < steps / cadence)
    mag = 9.81 + rng.normal(0, 0.25, len(t))
    mag += walking * (amp * np.sin(2 * np.pi * cadence * tau) + sway * np.sin(np.pi * cadence * tau))
    for _ in range(jolts):
        k = int(rng.integers(0, len(t)))
        mag[k:k + 4] += 14.0
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    imu = [[client_zero_ms + float(ti) * 1000.0, *(direction * m)] for ti, m in zip(t, mag)]
    tp = np.arange(0, total, 1.0 / pos_hz)
    dist_m = step_len * cadence * np.clip(tp - rest_before, 0, steps / cadence)
    track = []
    for ti, d in zip(tp, dist_m):
        status = 'lost' if lost_window and lost_window[0] <= ti <= lost_window[1] else 'tracking'
        session = 8 if new_session_at is not None and ti >= new_session_at else 7
        track.append((float(ti) + server_offset, float(d * scale + rng.normal(0, track_noise)),
                      float(rng.normal(0, track_noise)), float(rng.normal(0, track_noise)), status, session))
    return {'imu': imu, 'track': track, 'total': total, 'server_offset': server_offset, 'latency': latency,
            'client_zero_ms': client_zero_ms}


def run_learning(walk, step_assumed=0.7, known_scale=None, start_status='tracking', stand_still=False):
    """Drive a LearningController through a simulated walk on a fake clock."""
    clock = {'now': walk['server_offset'] - 1.0}
    events, saved = [], {}
    current = {'pos': {'session_id': 7, 'x': 0, 'y': 0, 'z': 0, 'status': start_status, 'timestamp': 'init'}}
    ctrl = motion.LearningController(lambda text, speak=True, learn=None: events.append((text, speak, learn)),
                                     lambda: current['pos'], lambda sid: known_scale,
                                     lambda sid, value: saved.update({sid: value}), clock=lambda: clock['now'])
    ctrl.start(step_assumed)
    imu, track = walk['imu'], walk['track']
    k = j = 0
    end = walk['server_offset'] + walk['total'] + 8 + (motion.LEARN_TIMEOUT_SECONDS if stand_still else 0)
    while clock['now'] < end and ctrl.state == 'learning':
        clock['now'] += 0.1
        batch = []
        while k < len(imu) and (imu[k][0] - walk['client_zero_ms']) / 1000.0 + walk['server_offset'] <= clock['now'] - walk['latency']:
            batch.append(imu[k])
            k += 1
        if batch:
            ctrl.on_motion(batch)
        while j < len(track) and track[j][0] <= clock['now']:
            x = track[j]
            current['pos'] = {'session_id': x[5], 'x': x[1], 'y': x[2], 'z': x[3], 'status': x[4], 'timestamp': f't{j}'}
            j += 1
        ctrl.tick()
    return ctrl, events, saved


def motion_checks():
    print('--- learning mode (motion sensors set the scale)')
    bad = 0
    for cadence in (1.4, 1.7, 2.0, 2.3):
        for amp in (1.5, 3.0, 5.0):
            for seed in range(4):
                w = simulate_walk(np.random.default_rng(seed), cadence=cadence, amp=amp)
                t = np.array([s[0] for s in w['imu']]) / 1000.0
                m = np.array([np.linalg.norm(s[1:]) for s in w['imu']])
                bad += abs(len(motion.detect_steps(t, m)) - 8) > 1
    check('steps are counted within one step of the truth (cadences 1.4 to 2.3 steps/s, small to large bounce, any phone angle)',
          bad == 0, f'{bad} of 48 walks off by more than one step')

    rng = np.random.default_rng(5)
    for name, kwargs, limit in (('standing still for 20 s', dict(steps=0, rest_before=10, rest_after=10), 0),
                                ('pocket bumps with no walking', dict(steps=0, rest_before=10, rest_after=10, jolts=4), motion.MIN_STEPS - 1)):
        w = simulate_walk(rng, **kwargs)
        t = np.array([s[0] for s in w['imu']]) / 1000.0
        m = np.array([np.linalg.norm(s[1:]) for s in w['imu']])
        n = len(motion.detect_steps(t, m))
        check(f'{name} is not mistaken for a walk', n <= limit, f'{n} steps')

    errors = []
    for seed in range(15):
        r = np.random.default_rng(100 + seed)
        w = simulate_walk(r, cadence=float(r.uniform(1.5, 2.2)), amp=float(r.uniform(2, 4.5)))
        ctrl, events, saved = run_learning(w)
        res = ctrl.last_result
        errors.append(abs(res['units_per_meter'] / 4.5 - 1) if res and res['ok'] else 9.9)
    check('with the right step length the scale is within 10% (15 simulated walks)', max(errors) < 0.10,
          f'worst {100 * max(errors):.1f}%')

    for true_step, label in ((0.60, 'short'), (0.85, 'long')):
        errs = []
        for seed in range(6):
            w = simulate_walk(np.random.default_rng(300 + seed), step_len=true_step)
            ctrl, _, _ = run_learning(w, step_assumed=0.70)
            res = ctrl.last_result
            errs.append(abs(res['units_per_meter'] / 4.5 - 1) if res and res['ok'] else 9.9)
        check(f'a {label} stride ({true_step} m, assumed 0.70 m) gives an error close to the stride error, as documented',
              max(errs) < abs(true_step / 0.70 - 1) + 0.08, f'worst {100 * max(errs):.0f}%')

    w = simulate_walk(np.random.default_rng(1))
    ctrl, events, saved = run_learning(w)
    texts = [e[0] for e in events]
    check('a good walk saves the scale for that session and says so',
          saved.get(7) and 3.5 < saved[7] < 5.5 and 'Scale saved.' in texts and events[-2][2] == 'done',
          f'saved {saved}')
    check('the prompts are short and the details are shown but not spoken',
          texts[0] == 'Learning mode. Walk 8 steps, then stop.' and all(not e[1] for e in events if len(e[0]) > 40))

    cases = [
        ('tracking lost during the walk', dict(lost_window=(3.2, 3.8)), 'lost'),
        ('a new map started during the walk', dict(new_session_at=3.5), 'new map'),
        ('too few steps', dict(steps=2), 'only'),
    ]
    for name, kwargs, word in cases:
        ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(2), **kwargs))
        check(f'{name}: nothing is saved and the user is told',
              not saved and any(e[2] == 'failed' for e in events) and word in ctrl.last_result['reason'],
              ctrl.last_result['reason'])
    ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(3), steps=0, rest_before=1, rest_after=1),
                                       stand_still=True)
    check('if nobody walks it gives up after the time-out and saves nothing',
          not saved and any(e[2] == 'failed' for e in events) and ctrl.state == 'idle')
    # every second bounce missing: the detector sees a regular but half-speed walk, which would halve the
    # distance and double the scale, so the pace check must refuse it
    missed = simulate_walk(np.random.default_rng(4))
    t_ms = np.array([s[0] for s in missed['imu']])
    for i in range(len(t_ms)):
        if int((t_ms[i] - t_ms[0]) / 1000.0 * 1.8) % 2 == 0:
            missed['imu'][i][1:] = [0.0, 0.0, 9.81]
    ctrl, events, saved = run_learning(missed)
    check('a walk with missed steps (pace too slow to be real) is not saved', not saved and 'pace' in (ctrl.last_result or {}).get('reason', ''),
          (ctrl.last_result or {}).get('reason', ''))
    ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(8), cadence=0.9, amp=3.0))
    check('a very slow shuffle is not saved', not saved)
    ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(9), cadence=1.3, amp=3.0, steps=7))
    check('a slow but normal walk (1.3 steps a second) is still accepted', bool(saved), (ctrl.last_result or {}).get('reason', ''))

    ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(6)), known_scale=4.0)
    check('a session that already has a measured scale is left alone',
          not saved and events[0][2] == 'skipped' and ctrl.state == 'idle')
    ctrl, events, saved = run_learning(simulate_walk(np.random.default_rng(6)), start_status='lost')
    check('learning does not start while tracking is not working', not saved and events[0][2] == 'failed')
    absurd = simulate_walk(np.random.default_rng(7), scale=900.0)
    ctrl, events, saved = run_learning(absurd)
    check('an absurd scale (outside 0.5 to 200) is refused', not saved and 'usual range' in ctrl.last_result['reason'])
    check('height gives a step length in a sane range',
          abs(motion.step_length_from_height_cm(175) - 0.726) < 0.002 and motion.step_length_from_height_cm(50) == motion.MIN_STEP_M)
    check('bad messages from a phone are ignored',
          motion.parse_message('not json') is None and motion.parse_message('[1,2]') is None
          and motion.parse_message('{"type":"motion"}') == {'type': 'motion'})
    ctrl = motion.LearningController(lambda *a, **k: None, lambda: None, lambda s: None, lambda s, v: None)
    ctrl.on_motion([[1, 2, 3, 4]])
    ctrl.tick()
    check('samples and ticks outside learning do nothing', ctrl.state == 'idle')


# --------------------------------------------------------------------------- the phone server
def server_checks():
    print('--- phone server (websocket)')
    try:
        import cv2
        from fastapi.testclient import TestClient
        import api
    except ImportError as exc:
        SKIPPED.append(f'phone server test: {exc.name} is not installed')
        print(f'  [skipped] {exc.name} is not installed')
        return
    old = tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE
    tmp = tempfile.mkdtemp()
    tracker.DB_PATH = os.path.join(tmp, 'server.db')
    tracker.POSITION_FILE = os.path.join(tmp, 'position.json')
    tracker.LIVE_SCENE_FILE = os.path.join(tmp, 'live_scene.json')
    conn = tracker.connect()
    sid = new_session(conn)
    conn.close()

    class FakePipeline(api.PhonePipeline):
        """The real learning mode and message handling, with the camera program replaced by a stand-in."""

        def start(self, first_frame):
            self.source = api.FrameQueueSource()
            self.state = {'status': 'tracking'}
            self.learner.reset()

            def drain():
                while True:
                    ok, _ = self.source.read()
                    if not ok:
                        return

            self.thread = threading.Thread(target=drain, daemon=True)
            self.thread.start()

    fake = FakePipeline()
    clock = {'now': 1000.0}
    fake.learner.clock = lambda: clock['now']
    old_pipe = api.PIPE
    api.PIPE = fake
    try:
        walk = simulate_walk(np.random.default_rng(1))
        walk['track'] = [(t, x, y, z, st, sid) for t, x, y, z, st, _ in walk['track']]
        jpeg = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))[1].tobytes()
        got = []
        with TestClient(api.app) as client, client.websocket_connect('/ws/camera') as ws:
            write_position(sid, np.eye(3), np.zeros((3, 1)), 'tracking')
            ws.send_bytes(jpeg)
            got += ws.receive_json()['events']
            fake.state = {'status': 'tracking', 'searching': True}
            ws.send_text(json.dumps({'type': 'learn', 'action': 'start'}))
            ws.send_bytes(jpeg)
            early = ws.receive_json()['events']
            check('"learn start" is refused while it still looks for the saved map',
                  any(e.get('learn') == 'failed' and 'saved map' in e['text'] for e in early), str(early)[:80])
            fake.state = {'status': 'tracking', 'searching': False}
            ws.send_text('this is not json')
            ws.send_text(json.dumps({'type': 'learn', 'action': 'start', 'step_m': 0.7}))
            ws.send_bytes(jpeg)
            reply = ws.receive_json()
            got += reply['events']
            check('"learn start" over the websocket gives a spoken prompt', any('Learning mode' in e['text'] and e['speak'] for e in got), str(got)[:80])
            imu, track = walk['imu'], walk['track']
            k = j = 0
            clock['now'] = walk['server_offset'] - 1.0
            while clock['now'] < walk['server_offset'] + walk['total'] + 8 and not any(e.get('learn') == 'done' for e in got):
                clock['now'] += 0.1
                batch = []
                while k < len(imu) and (imu[k][0] - walk['client_zero_ms']) / 1000.0 + walk['server_offset'] <= clock['now'] - walk['latency']:
                    batch.append(imu[k])
                    k += 1
                if batch:
                    ws.send_text(json.dumps({'type': 'motion', 's': batch}))
                while j < len(track) and track[j][0] <= clock['now']:
                    x = track[j]
                    j += 1
                    c = np.array([x[1], x[2], x[3]])
                    write_position(sid, np.eye(3), (-c).reshape(3, 1), x[4])
                ws.send_bytes(jpeg)
                got += ws.receive_json()['events']
        done = [e for e in got if e.get('learn') == 'done']
        check('the walk over the websocket ends with "Scale saved."', len(done) == 1 and done[0]['text'] == 'Scale saved.')
        conn = tracker.connect()
        saved = tracker.get_units_per_meter(conn, sid)
        conn.close()
        check('the scale was written to the database for that session', saved is not None and 3.5 < saved < 5.5, f'{saved}')
        check('events carry the speak flag, so quiet details are not spoken',
              all(isinstance(e, dict) and 'speak' in e for e in got) and any(not e['speak'] for e in got))
    finally:
        api.PIPE = old_pipe
        tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE = old


# --------------------------------------------------------------------------- the phone page
def page_checks():
    print('--- phone page logic (JavaScript, stand-in browser)')
    try:
        import api
    except ImportError as exc:
        SKIPPED.append(f'phone page test: {exc.name} is not installed')
        print(f'  [skipped] {exc.name} is not installed')
        return
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, 'page_test.js')
    import re
    page = re.findall(r'<script>(.*?)</script>', api.PAGE.replace('__CAMERA__', 'true'), re.S)[0]
    path = os.path.join(tempfile.mkdtemp(), 'page.js')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    try:
        syntax = subprocess.run(['node', '--check', path], capture_output=True, text=True, timeout=60)
        run = subprocess.run(['node', script, path], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        SKIPPED.append('phone page test: node is not installed')
        print('  [skipped] node is not installed (https://nodejs.org)')
        return
    check('the page JavaScript has no syntax errors', syntax.returncode == 0, syntax.stderr[:200])
    lines = [x for x in run.stdout.splitlines() if x.startswith('[')]
    for line in lines:
        check(line[5:].strip(), line.startswith('[ok]'))
    check('every page check ran', len(lines) >= 15 and run.returncode == 0, f'{len(lines)} checks')


def main():
    sound_checks()
    guidance_checks()
    motion_checks()
    server_checks()
    page_checks()
    for note in SKIPPED:
        print('skipped:', note)
    passed = all(RESULTS)
    print(f"{sum(RESULTS)} of {len(RESULTS)} checks passed")
    print('PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
