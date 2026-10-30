"""Self-test for the look-around scan and the fresh/old markers. No camera, phone or network needed.

Simulates a camera turning on the spot (the tracker's rotation changing), runs the scan on it, and
checks the coverage, the spoken lines, the stop conditions and the "what is ahead" wording.
Your real spatial_memory.db and current_position.json are never touched (temporary files).
Run:  python selftest_lookaround.py
"""
import datetime
import math
import os
import sys
import tempfile

import numpy as np

import guidance
import lookaround
import tracker
from lookaround import ScanController, YawTracker, age_words, bins_covered, missing_text
from tracker import connect, new_session, write_live_scene, write_position

failures = []


def check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f' ({detail})' if detail else ''))
    if not ok:
        failures.append(name)


def rot(yaw_deg, pitch_deg=0.0):
    """World-to-camera rotation after turning right by yaw and tilting up by pitch (world = the start view)."""
    a, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.sin(a) * math.cos(p), -math.sin(p), math.cos(a) * math.cos(p)])   # forward
    r = np.array([math.cos(a), 0.0, -math.sin(a)])                                        # right
    return np.array([r, np.cross(f, r), f])                                               # rows: right, down, forward


def wrap(deg):
    return (deg + 180) % 360 - 180


def position(R, status='tracking', session=1, seconds_ago=0.0):
    ts = datetime.datetime.now() - datetime.timedelta(seconds=seconds_ago)
    return {'status': status, 'rotation': np.asarray(R).tolist(), 'session_id': session, 'timestamp': ts.isoformat()}


class Sim:
    """A scan controller on a fake clock and a fake tracker position."""

    def __init__(self, lens='wide', start_R=None):
        self.now = 0.0
        self.pos = position(np.eye(3) if start_R is None else start_R)
        self.messages = []
        self.ctl = ScanController(lambda text, speak=True, **_: self.messages.append((text, bool(speak))),
                                  lambda: self.pos, lambda: lens, clock=lambda: self.now, save=False)

    def turn(self, to_yaw, status='tracking', **kw):
        self.pos = position(rot(to_yaw), status, **kw)
        self.now += 0.1
        self.ctl.tick()

    def sweep(self, step, total, direction=1):
        """Turn in steps; returns the degrees turned when the scan ended, or None if it never ended."""
        yaw = 0.0
        while abs(yaw) < total and self.ctl.active:
            yaw += direction * step
            self.turn(wrap(yaw))
        return None if self.ctl.active else abs(yaw)

    @property
    def last(self):
        return self.messages[-1][0] if self.messages else ''


# ------------------------------------------------------------------------------------------ the maths
print('--- turning maths')
yt = YawTracker(np.eye(3))
errs = [abs(wrap(yt.yaw(rot(a, p)) - a)) for a in range(-179, 180, 7) for p in (0, 30, -30)]
check('turning right is positive, and tilting the phone does not change the yaw', max(errs) < 1e-6,
      f'largest error {max(errs):.1e} deg')
check('a camera pointing straight up gives no yaw', yt.yaw(rot(0, 89.9)) is None)


def about_vertical(a, p):
    """Turn about the real vertical while the phone is tilted up by p, with the tracker frame set at the start."""
    ra, rp = math.radians(a), math.radians(p)
    Ry = np.array([[math.cos(ra), 0, math.sin(ra)], [0, 1, 0], [-math.sin(ra), 0, math.cos(ra)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rp), -math.sin(rp)], [0, math.sin(rp), math.cos(rp)]])
    return (Ry @ Rx).T @ Rx                                # world-to-camera in the start-view frame


tilt_err = max(abs(wrap(YawTracker(np.eye(3)).yaw(about_vertical(a, 20)) - a)) for a in range(-170, 180, 10))
check('a phone tilted 20 degrees at the start still measures the turn to within 15 degrees', tilt_err < 15,
      f'worst case {tilt_err:.1f} deg: hold the phone level when you start')

check('a normal lens covers 40 degrees at a time, an ultra-wide 60',
      len(bins_covered(0, 65)) == 4 and len(bins_covered(0, 105)) == 6,
      f'{len(bins_covered(0, 65)) * 10} and {len(bins_covered(0, 105)) * 10}')
check('coverage wraps around behind you (175 and -175 degrees are the same direction)',
      bins_covered(-175, 65) == bins_covered(185, 65))
check('nothing missing when the whole circle is seen', missing_text(set(range(36))) == '')
check('the missing directions are named', missing_text(set(range(0, 5)) | set(range(32, 36))) ==
      'Not seen: to your right, behind you, to your left.', missing_text(set(range(0, 5)) | set(range(32, 36))))
check('age words', [age_words(3), age_words(45), age_words(300), age_words(7300)] ==
      ['just now', '45 s ago', '5 min ago', '2 h ago'])

# ------------------------------------------------------------------------------------------ the scan
print('--- the scan')
wide = Sim('wide')
check('a scan starts when tracking works', wide.ctl.start() and wide.ctl.active)
check('the first line is spoken and tells you what to do', wide.messages == [
    ('Turn slowly all the way around on the spot.', True)], str(wide.messages))
turned_wide = wide.sweep(3, 400)
check('a full turn to the right finishes the scan', turned_wide is not None and wide.last.startswith('Look around done.'),
      f'turned {turned_wide} deg: {wide.last}')
check('a finished scan says everything was seen', wide.last == 'Look around done. I have seen all around you.', wide.last)
check('only the first and last lines are spoken, progress is shown but not spoken',
      [s for _, s in wide.messages] == [True, False, False, False, True], str(wide.messages))
check('progress is reported at 25, 50 and 75 percent',
      [t for t, s in wide.messages if not s] == ['Look around: 25% seen.', 'Look around: 50% seen.',
                                                  'Look around: 75% seen.'])

ultra = Sim('ultrawide')
ultra.ctl.start()
turned_ultra = ultra.sweep(3, 400)
check('an ultra-wide lens needs less turning than a normal one', turned_ultra is not None and turned_ultra < turned_wide,
      f'{turned_ultra} deg against {turned_wide} deg')

left = Sim('wide')
left.ctl.start()
turned_left = left.sweep(3, 400, direction=-1)
check('turning left works too', turned_left is not None and left.last.startswith('Look around done.'), left.last)

fast = Sim('wide')
fast.ctl.start()
fast.turn(179)
fast.turn(-179)
check('one jump to the other side does not count as seeing everything in between', fast.ctl.active)

half = Sim('wide')
half.ctl.start()
half.sweep(3, 150)
for _ in range(40):
    half.turn(150, status='lost')                  # tracking lost for 4 seconds of fake time
check('losing tracking stops the scan and says so', not half.ctl.active and half.last.startswith(
    'Tracking lost, look around stopped.'), half.last)
check('a stopped scan says how much it saw and what is missing', 'I saw ' in half.last and 'behind you' in half.last
      and 'to your left' in half.last, half.last)
check('a stopped scan is spoken', half.messages[-1][1] is True)

brief = Sim('wide')
brief.ctl.start()
brief.sweep(3, 90)
brief.turn(100, status='lost')
brief.turn(105)
check('a short loss of tracking does not stop the scan', brief.ctl.active)

stale = Sim('wide')
stale.ctl.start()
for _ in range(40):
    stale.pos = position(rot(10), seconds_ago=60)  # the camera program stopped: an old position
    stale.now += 0.1
    stale.ctl.tick()
check('an old position counts as lost tracking', not stale.ctl.active and stale.last.startswith('Tracking lost'),
      stale.last)

slow = Sim('wide')
slow.ctl.start()
slow.now += 61
slow.turn(5)
check('a scan that takes more than a minute stops', not slow.ctl.active and 'took too long' in slow.last, slow.last)

moved = Sim('wide')
moved.ctl.start()
moved.pos = position(rot(20), session=2)
moved.ctl.tick()
check('a new map during the scan stops it', not moved.ctl.active and 'map changed' in moved.last, moved.last)

again = Sim('wide')
again.ctl.start()
again.messages.clear()
check('starting twice does not restart it', again.ctl.start() is False and again.last.startswith('Already looking around'))

nope = Sim('wide')
nope.pos = position(np.eye(3), status='lost')
check('no scan while tracking is lost', nope.ctl.start() is False and not nope.ctl.active and
      nope.last == 'Tracking is not ready, so I cannot look around.', nope.last)
nope.pos = None
check('no scan before the camera has started', nope.ctl.start() is False)

stop = Sim('wide')
stop.ctl.start()
stop.ctl.cancel()
check('a scan can be cancelled', not stop.ctl.active and stop.last == 'Look around cancelled.' and
      stop.messages[-1][1] is False)

phrases = ['look around', 'Look all around', 'scan the room', 'please scan around', 'can you look round']
check('"look around" is recognised in different wordings', all(lookaround.LOOK_AROUND.search(p) for p in phrases))
others = ['what is ahead', 'take me to the chair', 'look at the chair', 'turn around', 'where is the laptop']
check('other commands are not mistaken for it', not any(lookaround.LOOK_AROUND.search(p) for p in others))

# ---------------------------------------------------------------- fresh and old in the printed answers
print('--- fresh and old markers')
tmp = tempfile.mkdtemp()
tracker.DB_PATH = os.path.join(tmp, 'test.db')
tracker.POSITION_FILE = os.path.join(tmp, 'position.json')
tracker.LIVE_SCENE_FILE = os.path.join(tmp, 'live_scene.json')
lookaround.LAST_SCAN_FILE = os.path.join(tmp, 'last_scan.json')

conn = connect()
sid = new_session(conn)
conn.execute('UPDATE sessions SET units_per_meter = ? WHERE id = ?', (10.0, sid))
now = datetime.datetime.now()


def add(label, x, y, z, seconds_ago):
    conn.execute('INSERT INTO detections (label, confidence, x, y, z, pos_source, session_id, timestamp) '
                 'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                 (label, 0.9, x, y, z, 'map', sid, (now - datetime.timedelta(seconds=seconds_ago)).isoformat()))
    conn.commit()


add('chair', 0.0, 0.0, 5.0, 5)         # 0.5 m ahead, seen 5 seconds ago
add('couch', -6.0, 0.0, 0.0, 300)      # 0.6 m to the left, seen 5 minutes ago
write_position(sid, np.eye(3), np.zeros((3, 1)), 'tracking')

check('obstacles_ahead still returns the same four values (live_run.py relies on it)',
      guidance.obstacles_ahead([{'label': 'chair', 'position': np.array([0.0, 0.0, 5.0])}],
                               np.zeros(3), np.eye(3), 10.0) == [('chair', 5.0, 0.0, 'chair')])

answer = guidance.describe('what is ahead')
check('a recently seen object is marked fresh', 'chair, about 50 cm, straight ahead, fresh' in answer['text'], answer['text'])
check('an old one says how long ago', 'couch, about 60 cm, to the left, old, seen 5 min ago' in answer['text'],
      answer['text'])
check('the spoken answer is unchanged: object and distance only', answer['short'] == 'Chair, 50 cm.'
      and 'old' not in answer['spoken_full'] and 'fresh' not in answer['spoken_full'],
      f"{answer['short']} | {answer['spoken_full']}")
check('the answer says no look-around has been done yet', 'No look-around yet' in answer['text'], answer['text'])

lookaround.save_last_scan(sid, 94, 'Not seen: behind you.')
answer = guidance.describe('what is ahead')
check('after a scan the answer says how much of the circle was seen',
      'Last look-around just now saw 94% of the circle. Not seen: behind you.' in answer['text'], answer['text'])
lookaround.save_last_scan(sid + 1, 100, '')
check('a scan from another map is not used', 'No look-around yet' in guidance.describe('what is ahead')['text'])

write_live_scene(sid, [{'label': 'couch', 'x': -6.0, 'y': 0.0, 'z': 0.0}])
answer = guidance.describe('what is ahead')
check('an object in the live view right now is fresh even if its stored time is old',
      'couch, about 60 cm, to the left, fresh' in answer['text'], answer['text'])
os.remove(tracker.LIVE_SCENE_FILE)

conn.execute('UPDATE detections SET timestamp = ? WHERE label = ?',
             ((now - datetime.timedelta(seconds=300)).isoformat(), 'chair'))
conn.commit()
guidance.SPEAK_AGE = True
answer = guidance.describe('what is ahead')
check('with SPEAK_AGE on, the spoken line adds "old" for an old object', answer['short'] == 'Chair, 50 cm, old.'
      and 'couch, 60 cm, to the left, old' in answer['spoken_full'], f"{answer['short']} | {answer['spoken_full']}")
guidance.SPEAK_AGE = False
check('SPEAK_AGE is off by default (the boss approved object and distance only)',
      guidance.describe('what is ahead')['short'] == 'Chair, 50 cm.')
conn.close()

print('PASS' if not failures else f'FAIL: {len(failures)} check(s) failed: ' + '; '.join(failures))
sys.exit(1 if failures else 0)
