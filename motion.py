"""Learning mode: set the map scale from the phone's motion sensors, with no tape measure.

A single camera cannot see real-world size, so every map needs a scale (map units per meter).
`set_scale.py` measures it with a tape. For a blind user that is not possible, so the phone page
has a learning mode: at the start the user walks about eight steps at a normal pace while
  * the accelerometer counts the steps, and
  * the camera tracker measures how far the camera moved in map units.
Scale = map units walked / (steps x step length).

Honest limits (see README):
  * The step length is an assumption (default 0.70 m, or 0.415 x height if the height is given).
    A real step that is 15% shorter or longer makes the scale 15% off. That is still far better
    than the default guess, which can be wrong by 2 times or more, but it is not a tape measure.
  * Steps are counted from the size of the up and down bounce, so the phone works in a hand or a
    pocket, but a bump inside the walk can add or hide a step (about one step in eight).
  * Learning is refused (nothing is saved) if tracking was lost during the walk, too few steps were
    seen, the walk had an uneven rhythm, or the result is outside the usual range.
Nothing here needs the web server: the functions only use numpy so they can be tested alone.
"""
import json
import math
import time

import numpy as np

STEP_LENGTH_M = 0.70            # default step length
HEIGHT_TO_STEP = 0.415          # step length = this x body height (common rule of thumb)
MIN_STEP_M, MAX_STEP_M = 0.40, 1.00
SAMPLE_RATE = 50.0              # Hz, the accelerometer is resampled to this
MIN_PEAK = 0.8                  # m/s^2: smaller bounces are noise
MIN_STEP_GAP = 0.28             # s: two steps are never closer than this
MAX_STEP_GAP = 1.2              # s: a longer pause ends a walking bout
MIN_STEPS = 4
TARGET_STEPS = 8
MAX_CADENCE_CV = 0.35           # uneven rhythm is rejected
MIN_CADENCE, MAX_CADENCE = 1.1, 3.0     # steps per second; outside this, steps were probably missed or double counted
IDLE_END_SECONDS = 2.0          # standing still this long after the walk ends learning
LEARN_TIMEOUT_SECONDS = 40.0
USUAL_UNITS_PER_METER = (0.5, 200.0)     # same range as set_scale.py
MAX_EDGE_GAP = 0.6              # s: the track must cover the start and end of the walk
MIN_TRACK_POINTS = 8


def step_length_from_height_cm(height_cm):
    """Step length in meters from body height, limited to a sensible range."""
    return float(min(MAX_STEP_M, max(MIN_STEP_M, HEIGHT_TO_STEP * float(height_cm) / 100.0)))


def _moving_average(x, n):
    n = max(1, int(n))
    if n == 1:
        return np.asarray(x, float).copy()
    pad = n // 2
    xp = np.pad(np.asarray(x, float), (pad, n - 1 - pad), mode='edge')
    return np.convolve(xp, np.ones(n) / n, mode='valid')


def detect_steps(t, magnitude):
    """Times (same clock as t, seconds) of the steps in the longest walking bout.

    t, magnitude: accelerometer samples, magnitude = length of the acceleration vector including
    gravity (m/s^2), so the phone's orientation does not matter."""
    t = np.asarray(t, float)
    magnitude = np.asarray(magnitude, float)
    if len(t) < 20 or t[-1] - t[0] < 1.5:
        return []
    order = np.argsort(t)
    t, magnitude = t[order], magnitude[order]
    tu = np.arange(t[0], t[-1], 1.0 / SAMPLE_RATE)
    mu = np.interp(tu, t, magnitude)
    dynamic = mu - _moving_average(mu, SAMPLE_RATE * 1.0)       # remove gravity and slow tilt
    smooth = _moving_average(dynamic, SAMPLE_RATE * 0.12)
    inner = smooth[1:-1]
    peaks = np.where((inner > smooth[:-2]) & (inner >= smooth[2:]) & (inner > MIN_PEAK))[0] + 1
    if len(peaks) == 0:
        return []
    threshold = max(MIN_PEAK, 0.4 * float(np.median(smooth[peaks])))   # robust against a few big bumps
    peaks = peaks[smooth[peaks] >= threshold]
    taken = []
    for i in peaks[np.argsort(-smooth[peaks])]:                  # strongest first, keep them apart
        if all(abs(i - j) >= MIN_STEP_GAP * SAMPLE_RATE for j in taken):
            taken.append(i)
    times = np.sort(tu[taken])
    if len(times) == 0:
        return []
    bouts, current = [], [times[0]]
    for a, b in zip(times[:-1], times[1:]):
        if b - a > MAX_STEP_GAP:
            bouts.append(current)
            current = []
        current.append(b)
    bouts.append(current)
    return [float(x) for x in max(bouts, key=len)]


def path_length(points, t0, t1, bin_s=0.25):
    """Length (map units) of the camera track between t0 and t1, and the number of points used.
    points: (time, x, y, z). The track is thinned to one point per bin so small tracker jitter does
    not add up to extra distance."""
    chosen = [(t, x, y, z) for t, x, y, z in points if t0 <= t <= t1]
    if len(chosen) < 2:
        return 0.0, len(chosen)
    bins = {}
    for t, x, y, z in chosen:
        bins[int((t - t0) / bin_s)] = (x, y, z)
    keys = sorted(bins)
    bins[keys[0]] = chosen[0][1:]            # keep the very first and very last point, so the ends
    bins[keys[-1]] = chosen[-1][1:]          # of the walk are not shortened by the thinning
    seq = [bins[k] for k in keys]
    return float(sum(math.dist(a, b) for a, b in zip(seq[:-1], seq[1:]))), len(chosen)


class LearningSession:
    """Collects accelerometer samples and camera positions for one learning walk."""

    def __init__(self, session_id, step_length=STEP_LENGTH_M, started=None):
        self.session_id = session_id
        self.step_length = float(step_length)
        self.started = time.monotonic() if started is None else started
        self.imu = []                    # (client time ms, magnitude)
        self.offset = None               # server time minus client time (s), smallest latency seen
        self.positions = []              # (server time, x, y, z) while tracking
        self.lost_times = []             # server times when tracking was not 'tracking'
        self.session_changed = False
        self._last_stamp = None

    # ---- input
    def add_imu(self, samples, received):
        """samples: [[client time in ms, ax, ay, az], ...] (acceleration including gravity)."""
        good = []
        for s in samples:
            try:
                t, ax, ay, az = (float(v) for v in s[:4])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (t, ax, ay, az)):
                continue
            good.append((t, math.sqrt(ax * ax + ay * ay + az * az)))
        if not good:
            return
        self.imu.extend(good)
        latency_offset = received - good[-1][0] / 1000.0
        self.offset = latency_offset if self.offset is None else min(self.offset, latency_offset)

    def add_position(self, pos, now):
        """pos: the dict from tracker.read_position()."""
        if pos is None:
            return
        if pos.get('session_id') != self.session_id:
            self.session_changed = True
            return
        if pos.get('status') != 'tracking':
            self.lost_times.append(now)
            return
        stamp = pos.get('timestamp')
        if stamp is not None and stamp == self._last_stamp:
            return                         # the same frame again
        self._last_stamp = stamp
        self.positions.append((now, float(pos['x']), float(pos['y']), float(pos['z'])))

    # ---- analysis
    def step_times(self):
        """Step times on the server clock."""
        if self.offset is None or len(self.imu) < 20:
            return []
        t = np.array([s[0] / 1000.0 + self.offset for s in self.imu])
        m = np.array([s[1] for s in self.imu])
        return detect_steps(t, m)

    def result(self):
        """{'ok': bool, 'reason': str, 'steps': int, 'units_per_meter': float or None, ...}"""
        steps = self.step_times()
        out = {'ok': False, 'reason': '', 'steps': len(steps), 'units_per_meter': None,
               'imu_meters': None, 'map_units': None, 'step_length': self.step_length}
        if self.session_changed:
            out['reason'] = 'tracking started a new map during the walk'
            return out
        if len(steps) < MIN_STEPS:
            out['reason'] = f'only {len(steps)} steps were detected (need at least {MIN_STEPS})'
            return out
        intervals = np.diff(steps)
        cv = float(np.std(intervals) / np.mean(intervals))
        if cv > MAX_CADENCE_CV:
            out['reason'] = 'the walking rhythm was too uneven'
            return out
        cadence = 1.0 / float(np.median(intervals))
        if not MIN_CADENCE <= cadence <= MAX_CADENCE:
            out['reason'] = (f'the walking pace ({cadence:.1f} steps a second) was outside the normal range, '
                             'so steps may have been missed; walk at a normal pace')
            return out
        half = 0.5 * float(np.median(intervals))
        t0, t1 = steps[0] - half, steps[-1] + half
        if any(t0 - 0.5 <= x <= t1 + 0.5 for x in self.lost_times):
            out['reason'] = 'tracking was lost during the walk'
            return out
        length, used = path_length(self.positions, t0, t1)
        if used < MIN_TRACK_POINTS:
            out['reason'] = 'the camera tracked too little of the walk'
            return out
        first = min(p[0] for p in self.positions if t0 <= p[0] <= t1)
        last = max(p[0] for p in self.positions if t0 <= p[0] <= t1)
        if first - t0 > MAX_EDGE_GAP or t1 - last > MAX_EDGE_GAP:
            out['reason'] = 'the camera did not track the start or the end of the walk'
            return out
        meters = len(steps) * self.step_length
        units_per_meter = length / meters
        out.update(imu_meters=meters, map_units=length)
        low, high = USUAL_UNITS_PER_METER
        if not low <= units_per_meter <= high:
            out['reason'] = f'the result ({units_per_meter:.2f} map units per meter) is outside the usual range'
            return out
        out.update(ok=True, units_per_meter=float(units_per_meter), reason='')
        return out


class LearningController:
    """Runs learning mode for the phone page. Talks to the outside world only through the
    functions given to it, so it can be tested without a phone, camera or database file.

    notify(text, speak=True, learn=None): spoken-style messages. `learn` is 'done', 'failed' or
    'skipped' on the message that ends learning, so the phone page knows it can carry on.
    get_position(): the latest camera position dict (or None).
    get_scale(session_id): a measured scale for the session, or None.
    save_scale(session_id, units_per_meter): store the scale."""

    def __init__(self, notify, get_position, get_scale, save_scale, clock=time.monotonic):
        self.notify = notify
        self.get_position = get_position
        self.get_scale = get_scale
        self.save_scale = save_scale
        self.clock = clock
        self.session = None
        self.state = 'idle'              # idle -> learning -> idle
        self.last_result = None

    def reset(self):
        self.session = None
        self.state = 'idle'

    def start(self, step_length=None):
        pos = self.get_position()
        if pos is None or pos.get('status') != 'tracking':
            self.notify('Not ready. Wait for tracking.', True, 'failed')
            return
        if self.get_scale(pos['session_id']):
            self.notify('Scale already known.', True, 'skipped')
            return
        length = STEP_LENGTH_M if step_length is None else float(
            min(MAX_STEP_M, max(MIN_STEP_M, float(step_length))))
        self.session = LearningSession(pos['session_id'], length, started=self.clock())
        self.state = 'learning'
        self.notify(f'Learning mode. Walk {TARGET_STEPS} steps, then stop.', True, None)
        self.notify(f'Learning started, step length {length:.2f} m.', False, None)

    def cancel(self):
        if self.state == 'learning':
            self.reset()
            self.notify('Learning cancelled.', True, 'failed')

    def on_motion(self, samples):
        if self.state == 'learning':
            self.session.add_imu(samples, self.clock())

    def on_unavailable(self):
        self.notify('Motion sensors are not available, so learning mode is off.', True, 'failed')
        self.reset()

    def tick(self):
        """Call after every camera frame. Reads the position and ends learning when the walk is over."""
        if self.state != 'learning':
            return
        now = self.clock()
        self.session.add_position(self.get_position(), now)
        steps = self.session.step_times()
        # Two or more steps and then standing still ends the walk. That is also how "too few steps" is
        # found quickly, instead of making the user wait for the time-out. One step alone is ignored (a bump).
        walked_then_stopped = len(steps) >= 2 and now - steps[-1] >= IDLE_END_SECONDS
        if walked_then_stopped or now - self.session.started >= LEARN_TIMEOUT_SECONDS:
            self.finish()

    def finish(self):
        result = self.session.result()
        session_id = self.session.session_id
        self.last_result = result
        self.reset()
        if result['ok']:
            self.save_scale(session_id, result['units_per_meter'])
            self.notify('Scale saved.', True, 'done')
            self.notify(f"Scale {result['units_per_meter']:.2f} map units per meter from {result['steps']} steps "
                        f"of {result['step_length']:.2f} m (about 20% uncertain).", False, None)
        else:
            self.notify('Learning failed. Say calibrate to try again.', True, 'failed')
            self.notify('Learning failed: ' + result['reason'] + '.', False, None)


def parse_message(raw):
    """Parse one JSON text message from the phone. Returns a dict, or None if it is not valid."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None
