"""Look-around scan: turn once on the spot and the system records which directions it has really seen.

The camera only sees a slice in front of it (about 65 degrees on a normal phone lens, about 105
on an ultra-wide one). Warnings for the sides and behind come from objects seen earlier, so they can
be out of date. This scan makes them fresh: you turn slowly all the way around, the tracker's
rotation says where the camera pointed, and every direction the lens swept over counts as seen.
Anything seen during the scan gets a new time stamp, so "what is ahead" can say fresh or old.

On the phone: press "Look around" on the page, or say "look around".
On a laptop (live_run.py running in another window):  python lookaround.py
It is an aid, not a safety device. Turning fast makes the tracker lose its place; if that happens
the scan stops and says how much it saw.
"""
import datetime
import json
import math
import os
import re
import threading
import time

import numpy as np

HFOV_DEG = {'wide': 65.0, 'ultrawide': 105.0}    # horizontal field of view per lens (typical phone values)
DEFAULT_LENS = 'wide'
BIN_DEG = 10                                      # the circle is split into 36 slices of 10 degrees
N_BINS = 360 // BIN_DEG
USE_FRACTION = 0.7                                # only the middle 70% of the lens counts (edges are blurry)
DONE_FRACTION = 0.97                              # finished when this share of the circle has been seen
MAX_SECONDS = 60.0
LOST_GIVE_UP_SECONDS = 3.0                        # tracking lost this long: stop and report
POSITION_MAX_AGE_SECONDS = 4.0                    # an older position means the camera program stopped
MIN_HORIZONTAL = 0.3                              # later pointing (almost) straight up or down: no yaw
MISSING_BINS = 3                                  # a direction counts as not seen when 3+ slices (30 deg) are empty
LAST_SCAN_FILE = 'last_scan.json'

LOOK_AROUND = re.compile(r'\b(look|scan)\s+(all\s+)?(around|round)\b|\bscan\s+(the\s+)?(room|area|surroundings)\b',
                         re.IGNORECASE)


def hfov(lens):
    return HFOV_DEG.get(lens, HFOV_DEG[DEFAULT_LENS])


def age_words(seconds):
    """'just now', '45 s ago', '3 min ago', '2 h ago'."""
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return 'just now'
    if seconds < 60:
        return f'{int(seconds)} s ago'
    if seconds < 3600:
        return f'{int(seconds // 60)} min ago'
    return f'{int(seconds // 3600)} h ago'


class YawTracker:
    """The angle turned (degrees, right is positive) about the vertical axis the camera had at the start.

    R is the tracker's world-to-camera rotation (camera x right, y down, z forward): the camera's
    forward direction in world coordinates is the third row of R, its down direction the second row.
    """

    def __init__(self, R0):
        R0 = np.asarray(R0, float)
        self.down = R0[1] / np.linalg.norm(R0[1])      # the camera's own down axis at the start
        forward = R0[2] - (R0[2] @ self.down) * self.down
        self.e1 = forward / np.linalg.norm(forward)    # (the forward axis is always square to the down axis)
        self.e2 = np.cross(self.down, self.e1)         # to the right of the starting direction

    def yaw(self, R):
        """Degrees turned since the start, or None when the camera now points (almost) straight up or down."""
        R = np.asarray(R, float)
        f = R[2] - (R[2] @ self.down) * self.down
        if float(np.linalg.norm(f)) < MIN_HORIZONTAL:
            return None
        return math.degrees(math.atan2(float(f @ self.e2), float(f @ self.e1)))


def bins_covered(yaw_deg, hfov_deg):
    """The 10-degree slices (0 to 35, relative to the start) that lie inside the lens at this yaw."""
    half = hfov_deg * USE_FRACTION / 2.0
    first = math.ceil((yaw_deg - half) / BIN_DEG)
    last = math.floor((yaw_deg + half) / BIN_DEG) - 1      # slices wholly inside the view only
    return {b % N_BINS for b in range(first, last + 1)}


SECTORS = ('in front', 'to your right', 'behind you', 'to your left')


def _sector_of(bin_index):
    """0 in front, 1 right, 2 behind, 3 left: the quarter of the circle the slice's middle lies in."""
    middle = bin_index * BIN_DEG + BIN_DEG / 2
    return int(((middle + 45) % 360) // 90)


def missing_text(covered):
    """'Not seen: behind you, to your left.' for the directions with gaps; '' when nothing is missing."""
    empty = [0, 0, 0, 0]
    for b in range(N_BINS):
        if b not in covered:
            empty[_sector_of(b)] += 1
    names = [SECTORS[i] for i in range(4) if empty[i] >= MISSING_BINS]
    return ('Not seen: ' + ', '.join(names) + '.') if names else ''


# ----------------------------------------------------------------- the last scan (for the text answers)
def _now():
    return datetime.datetime.now()


def save_last_scan(session_id, percent, missing, path=None):
    path = path or LAST_SCAN_FILE
    data = {'session_id': session_id, 'timestamp': _now().isoformat(), 'percent': int(percent), 'missing': missing}
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass                                           # a note about the scan is never worth a crash


def load_last_scan(session_id, path=None):
    """The last scan of this session with 'age_seconds', or None."""
    path = path or LAST_SCAN_FILE
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get('session_id') != session_id:
            return None
        data['age_seconds'] = (_now() - datetime.datetime.fromisoformat(data['timestamp'])).total_seconds()
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def coverage_note(session_id, path=None):
    """One sentence on the last look-around of this session, for the printed answers."""
    last = load_last_scan(session_id, path)
    if last is None:
        return 'No look-around yet, so only what the camera saw straight ahead is known.'
    note = f"Last look-around {age_words(last['age_seconds'])} saw {last['percent']}% of the circle."
    return note + (' ' + last['missing'] if last.get('missing') else '')


# ------------------------------------------------------------------------------------ the scan itself
def _usable(pos, clock_now=None):
    """A tracking position with a rotation that is not stale."""
    if not pos or pos.get('status') != 'tracking' or 'rotation' not in pos:
        return False
    try:
        age = ((clock_now or _now()) - datetime.datetime.fromisoformat(pos['timestamp'])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return True                                    # no time stamp to check (tests)
    return age <= POSITION_MAX_AGE_SECONDS


class ScanController:
    """One look-around scan. Call start(), then tick() once per camera frame."""

    def __init__(self, notify, read_position, get_lens=lambda: DEFAULT_LENS, clock=time.monotonic, save=True):
        self.notify = notify                           # notify(text, speak) like PhonePipeline.notify
        self.read_position = read_position
        self.get_lens = get_lens
        self.clock = clock
        self.save = save
        self._lock = threading.RLock()        # start() can come from a web request while tick() runs
        self.reset()

    def reset(self):
        self.active = False
        self.yaw = None
        self.covered = set()
        self.started = 0.0
        self.session = None
        self.lost_since = None
        self.next_progress = 25
        self.fov = HFOV_DEG[DEFAULT_LENS]

    @property
    def fraction(self):
        return len(self.covered) / N_BINS

    def _start(self):
        if self.active:
            self.notify('Already looking around. Keep turning slowly.', True)
            return False
        pos = self.read_position()
        if not _usable(pos):
            self.notify('Tracking is not ready, so I cannot look around.', True)
            return False
        self.yaw = YawTracker(pos['rotation'])
        self.fov = hfov(self.get_lens())
        self.session = pos.get('session_id')
        self.covered = set()
        self.started = self.clock()
        self.lost_since = None
        self.next_progress = 25
        self.active = True
        self._mark(pos)
        self.notify('Turn slowly all the way around on the spot.', True)
        return True

    def _cancel(self):
        if self.active:
            self.active = False
            self.notify('Look around cancelled.', False)

    def _tick(self):
        if not self.active:
            return
        if self.clock() - self.started > MAX_SECONDS:
            self._finish('Look around stopped, it took too long.')
            return
        pos = self.read_position()
        if not _usable(pos):
            if self.lost_since is None:
                self.lost_since = self.clock()
            elif self.clock() - self.lost_since > LOST_GIVE_UP_SECONDS:
                self._finish('Tracking lost, look around stopped.')
            return
        self.lost_since = None
        if pos.get('session_id') != self.session:
            self._finish('The map changed, look around stopped.')
            return
        self._mark(pos)
        percent = 100 * self.fraction
        while self.next_progress < 100 and percent >= self.next_progress:
            self.notify(f'Look around: {self.next_progress}% seen.', False)
            self.next_progress += 25
        if self.fraction >= DONE_FRACTION:
            self._finish(None)

    def start(self):
        with self._lock:
            return self._start()

    def cancel(self):
        with self._lock:
            self._cancel()

    def tick(self):
        with self._lock:
            self._tick()

    def _mark(self, pos):
        yaw = self.yaw.yaw(pos['rotation'])
        if yaw is not None:
            self.covered |= bins_covered(yaw, self.fov)

    def _finish(self, stopped):
        self.active = False
        percent = int(round(100 * self.fraction))
        missing = missing_text(self.covered)
        if stopped is None:
            text = 'Look around done. ' + (missing or 'I have seen all around you.')
        else:
            text = f'{stopped} I saw {percent} percent. {missing}'.strip()
        if self.save:
            save_last_scan(self.session, percent, missing)
        self.notify(text, True)


def main():
    """Laptop use: live_run.py runs in another window, this one guides the turn."""
    import argparse
    import sys

    from tracker import read_position
    parser = argparse.ArgumentParser(description='Turn on the spot so the system sees all around you.')
    parser.add_argument('--lens', choices=sorted(HFOV_DEG), default=DEFAULT_LENS,
                        help='field of view of the camera: wide is about 65 degrees (laptop webcams and normal '
                             'phone lenses), ultrawide about 105')
    args = parser.parse_args()

    def notify(text, speak=True, **_):
        print(text)

    scan = ScanController(notify, read_position, lambda: args.lens)
    if not scan.start():
        sys.exit(1)
    try:
        while scan.active:
            time.sleep(0.2)
            scan.tick()
    except KeyboardInterrupt:
        scan.cancel()


if __name__ == '__main__':
    main()
