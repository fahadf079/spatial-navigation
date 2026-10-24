"""Give the current session a real-world scale.

live_run.py must be RUNNING in another terminal while you use this, and both terminals must be
inside the project folder (cd path\\to\\spatial-navigation).

Use a SHORT, measured move (for example 0.3 m along a tape measure). Long walks make the
tracker lose its place, and a scale from a lost run is wrong.
"""
import datetime
import math
import sys
import threading

from tracker import connect, read_position

MAX_AGE_SECONDS = 3.0
MIN_METERS, MAX_METERS = 0.1, 3.0
MIN_UNITS = 0.05                  # below this the tracker did not follow the movement
USUAL_UNITS_PER_METER = (0.5, 200.0)
POLL_SECONDS = 0.1


def age_seconds(pos):
    try:
        return (datetime.datetime.now() - datetime.datetime.fromisoformat(pos['timestamp'])).total_seconds()
    except (KeyError, ValueError):
        return 1e9


def fresh_position(step):
    pos = read_position()
    if pos is None:
        sys.exit('No position found. Start live_run.py first (in the project folder).')
    age = age_seconds(pos)
    if age > MAX_AGE_SECONDS:
        sys.exit(f'The position is not updating (last update {age:.0f} s ago) at the {step}.\n'
                 'Keep live_run.py running in another terminal while you use set_scale.py.')
    if pos.get('status') != 'tracking':
        sys.exit(f"Tracking status is '{pos.get('status')}' at the {step}. Wait for 'tracking' and try again.")
    return pos


class Watcher(threading.Thread):
    """Watches the position file while you move, so a reset or a stall is caught."""

    def __init__(self, session_id):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.stop_event = threading.Event()
        self.problem = None

    def run(self):
        while not self.stop_event.is_set():
            pos = read_position()
            if pos is not None and self.problem is None:     # None = file busy, just try again
                if pos['session_id'] != self.session_id:
                    self.problem = 'the tracker started a new session (it lost its place)'
                elif age_seconds(pos) > MAX_AGE_SECONDS:
                    self.problem = 'live_run.py stopped updating the position'
            self.stop_event.wait(POLL_SECONDS)


def ask_meters():
    raw = input(f'How far will you move the camera, in meters? ({MIN_METERS} to {MAX_METERS}, '
                'measure it with a tape) ').strip().replace(',', '.')
    try:
        meters = float(raw)
    except ValueError:
        sys.exit('Please type a number such as 0.3.')
    if not MIN_METERS <= meters <= MAX_METERS:
        sys.exit(f'Use a distance between {MIN_METERS} and {MAX_METERS} m. '
                 'Long moves make the tracker lose its place.')
    return meters


meters = ask_meters()
input('Hold the camera at the start point, with live_run.py showing "tracking", and press Enter... ')
a = fresh_position('start point')
watcher = Watcher(a['session_id'])
watcher.start()
input(f'Now slide the camera {meters} m in a straight line, keep it pointed at the same textured '
      'area, hold it still at the end and press Enter... ')
watcher.stop_event.set()
b = fresh_position('end point')
if watcher.problem:
    sys.exit(f'Measurement discarded: {watcher.problem}. Try again.')
if a['session_id'] != b['session_id']:
    sys.exit('The tracking session changed while measuring. Try again.')

units = math.dist((a['x'], a['y'], a['z']), (b['x'], b['y'], b['z']))
if units < MIN_UNITS:
    sys.exit(f'The tracker only registered {units:.3f} map units of movement for a {meters} m move, '
             'so it did not follow the camera. Nothing was saved. Move smoothly toward a '
             'textured area and try again.')

per_meter = units / meters
low, high = USUAL_UNITS_PER_METER
if not low <= per_meter <= high:
    print(f'Warning: {per_meter:.3f} map units per meter is unusual (expected roughly {low} to {high}).')
    print('Check that you moved the distance you typed and that tracking stayed on.')
    if input('Save it anyway? (y/N) ').strip().lower() != 'y':
        sys.exit('Nothing was saved.')

conn = connect()
conn.execute('UPDATE sessions SET units_per_meter = ? WHERE id = ?', (per_meter, a['session_id']))
conn.commit()
conn.close()
print(f'Session {a["session_id"]}: {per_meter:.3f} map units per meter saved.')
