"""Compare the system's distance between two stored objects with a tape measure.

    python check_accuracy.py 75 78 0.5

75 and 78 are detection IDs from view_memory.py (both from the same session, with a scale set),
and 0.5 is the distance you measured between the real objects, in meters.
The pilot target is an error of up to about 1 m.
"""
import math
import sys

from tracker import connect, get_units_per_meter

TARGET_METERS = 1.0

if len(sys.argv) != 4:
    sys.exit('Usage: python check_accuracy.py ID1 ID2 TAPE_METERS   (example: python check_accuracy.py 75 78 0.5)')
try:
    id1, id2, tape = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3].replace(',', '.'))
except ValueError:
    sys.exit('ID1 and ID2 must be whole numbers from view_memory.py, and TAPE_METERS a number such as 0.5.')

conn = connect()
rows = {}
for id_ in (id1, id2):
    row = conn.execute('SELECT label, x, y, z, session_id FROM detections WHERE id = ?', (id_,)).fetchone()
    if row is None:
        sys.exit(f'No detection with ID {id_}. Look at python view_memory.py.')
    rows[id_] = row
(l1, x1, y1, z1, s1), (l2, x2, y2, z2, s2) = rows[id1], rows[id2]
if s1 != s2 or not s1:
    sys.exit('These two detections are from different sessions (or legacy data), so the distance is meaningless.')
scale = get_units_per_meter(conn, s1)
if not scale:
    sys.exit(f'Session {s1} has no scale. Run set_scale.py during a run first.')

measured = math.dist((x1, y1, z1), (x2, y2, z2)) / scale
error = abs(measured - tape)
print(f'{l1} (ID {id1}) to {l2} (ID {id2}), session {s1}')
print(f'Tape: {tape:.2f} m   System: {measured:.2f} m   Error: {error:.2f} m')
print(f'Within the pilot target of {TARGET_METERS:.1f} m: ' + ('YES' if error <= TARGET_METERS else 'NO'))
