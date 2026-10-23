"""Distance between two stored detections.

    python distance_diff.py            asks for the two IDs
    python distance_diff.py 12 15      uses IDs 12 and 15
"""
import math
import sys

from tracker import connect, get_units_per_meter

conn = connect()
rows = conn.execute('SELECT id, label, x, y, z, timestamp, session_id FROM detections '
                    'ORDER BY timestamp').fetchall()

if len(sys.argv) == 3:
    id1, id2 = int(sys.argv[1]), int(sys.argv[2])
else:
    print('=== Available detections ===')
    for id_, label, x, y, z, ts, sid in rows:
        tag = 'legacy' if sid == 0 else f'session {sid}'
        print(f'ID {id_}: {label} at ({x:.2f}, {y:.2f}, {z:.2f}) | {tag} | {ts}')
    print('\n=== Compare two detections ===')
    id1 = int(input('Enter first ID: '))
    id2 = int(input('Enter second ID: '))

by_id = {r[0]: r for r in rows}
if id1 not in by_id or id2 not in by_id:
    sys.exit('One of those IDs does not exist.')
a, b = by_id[id1], by_id[id2]
dist = math.dist(a[2:5], b[2:5])

print(f'\n{a[1]} (ID {id1}) at {a[5]}')
print(f'{b[1]} (ID {id2}) at {b[5]}')
print(f'Distance between them: {dist:.2f} units')

if a[6] != b[6] or a[6] == 0:
    print('\nWarning: these detections come from different sessions (or legacy data).')
    print('Each run starts at its own origin, so this number is not a real distance.')
else:
    upm = get_units_per_meter(conn, a[6])
    if upm:
        print(f'About {dist / upm:.2f} meters (scale set with set_scale.py)')
conn.close()
