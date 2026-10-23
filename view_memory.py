"""Show what is stored in spatial_memory.db.

    python view_memory.py                 newest session only (old test rows hidden)
    python view_memory.py --session 17    one chosen session
    python view_memory.py --label phone   only detections whose label contains this text
    python view_memory.py --all           everything, including legacy rows

Positions are in map units, and also in meters when the session has a scale (set_scale.py).
Coordinates are only comparable inside one session.
"""
import argparse

from tracker import connect, get_units_per_meter

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true', help='show every row, including legacy rows')
parser.add_argument('--session', type=int, default=None, help='show only this session')
parser.add_argument('--label', default=None, help='only labels containing this text')
args = parser.parse_args()

conn = connect()
rows = conn.execute('SELECT id, label, confidence, x, y, z, pos_source, session_id, timestamp '
                    'FROM detections ORDER BY id').fetchall()
if not rows:
    print('No detections stored yet. Run: python live_run.py')
    raise SystemExit

latest = max((r[7] for r in rows if r[7]), default=0)
if args.all:
    title, shown = 'all sessions', rows
elif args.session is not None:
    title, shown = f'session {args.session}', [r for r in rows if r[7] == args.session]
else:
    title, shown = f'newest session ({latest})', [r for r in rows if r[7] == latest and latest]
if args.label:
    shown = [r for r in shown if args.label.lower() in (r[1] or '').lower()]

print(f'=== Detections in spatial_memory.db: {title} ===')
last_session = None
for id_, label, conf, x, y, z, source, session, ts in shown:
    if session != last_session:
        last_session = session
        if session:
            scale = get_units_per_meter(conn, session)
            started = conn.execute('SELECT started_at FROM sessions WHERE id = ?', (session,)).fetchone()
            print(f'--- Session {session}'
                  + (f' (started {started[0][:19]})' if started and started[0] else '')
                  + (f', scale {scale:.2f} map units per meter' if scale else ', no scale set'))
        else:
            print('--- Legacy rows (placeholder positions, not real)')
    scale = get_units_per_meter(conn, session) if session else None
    meters = f' = ({x / scale:.2f}, {y / scale:.2f}, {z / scale:.2f}) m' if scale else ''
    where = f'session {session}' if session else 'legacy'
    print(f'ID {id_}: {label} | Conf: {conf:.2f} | Pos: ({x:.2f}, {y:.2f}, {z:.2f}){meters} '
          f'[{source}] | {where} | {ts}')

hidden = len(rows) - len(shown)
if not shown:
    print('(nothing matches)')
if hidden and not args.all:
    print(f'\n{hidden} other rows hidden (older sessions and legacy rows). Use --all to see them.')
