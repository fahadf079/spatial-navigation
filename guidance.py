"""Directions to a stored object or person, "what is ahead" answers and obstacle warnings.
Used by navigate.py, voice.py, api.py and live_run.py.

A warning means anything known within about 1 m of the camera in ANY direction (a full 360 degrees,
not only straight ahead). The camera itself only sees forward, so things beside or behind you are
known only from what it saw earlier. Left, right and behind are worked out from the camera's current
rotation. Positions are approximate (the pilot target is about 1 m), so answers say so.
Everything here only reads the database and the live files.
"""
import datetime
import math
import re

import numpy as np

import lookaround
import people
from tracker import connect, effective_scale, read_live_scene, read_position

STRAIGHT_DEG = 10          # closer than this to the heading counts as "straight ahead"
BEHIND_DEG = 120           # farther than this: "turn around"
CORRIDOR_METERS = 0.5      # an object this close to the straight path is an obstacle
END_MARGIN_METERS = 0.3    # ignore things right next to the target itself
ALERT_METERS = 1.0         # a warning means anything known within about 1 m, in any direction (360 degrees)
LIMITED_SPEECH = True      # spoken answers are one short line (object and distance); the full text is still printed
POSITION_MAX_AGE_SECONDS = 4.0   # a position older than this means the camera program is not running
FRESH_SECONDS = 20.0       # an object seen within this long ago is "fresh"; anything older is "old"
SPEAK_AGE = False          # True: the spoken line adds ", old" for objects not seen recently (needs the boss's OK)

# Objects big enough to be in the way (COCO labels).
OBSTACLE_LABELS = {
    'chair', 'couch', 'bed', 'dining table', 'tv', 'refrigerator', 'oven', 'microwave', 'sink',
    'toilet', 'potted plant', 'bench', 'suitcase', 'backpack', 'umbrella', 'bicycle',
    'motorcycle', 'car', 'person', 'laptop',
}

AHEAD = re.compile(
    r"\b(?:what|who|anything|anyone|something|someone)\b.*\b(?:ahead|in front|front of me|next|near me|close|"
    r"around me|in my way|on my way)\b|\bwhat(?:'s| is) (?:next|there)\b|\b(?:is the way|is it|is the path) clear\b",
    re.I)
WHERE_WORDS = {'take', 'go', 'where', 'find', 'guide', 'bring', 'lead', 'locate', 'navigate', 'walk', 'show'}

STOPWORDS = {
    'take', 'me', 'to', 'the', 'a', 'an', 'go', 'where', 'is', 'are', 'find', 'navigate',
    'show', 'my', 'please', 'guide', 'bring', 'i', 'want', 'need', 'near', 'at', 'for', 'can',
    'you', 'and', 'of', 'in', 'on', 'locate', 'look', 'walk', 'lead', 'us', 'now', 'how',
    'do', 'get', 'toward', 'towards', 'direction',
}
SYNONYMS = {
    'sofa': 'couch', 'television': 'tv', 'monitor': 'tv', 'screen': 'tv', 'mobile': 'phone',
    'cellphone': 'phone', 'smartphone': 'phone', 'fridge': 'refrigerator', 'bike': 'bicycle',
    'plant': 'plant', 'desk': 'table', 'notebook': 'laptop', 'computer': 'laptop',
}


def one_line(text, max_words=14):
    """The first sentence of a long answer (two if the first is only 'OK.'), at most max_words words."""
    sentences = [x.strip() for x in re.split(r'(?<=[.!?])\s+', ' '.join(str(text).split())) if x.strip()]
    if not sentences:
        return ''
    out = sentences[0]
    if len(out.split()) <= 2 and len(sentences) > 1:
        out += ' ' + sentences[1]
    words = out.split()
    return ' '.join(words[:max_words]).rstrip(',;:') + ('' if len(words) <= max_words else '.')


def _cap(text):
    return text[:1].upper() + text[1:]


def warning_line(name, meters):
    """The one spoken line for an obstacle warning: the object and its distance, nothing else.

    No "ahead": a warning now covers every direction around you."""
    return f'{_cap(name)}, {meters:.1f} meters'


def _singular(word):
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 3 and word.endswith(('ches', 'shes', 'sses', 'xes', 'zes')):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def _words(text):
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        w = SYNONYMS.get(_singular(w), _singular(w))
        if w not in STOPWORDS:
            out.append(w)
    return out


def label_matches(query, label):
    """True if the query names this label: the whole phrase, or any single word of it."""
    q, lab = _words(query), _words(label)
    if not q or not lab:
        return False
    n = len(lab)
    if any(q[i:i + n] == lab for i in range(len(q) - n + 1)):
        return True
    return any(w in q for w in lab)


def _heading_and_angles(pos, target):
    """Horizontal turn (degrees, right is positive), height angle (up is positive), distance."""
    R = np.array(pos['rotation'], float)
    C = np.array([pos['x'], pos['y'], pos['z']], float)
    v = R @ (np.asarray(target, float) - C)           # target in camera coordinates
    horizontal = math.degrees(math.atan2(v[0], v[2]))
    vertical = math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2])))
    return horizontal, vertical, float(np.linalg.norm(v)), C


def side_text(angle):
    if abs(angle) <= STRAIGHT_DEG:
        return ''
    if abs(angle) > BEHIND_DEG:
        return 'behind you'
    side = 'right' if angle > 0 else 'left'
    return f'slightly {side}' if abs(angle) <= 30 else f'to the {side}'


def _dist_text(units, scale):
    if scale:
        m = units / scale
        return f'{m:.1f} m' if m >= 0.95 else f'{m * 100:.0f} cm'
    return f'{units:.1f} units'


def find_obstacles(rows, target_id, C, X, R, scale, limit=2):
    """Stored objects standing near the straight path from the camera C to the target X."""
    path = X - C
    length = float(np.linalg.norm(path))
    if length < 1e-6:
        return []
    u = path / length
    corridor = CORRIDOR_METERS * scale if scale else 0.2 * length
    margin = END_MARGIN_METERS * scale if scale else 0.1 * length
    found = []
    for id_, label, x, y, z in rows:
        if id_ == target_id or label not in OBSTACLE_LABELS:
            continue
        v = np.array([x, y, z], float) - C
        along = float(v @ u)
        side = float(np.linalg.norm(v - along * u))
        if 0.05 * length < along < length - margin and side < corridor:
            horizontal = math.degrees(math.atan2((R @ v)[0], (R @ v)[2]))
            found.append((along, label, horizontal))
    found.sort()
    return found[:limit]


def obstacles_around(objects, C, R, scale, only_obstacles=True, meters=ALERT_METERS):
    """Everything known within `meters` of the camera, in any direction (360 degrees).

    objects: dicts with 'label', 'position' (map units) and optionally 'say' (what to call it,
    for example a person's name) and 'age' (seconds since it was last seen). Returns dicts with
    'name', 'dist' (3D straight-line distance in map units), 'angle', 'label' and 'age', nearest
    first. The angle is the horizontal bearing from straight ahead (right is positive, beyond
    120 degrees means behind you).
    """
    if not scale:
        return []
    out = []
    for obj in objects:
        if only_obstacles and obj['label'] not in OBSTACLE_LABELS:
            continue
        v = np.asarray(obj['position'], float) - C
        dist = float(np.linalg.norm(v))
        if dist > meters * scale or dist < 1e-6:
            continue
        cam = R @ v
        angle = math.degrees(math.atan2(cam[0], cam[2]))
        out.append({'dist': dist, 'name': obj.get('say') or obj['label'], 'angle': angle,
                    'label': obj['label'], 'age': obj.get('age')})
    out.sort(key=lambda d: (d['dist'], d['name'], d['angle'], d['label']))
    return out


def obstacles_ahead(objects, C, R, scale, only_obstacles=True, meters=ALERT_METERS):
    """Same as obstacles_around, as [(name, distance in map units, angle, label)] (used by live_run.py)."""
    return [(d['name'], d['dist'], d['angle'], d['label'])
            for d in obstacles_around(objects, C, R, scale, only_obstacles, meters)]


def age_seconds(timestamp):
    """Seconds since an ISO time stamp, or None if it cannot be read."""
    try:
        return max(0.0, (datetime.datetime.now() - datetime.datetime.fromisoformat(timestamp)).total_seconds())
    except (TypeError, ValueError):
        return None


def is_fresh(age):
    return age is not None and age <= FRESH_SECONDS


def age_tag(age):
    """'fresh', or 'old, seen 3 min ago'; '' when the age is unknown."""
    if age is None:
        return ''
    return 'fresh' if is_fresh(age) else 'old, seen ' + lookaround.age_words(age)


def _fresher(a, b):
    known = [x for x in (a, b) if x is not None]
    return min(known) if known else None


def scene_items(conn, session):
    """Everything known to be around: stored objects, people seen just now, and the live view."""
    items = [{'label': label, 'position': np.array([x, y, z], float), 'say': label, 'age': age_seconds(ts)}
             for label, x, y, z, ts in conn.execute(
                 'SELECT label, x, y, z, timestamp FROM detections WHERE session_id = ?', (session,))]
    for name, who, x, y, z in people.recent_seen(conn, session):         # only people seen a moment ago
        items.append({'label': 'person', 'position': np.array([x, y, z], float),
                      'say': name + (f', {people.speak_who(who)}' if who else ''), 'age': 0.0})
    scene = read_live_scene()
    if scene.get('session_id') == session:
        for it in scene['items']:
            if it.get('name'):
                continue                         # a recognised person is already listed above
            items.append({'label': it['label'], 'position': np.array([it['x'], it['y'], it['z']], float),
                          'say': it['label'], 'age': 0.0})              # in the live view right now
    unique = []
    for it in items:                             # the same thing seen twice counts once (the fresher time wins)
        match = next((u for u in unique if u['say'] == it['say']
                      and np.linalg.norm(u['position'] - it['position']) < 0.5), None)
        if match is None:
            unique.append(it)
        else:
            match['age'] = _fresher(match.get('age'), it.get('age'))
    return unique


def describe(query, conn=None):
    """Answer a text query. Returns {'found': bool, 'text': printed answer, 'spoken': short version}."""
    own = conn is None
    conn = conn or connect()
    try:
        answer = _describe(query, conn)
    finally:
        if own:
            conn.close()
    full = answer.get('spoken', '')
    answer['spoken_full'] = full
    if LIMITED_SPEECH:                               # one short line, then silence
        answer['spoken'] = answer.get('short') or one_line(full)
    return answer


def _describe(query, conn):
    if AHEAD.search(query):                          # "what is ahead", "who is in front of me"
        return _ahead(conn)
    answer = people.handle(query, conn)              # remember / who is / forget / list people
    if answer is not None:
        return answer

    person = people.find_in_text(conn, query)
    words = set(re.findall(r"[a-z]+", query.lower()))
    if person and not (words & {'who'}):
        return _to_person(conn, person)
    return _to_object(conn, query)


def position_age(pos):
    """Seconds since the position was written, or None if it has no usable time stamp."""
    try:
        return (datetime.datetime.now() - datetime.datetime.fromisoformat(pos['timestamp'])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def _position_or_message(conn):
    pos = read_position()
    age = position_age(pos) if pos is not None else None
    if pos is None or age is None or age > POSITION_MAX_AGE_SECONDS:
        # No position, or an old one left behind by a camera program that has stopped: never answer from it.
        msg = 'The camera is not running, so I cannot say where you are. Start live_run.py first.'
        return None, {'found': False, 'text': msg, 'spoken': msg, 'short': 'Camera is not running.'}
    return pos, None


def _not_tracking(pos):
    """The tracker has lost its place: its position is stale, so give no directions and list nothing."""
    status = pos.get('status')
    text = (f"Tracking is not working right now (status: {status}), so I cannot give your position or say what "
            'is ahead. Hold still and look back at a place you scanned.')
    short = 'Tracking lost.' if status == 'lost' else 'Tracking is not ready.'
    return {'found': False, 'text': text, 'spoken': short, 'short': short, 'tracking': False}


def _spoken_distance(dist_units, scale, measured):
    return ('' if measured else 'about ') + _dist_text(dist_units, scale)


def _scale_note(measured):
    return '' if measured else ' (estimated scale: set_scale.py makes it accurate)'


def _ahead(conn):
    pos, error = _position_or_message(conn)
    if error:
        return error
    if pos.get('status') != 'tracking':
        return _not_tracking(pos)
    session = pos['session_id']
    scale, measured = effective_scale(conn, session)
    R = np.array(pos['rotation'], float)
    C = np.array([pos['x'], pos['y'], pos['z']], float)
    found = obstacles_around(scene_items(conn, session), C, R, scale, only_obstacles=False)
    note = ' ' + lookaround.coverage_note(session)

    def old_word(d):                                     # spoken only when SPEAK_AGE is on
        return ', old' if SPEAK_AGE and d['age'] is not None and not is_fresh(d['age']) else ''

    lines, spoken = [], []
    for d in found[:4]:
        where = side_text(d['angle']) or 'straight ahead'
        tag = age_tag(d['age'])
        lines.append(f"{d['name']}, about {_dist_text(d['dist'], scale)}, {where}" + (f', {tag}' if tag else ''))
        spoken.append(f"{d['name']}, {_dist_text(d['dist'], scale)}, {where}" + old_word(d))
    if found:
        text = 'Within about 1 m around you: ' + '; '.join(lines) + '.' + _scale_note(measured) + note
        say = 'Around you: ' + '; '.join(spoken) + '.'
        short = f"{_cap(found[0]['name'])}, {_spoken_distance(found[0]['dist'], scale, measured)}{old_word(found[0])}."
    else:
        text = ('Nothing known within about 1 m in any direction. I only know what the camera has already seen, '
                'and it only looks forward.' + _scale_note(measured) + note)
        say = 'Nothing known within one meter.'
        short = 'Nothing known within one meter.'
    return {'found': True, 'text': text, 'spoken': say, 'short': short}


def _to_person(conn, person):
    pos, error = _position_or_message(conn)
    if error:
        return error
    pid, name, who = person
    seen = people.last_seen(conn, pid, pos['session_id'])
    if not seen:
        when = people.last_seen_any(conn, pid)
        msg = (f'I have not seen {name} in this map.'
               + (f' Last seen {people.minutes_ago_text(when)}, in an earlier session, so I cannot give directions.'
                  if when else ''))
        return {'found': False, 'text': msg, 'spoken': msg, 'short': f'Have not seen {name} here.'}
    x, y, z, ts = seen
    stale = ''
    if people.minutes_ago_text(ts) != 'just now':
        stale = f' Last seen {people.minutes_ago_text(ts)}; people move, so they may not be there.'
    return _directions(conn, pos, name + (f', {people.speak_who(who)}' if who else ''), np.array([x, y, z], float),
                       ts, None, extra=stale)


def _to_object(conn, query):
    rows = conn.execute('SELECT id, label, x, y, z, session_id, timestamp, pos_source '
                        'FROM detections ORDER BY timestamp DESC, id DESC').fetchall()
    hits = [r for r in rows if label_matches(query, r[1])]
    if not hits:
        names = sorted({r[1] for r in rows if r[5]})
        msg = f"I have not seen anything matching '{query}' yet."
        if names:
            msg += ' I know: ' + ', '.join(names) + '.'
        return {'found': False, 'text': msg, 'spoken': msg, 'short': 'Not seen yet.'}

    pos, error = _position_or_message(conn)
    if error:
        return error
    session = pos['session_id']
    current = [r for r in hits if r[5] == session]
    if not current:
        old = hits[0]
        msg = (f"I know '{old[1]}' (last seen {old[6]}), but only from an earlier session.\n"
               'Each run has its own origin, so I cannot give a distance. Run live_run.py and look '
               'around to see it again.')
        spoken = f'I know the {old[1]}, but only from an earlier session. Look around to see it again.'
        return {'found': False, 'text': msg, 'spoken': spoken, 'short': 'Not seen in this map. Look around.'}
    id_, label, x, y, z, _, ts, source = current[0]
    return _directions(conn, pos, label, np.array([x, y, z], float), ts, id_)


def _directions(conn, pos, label, X, ts, target_id, extra=''):
    if pos.get('status') != 'tracking':
        return _not_tracking(pos)
    session = pos['session_id']
    scale, measured = effective_scale(conn, session)
    horizontal, vertical, dist, C = _heading_and_angles(pos, X)

    if abs(horizontal) > BEHIND_DEG:
        direction = 'Turn around, it is behind you'
    elif abs(horizontal) <= STRAIGHT_DEG:
        direction = 'Straight ahead'
    else:
        direction = f"Turn {'right' if horizontal > 0 else 'left'} {abs(horizontal):.0f} degrees, then go forward"
    if vertical > 15:
        direction += ' (target is above camera height)'
    elif vertical < -15:
        direction += ' (target is below camera height)'

    lines = [f'Target: {label} (last seen {ts})']
    if measured:
        lines.append(f'Distance: {dist:.2f} units (about {dist / scale:.1f} m)')
    else:
        lines.append(f'Distance: {dist:.2f} units (about {dist / scale:.1f} m, estimated scale: '
                     'set_scale.py makes it accurate)')
    lines.append(f'Direction: {direction}')

    rows = [(r[0], r[1], r[2], r[3], r[4]) for r in conn.execute(
        'SELECT id, label, x, y, z FROM detections WHERE session_id = ?', (session,))]
    R = np.array(pos['rotation'], float)
    obstacles = find_obstacles(rows, target_id, C, X, R, scale)
    spoken_obs = []
    for along, name, angle in obstacles:
        side = side_text(angle)
        lines.append(f'Obstacle: {name} on the way, about {_dist_text(along, scale)} ahead'
                     + (f', {side}' if side else ''))
        spoken_obs.append(f'Obstacle: {name}, {_dist_text(along, scale)} ahead'
                          + (f', {side}' if side else '') + '.')
    if not obstacles:
        lines.append('Obstacles: none known on the straight path')
    if extra:
        lines.append(extra.strip())
    lines.append('Note: positions are approximate (about 1 m).')

    spoken = f'{label[0].upper() + label[1:]}, {_dist_text(dist, scale)}. {direction}.'
    spoken = re.sub(r'\s*\(.*?\)', '', spoken) + ' ' + ' '.join(spoken_obs)
    short = f'{_cap(label.split(",")[0])}, {_spoken_distance(dist, scale, measured)}.'
    return {'found': True, 'text': '\n'.join(lines), 'spoken': spoken.strip(), 'short': short, 'label': label,
            'distance_units': dist, 'distance_m': dist / scale, 'turn_degrees': horizontal}
