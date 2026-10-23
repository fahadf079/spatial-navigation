"""People memory: who someone is, what they look like (face samples) and where they were last seen.

Commands (typed or spoken), handled by handle():
    remember this person as Ahmed, my brother     learn the face in front of the camera
    remember that Ahmed is my brother             save a name and who they are, without a face
    who is Ahmed                                  say who they are and when they were last seen
    who do you know                               list everyone remembered
    forget Ahmed                                  remove the person, their face samples and sightings

Face samples are biometric data. They stay in the local database only, and only for people
who agreed to be remembered. `forget` removes them.
"""
import datetime
import re

import numpy as np

SAME_PERSON = 0.40          # cosine similarity needed to say two faces are the same person
MATCH_MARGIN = 0.05         # and it must beat the second-best person by this much
ENROLL_SAMPLES = 3          # face pictures taken when learning someone
ENROLL_TIMEOUT_SECONDS = 60
SEEN_RECENT_SECONDS = 15    # a sighting this recent counts as "here right now"


# ------------------------------------------------------------------ small helpers
def clean_name(text):
    text = re.sub(r"[^A-Za-z' -]", ' ', text or '')
    text = ' '.join(text.split())[:40]
    return text.title()


def speak_who(who):
    """'my brother' is spoken to the user as 'your brother'."""
    return re.sub(r'\bmy\b', 'your', who or '', flags=re.I).strip()


def _now():
    return datetime.datetime.now()


def minutes_ago_text(timestamp):
    try:
        seconds = (_now() - datetime.datetime.fromisoformat(timestamp)).total_seconds()
    except (TypeError, ValueError):
        return 'at an unknown time'
    if seconds < 90:
        return 'just now'
    if seconds < 3600:
        return f'{seconds / 60:.0f} minutes ago'
    if seconds < 86400:
        return f'{seconds / 3600:.0f} hours ago'
    return f'{seconds / 86400:.0f} days ago'


# --------------------------------------------------------------------- database
def add_person(conn, name, who=''):
    """Add a person, or update who they are. Returns the person id."""
    name = clean_name(name)
    if not name:
        raise ValueError('A name is needed.')
    row = conn.execute('SELECT id, who FROM people WHERE name = ?', (name,)).fetchone()
    if row:
        if who:
            conn.execute('UPDATE people SET who = ? WHERE id = ?', (who, row[0]))
            conn.commit()
        return row[0]
    cur = conn.execute('INSERT INTO people (name, who, created_at) VALUES (?, ?, ?)',
                       (name, who, _now().isoformat()))
    conn.commit()
    return cur.lastrowid


def get_person(conn, person_id):
    return conn.execute('SELECT id, name, who FROM people WHERE id = ?', (person_id,)).fetchone()


def list_people(conn):
    return conn.execute('SELECT id, name, who FROM people ORDER BY name').fetchall()


def find_in_text(conn, text):
    """The remembered person named in a sentence (whole-word, any case), or None."""
    words = set(re.findall(r"[a-z']+", (text or '').lower()))
    best = None
    for id_, name, who in list_people(conn):
        parts = name.lower().split()
        if parts and all(p in words for p in parts):
            return (id_, name, who)
        if parts and parts[0] in words and best is None:
            best = (id_, name, who)
    return best


def forget_person(conn, person_id):
    for table in ('face_samples', 'people_seen', 'enroll_requests'):
        conn.execute(f'DELETE FROM {table} WHERE person_id = ?', (person_id,))
    conn.execute('DELETE FROM people WHERE id = ?', (person_id,))
    conn.commit()


def note_seen(conn, person_id, session_id, pos):
    """Remember where a person was last seen in a session (map coordinates)."""
    conn.execute(
        'INSERT INTO people_seen (person_id, session_id, x, y, z, timestamp) VALUES (?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (person_id, session_id) DO UPDATE SET x = excluded.x, y = excluded.y, '
        'z = excluded.z, timestamp = excluded.timestamp',
        (person_id, session_id, float(pos[0]), float(pos[1]), float(pos[2]), _now().isoformat()))
    conn.commit()


def last_seen(conn, person_id, session_id):
    """(x, y, z, timestamp) of the person in this session, or None."""
    return conn.execute('SELECT x, y, z, timestamp FROM people_seen WHERE person_id = ? AND session_id = ?',
                        (person_id, session_id)).fetchone()


def last_seen_any(conn, person_id):
    row = conn.execute('SELECT timestamp FROM people_seen WHERE person_id = ? ORDER BY timestamp DESC LIMIT 1',
                       (person_id,)).fetchone()
    return row[0] if row else None


def recent_seen(conn, session_id, seconds=SEEN_RECENT_SECONDS):
    """People seen in this session in the last few seconds: [(name, who, x, y, z)]."""
    out = []
    for name, who, x, y, z, ts in conn.execute(
            'SELECT p.name, p.who, s.x, s.y, s.z, s.timestamp FROM people_seen s '
            'JOIN people p ON p.id = s.person_id WHERE s.session_id = ?', (session_id,)):
        try:
            age = (_now() - datetime.datetime.fromisoformat(ts)).total_seconds()
        except (TypeError, ValueError):
            continue
        if age <= seconds:
            out.append((name, who, x, y, z))
    return out


# ---------------------------------------------------------------- face samples
def _unit(vec):
    vec = np.asarray(vec, np.float32).ravel()
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else vec


def save_sample(conn, person_id, embedding):
    conn.execute('INSERT INTO face_samples (person_id, embedding, created_at) VALUES (?, ?, ?)',
                 (person_id, _unit(embedding).tobytes(), _now().isoformat()))
    conn.commit()


def load_known(conn):
    """[(person_id, name, who, unit embedding)] for every stored face sample."""
    out = []
    for pid, name, who, blob in conn.execute(
            'SELECT p.id, p.name, p.who, f.embedding FROM face_samples f JOIN people p ON p.id = f.person_id'):
        out.append((pid, name, who, np.frombuffer(blob, np.float32)))
    return out


def best_match(embedding, known):
    """The remembered person whose face is closest, or None if nobody is close enough.

    The best person must reach SAME_PERSON and beat every other person by MATCH_MARGIN, so two
    similar-looking people are not confused.
    """
    if not known:
        return None
    emb = _unit(embedding)
    scores = {}
    for pid, name, who, vec in known:
        s = float(emb @ vec)
        if s > scores.get(pid, (-1.0,))[0]:
            scores[pid] = (s, name, who)
    ranked = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)
    top_pid, (top, name, who) = ranked[0]
    if top < SAME_PERSON:
        return None
    if len(ranked) > 1 and top - ranked[1][1][0] < MATCH_MARGIN:
        return None
    return top_pid, name, who, top


# --------------------------------------------------------------- enrollment queue
def request_enrollment(conn, person_id, samples=ENROLL_SAMPLES):
    """Ask the running camera to learn the next face it sees as this person."""
    conn.execute("UPDATE enroll_requests SET status = 'replaced' WHERE status = 'pending'")
    cur = conn.execute('INSERT INTO enroll_requests (person_id, samples_needed, created_at, status) '
                       "VALUES (?, ?, ?, 'pending')", (person_id, samples, _now().isoformat()))
    conn.commit()
    return cur.lastrowid


def pending_enrollment(conn):
    """(request id, person id, samples still needed, expired?) for the newest pending request."""
    row = conn.execute("SELECT id, person_id, samples_needed, created_at FROM enroll_requests "
                       "WHERE status = 'pending' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    try:
        age = (_now() - datetime.datetime.fromisoformat(row[3])).total_seconds()
    except (TypeError, ValueError):
        age = 0
    return row[0], row[1], row[2], age > ENROLL_TIMEOUT_SECONDS


def add_enroll_sample(conn, request_id, person_id, embedding):
    """Store one face sample. Returns how many more are needed (0 means done)."""
    save_sample(conn, person_id, embedding)
    conn.execute('UPDATE enroll_requests SET samples_needed = samples_needed - 1 WHERE id = ?', (request_id,))
    left = conn.execute('SELECT samples_needed FROM enroll_requests WHERE id = ?', (request_id,)).fetchone()[0]
    if left <= 0:
        conn.execute("UPDATE enroll_requests SET status = 'done' WHERE id = ?", (request_id,))
    conn.commit()
    return max(left, 0)


def finish_request(conn, request_id, status):
    conn.execute('UPDATE enroll_requests SET status = ? WHERE id = ?', (status, request_id))
    conn.commit()


# ------------------------------------------------------------- spoken commands
ENROLL = re.compile(r"^(?:please\s+)?(?:remember|learn)\s+this\s+(?:person|face|man|woman|guy|lady)\s+as\s+(.+)$"
                    r"|^(?:please\s+)?this\s+person\s+is\s+(.+)$", re.I)
TEXT_MEMORY = re.compile(r"^(?:please\s+)?remember\s+(?:that\s+)?(.+?)\s+is\s+(.+)$", re.I)
WHO_IS = re.compile(r"^(?:who\s+is|who's|who\s+was|tell\s+me\s+about)\s+(.+)$", re.I)
LIST_PEOPLE = re.compile(r"^(?:who\s+do\s+you\s+know|who\s+do\s+i\s+know|who\s+are\s+my\s+people|"
                         r"list\s+(?:all\s+)?(?:the\s+|my\s+)?(?:people|contacts)|show\s+(?:me\s+)?(?:my\s+)?people)$", re.I)
FORGET = re.compile(r"^(?:please\s+)?(?:forget|delete|remove)\s+(?:about\s+)?(.+)$", re.I)
SPLIT_NAME_WHO = re.compile(r"\s*,\s*|\s+(?=(?:my|our|your|the|a|an)\s)|\s+who\s+is\s+|"
                            r"\s+(?:he|she|they)\s+(?:is|are)\s+|\s+is\s+", re.I)


def split_name_who(text):
    parts = SPLIT_NAME_WHO.split(text.strip(' .,!?'), maxsplit=1)
    name = clean_name(parts[0])
    who = parts[1].strip(' .,!?') if len(parts) > 1 else ''
    return name, who


def _reply(text, spoken=None, found=True):
    return {'found': found, 'text': text, 'spoken': spoken or text}


def handle(query, conn):
    """Answer a people command, or return None if the sentence is not about people."""
    q = ' '.join((query or '').split()).strip(' .!?')
    if not q:
        return None

    if LIST_PEOPLE.match(q):
        people = list_people(conn)
        if not people:
            return _reply('I do not know anyone yet. Say: remember this person as NAME, my friend.')
        lines = [f'{name}' + (f', {speak_who(who)}' if who else '') for _, name, who in people]
        return _reply('People I know: ' + '; '.join(lines) + '.')

    m = ENROLL.match(q)
    if m:
        name, who = split_name_who(m.group(1) or m.group(2))
        if not name:
            return _reply('I did not catch the name. Say: remember this person as NAME, my friend.', found=False)
        pid = add_person(conn, name, who)
        request_enrollment(conn, pid)
        text = (f'OK. Point the camera at {name}'
                + (f' ({speak_who(who)})' if who else '')
                + ' for a few seconds. I will tell you when I have learned the face. '
                  'Only do this with their agreement.')
        return _reply(text, f'OK. Point the camera at {name} for a few seconds.')

    m = TEXT_MEMORY.match(q)
    if m and not re.match(r'^(?:this|that|it)$', m.group(1).strip(), re.I):
        name, who = clean_name(m.group(1)), m.group(2).strip(' .,!?')
        if name and who:
            add_person(conn, name, who)
            text = (f'OK, I will remember that {name} is {speak_who(who)}. To teach me the face, say: '
                    f'remember this person as {name}.')
            return _reply(text, f'OK, I will remember that {name} is {speak_who(who)}.')

    m = FORGET.match(q)
    if m:
        person = find_in_text(conn, m.group(1))
        if person:
            forget_person(conn, person[0])
            return _reply(f'OK. I forgot {person[1]}, including their face samples.')
        return None

    m = WHO_IS.match(q)
    if m:
        person = find_in_text(conn, m.group(1))
        if not person:
            asked = clean_name(m.group(1))
            return _reply(f'I do not know anyone called {asked}. Say: remember that {asked} is my friend.'
                          if asked else 'I did not catch the name.', found=False)
        _, name, who = person
        seen = last_seen_any(conn, person[0])
        text = f'{name} is {speak_who(who)}.' if who else f'I know {name}, but not who they are yet.'
        text += f' Last seen {minutes_ago_text(seen)}.' if seen else ' I have not seen them yet.'
        return _reply(text)
    return None
