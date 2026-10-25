"""Live backend: odometry, 3D mapping and object detection from one camera.

    python live_run.py                 odometry + map + detection
    python live_run.py --no-detect     odometry + map only
    python live_run.py --no-map        odometry + detection only
    python live_run.py --video clip.mp4   use a video file instead of the webcam
    python live_run.py --conf 0.5      lower the detection confidence (default 0.6)
    python live_run.py --dshow         open the webcam with DirectShow (try this if the picture
                                       turns green or freezes on Windows)
    python live_run.py --ignore person,chair   labels never saved (default: person, because
                                               people move and are not landmarks)
    python live_run.py --ignore none           save every label
    python live_run.py --model yolov8s.pt      use a bigger, more accurate (slower) YOLO model
    python live_run.py --new                   do not look for the previous map; start a fresh one
    python live_run.py --map 17                continue the saved map of session 17
    python live_run.py --speak                 say warnings out loud (tracking lost, obstacle within 1 m...)
    python live_run.py --no-faces              never try to recognise faces

By default a run first looks for the most recently saved map for a few seconds. If you aim the
camera at a place you scanned before, it continues in that map's coordinates (same session, same
scale) instead of starting a new session.

Everything is saved to spatial_memory.db as it happens. Press q to quit.
"""
import argparse
import inspect
import datetime
import time

import cv2
import numpy as np

import faces
import people
from guidance import obstacles_ahead, warning_line
from sound import Speaker
from tracker import (CALIB_FILE, LOST_FRAMES_BEFORE_RESEED, VisualOdometry, connect,
                     effective_scale, get_units_per_meter, latest_map_session, load_keyframes,
                     new_session, save_keyframe, write_live_scene, write_position)

DETECT_EVERY = 10      # run YOLO on every Nth frame
MIN_CONFIDENCE = 0.6   # ignore weaker detections (0.4 gave many wrong objects)
DEDUP_UNITS = 1.0      # same label closer than this (in map units) counts as the same object
MERGE_METERS = 0.75    # once the scale is known: same label within this many meters is the same object
MAX_SIGHTINGS_KEPT = 15  # the stored position is the median of the latest sightings
DEDUP_DEPTH_FRACTION = 0.25  # far objects have noisier positions, so allow 25% of their distance
MIN_MAP_POINTS_IN_BOX = 3
BOX_GROW = 0.2         # if the box has too few map points, retry with it 20% larger each side
NEAR_FRACTION = 0.3    # use only the nearest 30% of the points in a box: the object, not the wall behind it
ORB_FEATURES = 2500
IGNORE_LABELS = ('person',)   # moving things are not landmarks
RESUME_SEARCH_FRAMES = 120    # how long a run looks for the previous map before starting a new one
BLANK_STD = 3.0              # a picture flatter than this (all green, all black) is not a real view
BAD_FRAMES_BEFORE_REOPEN = 30  # about 1 second of blank or frozen frames: reopen the webcam


def make_detector(conf=MIN_CONFIDENCE, model_name='yolov8n.pt'):
    from ultralytics import YOLO
    model = YOLO(model_name)

    def detect(frame):
        result = model(frame, verbose=False, conf=conf)[0]
        out = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append((model.names[int(box.cls[0])], float(box.conf[0]), (x1, y1, x2, y2)))
        return out

    return detect


def locate_object(vo, bbox):
    """World position of an object: median of the NEAREST tracked map points inside its box.
    A box also contains the wall or table behind the object, and phones and remotes have few
    features of their own, so the median of all points lands on the background.
    Returns (None, 'none') when too few map points are inside, so no guess is saved."""
    x1, y1, x2, y2 = bbox
    px, xyz = vo.tracked_px, vo.tracked_xyz
    if len(px):
        w, h = x2 - x1, y2 - y1
        for grow in (0.0, BOX_GROW):
            gx, gy = grow * w, grow * h
            inside = ((px[:, 0] >= x1 - gx) & (px[:, 0] <= x2 + gx) &
                      (px[:, 1] >= y1 - gy) & (px[:, 1] <= y2 + gy))
            if inside.sum() >= MIN_MAP_POINTS_IN_BOX:
                pts = xyz[inside]
                depth = (vo.R @ pts.T + vo.t).T[:, 2]        # distance in front of the camera
                front = depth > 0.05
                if front.sum() < MIN_MAP_POINTS_IN_BOX:
                    continue
                pts, depth = pts[front], depth[front]
                keep = max(MIN_MAP_POINTS_IN_BOX, int(round(NEAR_FRACTION * len(pts))))
                return np.median(pts[np.argsort(depth)[:keep]], axis=0), 'map'
    return None, 'none'


def merge_gate(conn, session_id, distance):
    """How close (in map units) a new sighting must be to count as an earlier object."""
    row = conn.execute('SELECT units_per_meter FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if row and row[0]:
        return MERGE_METERS * float(row[0])
    return max(DEDUP_UNITS, DEDUP_DEPTH_FRACTION * distance)


def record_sighting(conn, session_id, objects, label, conf, pos, distance, source_kind):
    """Keep ONE database row per object. A repeat sighting updates that row with the median
    position of the sightings so far (single sightings are noisy). Returns 'new' or 'merged'."""
    gate = merge_gate(conn, session_id, distance)
    now = datetime.datetime.now().isoformat()
    best, best_d = None, None
    for obj in objects:
        if obj['label'] != label:
            continue
        d = float(np.linalg.norm(obj['position'] - pos))
        if d < gate and (best_d is None or d < best_d):
            best, best_d = obj, d
    if best is not None:
        best['sightings'] = (best['sightings'] + [pos])[-MAX_SIGHTINGS_KEPT:]
        best['position'] = np.median(np.array(best['sightings']), axis=0)
        best['conf'] = max(best['conf'], conf)
        p = best['position']
        conn.execute('UPDATE detections SET confidence = ?, timestamp = ?, x = ?, y = ?, z = ? WHERE id = ?',
                     (best['conf'], now, float(p[0]), float(p[1]), float(p[2]), best['row_id']))
        conn.commit()
        return 'merged'
    cur = conn.execute(
        'INSERT INTO detections (label, confidence, image_source, timestamp, x, y, z, '
        'session_id, pos_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (label, conf, 'live_camera', now, float(pos[0]), float(pos[1]), float(pos[2]),
         session_id, source_kind))
    conn.commit()
    objects.append({'label': label, 'row_id': cur.lastrowid, 'position': np.array(pos, float),
                    'sightings': [np.array(pos, float)], 'conf': conf})
    return 'new'


def load_objects(conn, session_id):
    """Objects already stored for a session, so a resumed run merges into them."""
    out = []
    for id_, label, conf, x, y, z in conn.execute(
            'SELECT id, label, confidence, x, y, z FROM detections WHERE session_id = ?', (session_id,)):
        p = np.array([x, y, z], float)
        out.append({'label': label, 'row_id': id_, 'position': p, 'sightings': [p], 'conf': conf})
    return out


def open_capture(source, dshow=False):
    if hasattr(source, 'read'):          # a ready-made frame source (the phone pipeline)
        return source
    if isinstance(source, int) and dshow and hasattr(cv2, 'CAP_DSHOW'):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def draw_overlay(frame, vo, status, points_saved, detections_saved, boxes, bad_frames, note=None):
    """Position, status, detection boxes and a clear warning when tracking is not working."""
    GREEN, RED, YELLOW, ORANGE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 165, 255)
    x, y, z = vo.position
    status_colour = GREEN if status == 'tracking' else RED if status == 'lost' else YELLOW
    cv2.putText(frame, f'X:{x:.1f} Y:{y:.1f} Z:{z:.1f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
    cv2.putText(frame, f'{status} | points {points_saved} | objects {detections_saved}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_colour, 1)
    for label, (x1, y1, x2, y2), saved in boxes:
        colour = GREEN if saved else ORANGE
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1)
        cv2.putText(frame, label if saved else f'{label} (not saved)', (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    message = None
    if note:
        message = note
    elif bad_frames >= 10:
        message = 'CAMERA PICTURE BLANK OR FROZEN - reopening'
    elif status == 'lost':
        message = f'LOST {vo.lost}/{LOST_FRAMES_BEFORE_RESEED} - point back at the scanned area'
    if message:
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 34), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, message, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)


def _takes_arg(fn, name):
    """True if the notify function accepts a name=... argument (older ones take only the text)."""
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _takes_speak(fn):
    return _takes_arg(fn, 'speak')


def run(save_map=True, detect=True, source=0, show=True, max_frames=None, detector=None,
        conf=MIN_CONFIDENCE, dshow=False, ignore=IGNORE_LABELS, model='yolov8n.pt', resume='auto',
        K=None, state=None, notify=None, face_engine=None, use_faces=True):
    """state: optional dict updated every frame. notify: optional function(text) that receives
    short spoken-style messages (tracking lost, obstacle within 1 m...)."""
    if K is None:
        K = np.load(CALIB_FILE)['camera_matrix']
    conn = connect()
    session_id = new_session(conn)

    # Is there a saved map to continue? Only look if the user did not ask for a fresh start.
    resume_session, saved_kfs = None, []
    if resume != 'new':
        candidate = resume if isinstance(resume, int) else latest_map_session(conn)
        if candidate is not None and candidate != session_id:
            saved_kfs = load_keyframes(conn, candidate)
            resume_session = candidate if saved_kfs else None
        if isinstance(resume, int) and not saved_kfs:
            print(f'Session {resume} has no saved map. Starting a new one.')
    searching = bool(saved_kfs)
    search_left = RESUME_SEARCH_FRAMES
    buffered_points, buffered_kfs = [], []     # provisional data, kept only if no saved map is found
    if searching:
        print(f'Looking for the saved map of session {resume_session} ({len(saved_kfs)} views). '
              'Aim the camera at a place you scanned before.')
    else:
        print(f'Session {session_id} started. Coordinates are only comparable inside this session.')

    if detect and detector is None:
        detector = make_detector(conf, model)

    cap = open_capture(source, dshow)
    webcam = isinstance(source, int)
    prev_gray = None
    bad_frames = 0
    reopened = 0
    orb = cv2.ORB_create(ORB_FEATURES)
    vo = VisualOdometry(K)
    objects = []                   # one entry per object stored this session
    merged = 0
    points_saved = detections_saved = skipped = frame_idx = 0
    sessions_used = [session_id]
    generation = 0
    last_status = None
    boxes = []
    last_event = {}
    scale, scale_frame = None, -1000
    engine, engine_failed = face_engine, not use_faces
    known_faces, pending, people_frame = [], None, -1000
    last_sample = 0.0
    live_objs, live_time = [], 0.0       # what the last detection pass saw (people included)

    notify_takes_speak = _takes_speak(notify)
    notify_takes_alert = _takes_arg(notify, 'alert')

    def event(text, key=None, gap=8.0, speak=True, alert=False):
        """Send a short message. speak=False marks it as information to show but not say out loud
        (limited speech: only warnings and answers are spoken). alert=True marks an obstacle warning,
        so the phone can beep or vibrate (the spoken line itself no longer says "ahead")."""
        if notify is None:
            return
        key = key or text
        now = time.time()
        if now - last_event.get(key, 0) < gap:
            return
        last_event[key] = now
        if alert and notify_takes_alert:
            notify(text, alert=True)
        elif not speak and notify_takes_speak:
            notify(text, speak=False)
        else:
            notify(text)

    def get_engine():
        nonlocal engine, engine_failed
        if engine is None and not engine_failed:
            try:
                engine = faces.FaceEngine()
            except Exception as exc:                  # no models, no internet, old OpenCV...
                engine_failed = True
                print(f'Face recognition is not available: {exc}')
                event('Face recognition is not available. People can be remembered by name only.',
                      'nofaces', 600, speak=False)
        return engine

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_idx >= max_frames):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if webcam:
            # A green or black picture, or one that never changes, means the camera driver
            # glitched. Nothing can be tracked from it, so say so and reopen the camera.
            glitch = float(gray.std()) < BLANK_STD or (prev_gray is not None and np.array_equal(gray, prev_gray))
            bad_frames = bad_frames + 1 if glitch else 0
            prev_gray = gray
            if bad_frames >= BAD_FRAMES_BEFORE_REOPEN:
                reopened += 1
                print(f'Camera picture is blank or frozen. Reopening the camera (attempt {reopened}).')
                if reopened == 1:
                    print('If this keeps happening: close other apps using the camera, unplug and '
                          'replug it, or start with: python live_run.py --dshow')
                cap.release()
                cap = open_capture(source, dshow)
                bad_frames = 0
        kp, des = orb.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.empty((0, 2), np.float32)
        status = vo.update(pts, des)
        relocated_now = False
        if searching:
            if vo.generation != generation:     # the provisional tracker restarted: its data is stale
                generation = vo.generation
                buffered_points.clear()
                buffered_kfs.clear()
            elif frame_idx % 2 == 0:
                hit = vo.relocalize(pts, des, saved_kfs)
                if hit is not None:
                    vo.load_map(saved_kfs, hit[1], hit[0])
                    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
                    conn.commit()
                    session_id = resume_session
                    sessions_used = [session_id]
                    objects = load_objects(conn, session_id)
                    buffered_points.clear()
                    buffered_kfs.clear()
                    generation = vo.generation
                    status = vo.status
                    searching = False
                    relocated_now = True
                    saved_scale = get_units_per_meter(conn, session_id)
                    scale, scale_measured = effective_scale(conn, session_id)
                    scale_frame = frame_idx
                    print(f'Found the saved map of session {session_id}. Continuing in its coordinates'
                          + (f' ({saved_scale:.2f} map units per meter).' if saved_scale else '.'))
                    event('Saved map found. Continuing where you left off.', speak=False)
            search_left -= 1
            if searching and search_left <= 0:
                searching = False
                if save_map:
                    now = datetime.datetime.now().isoformat()
                    for pts_new in buffered_points:
                        conn.executemany(
                            'INSERT INTO point_cloud (x, y, z, timestamp, session_id) VALUES (?, ?, ?, ?, ?)',
                            [(float(p[0]), float(p[1]), float(p[2]), now, session_id) for p in pts_new])
                        points_saved += len(pts_new)
                    for kf in buffered_kfs:
                        save_keyframe(conn, session_id, kf)
                    conn.commit()
                buffered_points.clear()
                buffered_kfs.clear()
                print(f'The saved map was not recognised here. Starting a new map: session {session_id}. '
                      'Coordinates are only comparable inside this session.')
                event('Saved map not recognised here. Starting a new map.', speak=False)
        elif vo.generation != generation:
            # Tracking was lost for too long and no saved view matched, so the tracker started
            # a new world frame. Anything saved from now on must not be mixed with the old map.
            generation = vo.generation
            session_id = new_session(conn)
            sessions_used.append(session_id)
            objects = []
            print(f'Tracking was lost for too long. New world frame: session {session_id} started.')
            event('Tracking lost. New map started.')
        write_position(session_id, vo.R, vo.t, status)
        if status != last_status:
            print(f'Tracking status: {status}' + (f' ({vo.reason})' if status == 'lost' else ''))
            if status == 'lost':
                event('Tracking lost.', 'lost', 10)
            elif status == 'tracking' and last_status == 'lost':
                event('Tracking restored.', 'restored', 5, speak=False)
            last_status = status

        # Refresh what we know about people and the scale every second or so.
        if frame_idx - people_frame >= 30:
            people_frame = frame_idx
            known_faces = people.load_known(conn)
            pending = people.pending_enrollment(conn)
            if pending and pending[3]:                # nobody showed a face in time
                person = people.get_person(conn, pending[1])
                people.finish_request(conn, pending[0], 'expired')
                event(f'I could not see a face to learn {person[1] if person else "that person"}. '
                      'Point the camera at their face and say it again.', 'enroll-expired', 5)
                pending = None
        if frame_idx - scale_frame >= 60:
            scale, scale_measured = effective_scale(conn, session_id)
            scale_frame = frame_idx

        # Warn about anything within about 1 m in ANY direction (360 degrees): stored objects and people seen just now.
        if notify is not None and status == 'tracking' and not searching and frame_idx % 15 == 0:
            items = [{'label': o['label'], 'position': o['position'], 'say': o['label']} for o in objects]
            if time.time() - live_time < 3.0:
                items += live_objs
            alerts = obstacles_ahead(items, vo.position, vo.R, scale)
            if alerts:
                name, dist, angle, kind = alerts[0]
                # limited speech: the object and its distance in one short line, no direction
                event(warning_line(name, dist / scale), f'obstacle-{name}', 6, alert=True)

        new_kfs = vo.drain_new_keyframes()
        new_points = np.empty((0, 3)) if relocated_now else vo.new_points
        if searching:
            if save_map:
                if len(new_points):
                    buffered_points.append(new_points.copy())
                buffered_kfs.extend(new_kfs)
        elif save_map:
            if len(new_points):
                now = datetime.datetime.now().isoformat()
                conn.executemany(
                    'INSERT INTO point_cloud (x, y, z, timestamp, session_id) VALUES (?, ?, ?, ?, ?)',
                    [(float(p[0]), float(p[1]), float(p[2]), now, session_id) for p in new_points])
                conn.commit()
                points_saved += len(new_points)
            for kf in new_kfs:
                save_keyframe(conn, session_id, kf)

        if detect and frame_idx % DETECT_EVERY == 0:
            boxes = []
            live_items = []
            live_objs = []
            can_save = status == 'tracking' and not searching  # no pose, or still looking for the map
            detections_now = detector(frame)

            # Faces: needed to recognise remembered people, or to learn a new one.
            face_for = {}
            person_ids = [i for i, d in enumerate(detections_now) if d[0] == 'person']
            if person_ids and (known_faces or pending) and get_engine() is not None:
                try:
                    found = engine.detect(frame)
                    pairs = faces.faces_in_boxes(found, [detections_now[i][2] for i in person_ids])
                    face_for = dict(zip(person_ids, pairs))
                except Exception as exc:
                    print(f'Face detection failed: {exc}')

            for i, (label, conf, bbox) in enumerate(detections_now):
                pos, source_kind = locate_object(vo, bbox) if can_save else (None, 'none')
                name = who = pid = None
                face = face_for.get(i)
                if label == 'person' and face is not None:
                    try:
                        emb = engine.embed(frame, face)
                        if pending and not pending[3]:
                            if time.time() - last_sample >= 0.7:      # a few different pictures
                                last_sample = time.time()
                                person = people.get_person(conn, pending[1])
                                left = people.add_enroll_sample(conn, pending[0], pending[1], emb)
                                if left == 0:
                                    known_faces = people.load_known(conn)
                                    event(f'I learned {person[1]}.', 'enrolled', 2)
                                    print(f'Learned the face of {person[1]}.')
                                    pending = None
                                else:
                                    pending = (pending[0], pending[1], left, False)
                        else:
                            match = people.best_match(emb, known_faces)
                            if match:
                                pid, name, who, score = match
                    except Exception as exc:
                        print(f'Face recognition failed: {exc}')
                if label == 'person' and pending and engine_failed and not pending[3]:
                    # no face models: note who it is and where, but they cannot be recognised later
                    person = people.get_person(conn, pending[1])
                    people.finish_request(conn, pending[0], 'no-faces')
                    if pos is not None and person:
                        people.note_seen(conn, pending[1], session_id, pos)
                    event(f'I noted {person[1] if person else "them"}, but face recognition is not '
                          'installed, so I cannot recognise them later.', 'enrolled', 2)
                    pending = None
                if name and pos is not None:
                    people.note_seen(conn, pid, session_id, pos)
                    dist_m = float(np.linalg.norm(pos - vo.position)) / scale
                    if dist_m <= 3.0:
                        event(f'{name}' + (f', {people.speak_who(who)}' if who else '')
                              + f', {dist_m:.1f} meters', f'seen-{pid}', 30)
                if pos is not None:
                    item = {'label': label, 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}
                    if name:
                        item['name'] = name
                    live_items.append(item)
                    say = (name + (f', {people.speak_who(who)}' if who else '')) if name else label
                    live_objs.append({'label': label, 'position': pos, 'say': say})
                if name:
                    boxes.append((name, bbox, can_save))
                    continue                       # a remembered person is not stored as an object
                if label in ignore:
                    continue
                boxes.append((label, bbox, can_save))
                if not can_save:
                    continue
                if pos is None:
                    skipped += 1       # no map points on the object, so its position is unknown
                    continue
                distance = float(np.linalg.norm(pos - vo.position))
                if record_sighting(conn, session_id, objects, label, conf, pos, distance, source_kind) == 'merged':
                    merged += 1
                    continue
                detections_saved += 1
                print(f'Saved {label} ({conf:.2f}) at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) [{source_kind}]')
                event(f'{label} saved', f'saved-{label}', 15, speak=False)
            live_time = time.time()
            write_live_scene(session_id, live_items)

        if state is not None:
            c = vo.position
            state.update(status=status, x=float(c[0]), y=float(c[1]), z=float(c[2]),
                         session_id=session_id, points=points_saved, objects=len(objects),
                         searching=searching, lost_frames=vo.lost)
        if show:
            draw_overlay(frame, vo, status, points_saved, detections_saved, boxes, bad_frames,
                         note=f'LOOKING FOR SAVED MAP {search_left}' if searching else None)
            cv2.imshow('Spatial Navigation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if searching and save_map:                # the video ended before the search finished
        for pts_new in buffered_points:
            conn.executemany(
                'INSERT INTO point_cloud (x, y, z, timestamp, session_id) VALUES (?, ?, ?, ?, ?)',
                [(float(p[0]), float(p[1]), float(p[2]), datetime.datetime.now().isoformat(), session_id)
                 for p in pts_new])
        for kf in buffered_kfs:
            save_keyframe(conn, session_id, kf)
        conn.commit()
    conn.close()
    names = ', '.join(str(i) for i in sessions_used)
    print(f'Done. Session(s) {names}: {points_saved} map points, {detections_saved} detections saved, '
          f'{merged} repeat sightings merged into existing objects, '
          f'{skipped} detections skipped (no map points on the object).')
    return session_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-map', action='store_true')
    parser.add_argument('--no-detect', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--conf', type=float, default=MIN_CONFIDENCE)
    parser.add_argument('--dshow', action='store_true')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO weights, e.g. yolov8s.pt')
    parser.add_argument('--speak', action='store_true', help='say warnings out loud')
    parser.add_argument('--no-faces', action='store_true', help='never try to recognise faces')
    parser.add_argument('--new', action='store_true', help='do not look for a saved map')
    parser.add_argument('--map', type=int, default=None, help='continue the saved map of this session')
    parser.add_argument('--ignore', default=','.join(IGNORE_LABELS),
                        help='comma-separated labels never saved, or none')
    args = parser.parse_args()
    run(save_map=not args.no_map, detect=not args.no_detect,
        source=args.video if args.video else 0, conf=args.conf, dshow=args.dshow,
        model=args.model, resume='new' if args.new else (args.map if args.map else 'auto'),
        notify=(lambda text, speak=True, sp=Speaker(): sp.say(text, gap=0) if speak else None) if args.speak else None,
        use_faces=not args.no_faces,
        ignore=() if args.ignore.strip().lower() == 'none'
        else tuple(x.strip() for x in args.ignore.split(',') if x.strip()))
