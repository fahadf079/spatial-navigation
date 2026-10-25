"""Self-test for the tracker. No webcam needed.

Builds a synthetic room, moves a virtual camera through it (translation and
rotation), runs the tracker, and checks the estimated path and map against the truth.
It also checks the saved-map, recovery, object-placement, object-merging, guidance, "what is ahead" and people-memory logic.
Your real spatial_memory.db and current_position.json are never touched (temporary files).
Run:  python selftest.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

import tracker
from tracker import (VisualOdometry, camera_center, connect, load_keyframes, new_session,
                     save_keyframe, write_live_scene, write_position)

rng = np.random.default_rng(7)
K = np.array([[387.0, 0, 320.0], [0, 387.0, 236.0], [0, 0, 1.0]])
W, H = 640, 480
N_LAND = 1800
landmarks = np.column_stack([rng.uniform(-10, 10, N_LAND), rng.uniform(-3, 3, N_LAND),
                             rng.uniform(3, 10, N_LAND)])
descs = rng.integers(0, 256, (N_LAND, 32), dtype=np.uint8)


SCENARIO = 'turn'


def pose(i):
    if SCENARIO == 'turn':      # steady turn to the right
        rv = np.array([0, np.radians(0.9 * i), 0.0])
    else:                       # yaw swings left and right while pitching up (axes do not commute)
        rv = np.array([np.radians(0.35 * i), np.radians(25 * np.sin(i / 12.0)), np.radians(0.15 * i)])
        rv = cv2.Rodrigues(cv2.Rodrigues(np.array([0, rv[1], 0.0]))[0] @
                           cv2.Rodrigues(np.array([rv[0], 0, rv[2]]))[0])[0].ravel()
    R = cv2.Rodrigues(rv)[0]                       # world -> camera
    C = np.array([0.10 * i, 0.01 * i, 0.04 * i])   # camera center in world
    return R, -R @ C.reshape(3, 1)


def observe(R, t):
    Xc = (R @ landmarks.T + t).T
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    vis = (Xc[:, 2] > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    idx = np.flatnonzero(vis)
    pts = (uv[idx] + rng.normal(0, 0.5, (len(idx), 2))).astype(np.float32)
    d = descs[idx].copy()
    bad = rng.random(len(idx)) < 0.05              # 5% wrong matches
    d[bad] = rng.integers(0, 256, (bad.sum(), 32), dtype=np.uint8)
    return pts, d


def run_scenario(name):
    global SCENARIO
    SCENARIO = name
    print(f'--- scenario: {name}')
    R0, t0 = pose(0)
    # make the first camera the world frame, like the real system
    vo = VisualOdometry(K)
    est, truth, cloud = [], [], []
    for i in range(60):
        R, t = pose(i)
        pts, d = observe(R, t)
        vo.update(pts, d)
        est.append(vo.position.copy())
        truth.append(camera_center(R, t))
        if len(vo.new_points):
            cloud.append(vo.new_points)
    est, truth = np.array(est), np.array(truth)
    ok = True

    print(f"final status: {vo.status}")
    print(f"map points created: {sum(len(c) for c in cloud)}")

    # the first camera is the world origin, so only the scale is unknown
    s = float(np.sum(est * truth) / np.sum(est * est))
    aligned = s * est
    err = np.linalg.norm(aligned - truth, axis=1)
    path_len = np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1))
    rel = err.mean() / path_len * 100
    print(f"path error after scale alignment: mean {err.mean():.3f}, {rel:.1f}% of path length")
    if rel > 6 or vo.status != 'tracking':
        ok = False

    # rotation: estimated heading must follow the true turn (about 53 degrees)
    Rt_final = pose(59)[0]
    ang_err = np.degrees(np.arccos(np.clip((np.trace(vo.R @ Rt_final.T) - 1) / 2, -1, 1)))
    print(f"final heading error: {ang_err:.1f} deg")
    if ang_err > 8:
        ok = False

    # map: points, scaled into the true frame, should sit on real landmarks
    cloud = np.vstack(cloud)
    cloud_true = s * cloud
    dist = np.min(np.linalg.norm(cloud_true[:, None, :] - landmarks[None, :, :], axis=2), axis=1)
    print(f"map point error: median {np.median(dist):.2f}, 90th percentile {np.percentile(dist, 90):.2f} (scene depth 3 to 10)")
    if np.median(dist) > 0.4:
        ok = False

    return ok


def check(name, passed, detail=''):
    print(f"  [{'ok' if passed else 'FAIL'}] {name}" + (f' ({detail})' if detail else ''))
    return passed


def run_logic_checks():
    """Saved maps, recovery, object placement and merging. Uses a temporary database."""
    global SCENARIO, landmarks, descs
    print('--- logic checks')
    SCENARIO = 'turn'
    ok = True
    old_db, old_pos, old_scene = tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE
    tmp = tempfile.mkdtemp()
    tracker.DB_PATH = os.path.join(tmp, 'selftest.db')
    tracker.POSITION_FILE = os.path.join(tmp, 'position.json')
    tracker.LIVE_SCENE_FILE = os.path.join(tmp, 'live_scene.json')
    try:
        # 1. scan, save the map, load it in a new tracker that starts mid-path
        vo1, est = VisualOdometry(K), []
        conn = connect()
        sid = new_session(conn)
        for i in range(60):
            R, t = pose(i)
            pts, d = observe(R, t)
            vo1.update(pts, d)
            est.append(vo1.position.copy())
            for kf in vo1.drain_new_keyframes():
                save_keyframe(conn, sid, kf)
        saved = load_keyframes(conn, sid)
        ok &= check('keyframes saved and loaded', len(saved) >= 5, f'{len(saved)} keyframes')
        vo2, err = VisualOdometry(K), None
        for j in range(40):
            R, t = pose(30 + j)
            pts, d = observe(R, t)
            hit = vo2.relocalize(pts, d, saved)
            if hit:
                vo2.load_map(saved, hit[1], hit[0])
                err = float(np.linalg.norm(vo2.position - est[30 + j]))
                break
        ok &= check('a new run recognises the saved map', err is not None and err < 0.5,
                    'not found' if err is None else f'position error {err:.2f} units')

        # 2. a different room must never be accepted
        keep_l, keep_d = landmarks, descs
        r2 = np.random.default_rng(99)
        landmarks = np.column_stack([r2.uniform(-10, 10, N_LAND), r2.uniform(-3, 3, N_LAND),
                                     r2.uniform(3, 10, N_LAND)])
        descs = r2.integers(0, 256, (N_LAND, 32), dtype=np.uint8)
        vo3, accepted = VisualOdometry(K), 0
        for i in range(60):
            R, t = pose(i)
            pts, d = observe(R, t)
            accepted += vo3.relocalize(pts, d, saved) is not None
        landmarks, descs = keep_l, keep_d
        ok &= check('a different room is not mistaken for the saved map', accepted == 0,
                    f'{accepted} false matches')

        # 3. covered camera for 1.5 s, then back at an earlier part of the room
        vo4 = VisualOdometry(K)
        for i in range(60):
            R, t = pose(i)
            vo4.update(*observe(R, t))
        r3 = np.random.default_rng(5)
        for _ in range(45):
            vo4.update(r3.uniform(0, 640, (300, 2)).astype(np.float32),
                       r3.integers(0, 256, (300, 32), dtype=np.uint8))
        for j in range(30):
            R, t = pose(8 + j)
            vo4.update(*observe(R, t))
        ok &= check('recovers after a short loss without starting a new map',
                    vo4.status == 'tracking' and vo4.generation == 0,
                    f'status {vo4.status}, generation {vo4.generation}')

        # 4. a long loss in an unrecognisable place starts a fresh world frame
        vo5 = VisualOdometry(K)
        for i in range(50):
            R, t = pose(i)
            vo5.update(*observe(R, t))
        for _ in range(tracker.LOST_FRAMES_BEFORE_RESEED + 10):
            vo5.update(r3.uniform(0, 640, (300, 2)).astype(np.float32),
                       r3.integers(0, 256, (300, 32), dtype=np.uint8))
        ok &= check('a long loss starts a new world frame', vo5.generation == 1,
                    f'generation {vo5.generation}')

        # 5. object placement uses the nearest points, not the wall behind
        import live_run
        rng_o = np.random.default_rng(0)
        obj = np.column_stack([rng_o.normal(0, 0.15, 6), rng_o.normal(0.3, 0.1, 6), rng_o.normal(3, 0.1, 6)])
        wall = np.column_stack([rng_o.uniform(-1.5, 1.5, 25), rng_o.uniform(-1, 1, 25), rng_o.normal(9, 0.2, 25)])
        X = np.vstack([obj, wall])
        fake = type('V', (), {})()
        fake.R, fake.t, fake.tracked_xyz = np.eye(3), np.zeros((3, 1)), X
        uv = (K @ X.T).T
        fake.tracked_px = uv[:, :2] / uv[:, 2:3]
        pos, _ = live_run.locate_object(fake, (230., 150., 410., 330.))
        ok &= check('objects are placed on the object, not the wall behind it',
                    pos is not None and abs(pos[2] - 3.0) < 0.5,
                    'no position' if pos is None else f'depth {pos[2]:.2f}, wall is 9')
        fake.tracked_xyz, fake.tracked_px = X[:2], uv[:2, :2] / uv[:2, 2:3]
        ok &= check('too few map points: no guess is saved',
                    live_run.locate_object(fake, (0., 0., 640., 480.))[0] is None)

        # 6. repeated noisy sightings of two objects end up as two rows
        sid2 = new_session(conn)
        conn.execute('UPDATE sessions SET units_per_meter = 5.6 WHERE id = ?', (sid2,))
        conn.commit()
        truth = [np.array([-2.0, 0.0, 5.0]), np.array([4.0, 0.0, 6.0])]
        r4, objects = np.random.default_rng(5), []
        for i in range(24):
            live_run.record_sighting(conn, sid2, objects, 'cell phone', 0.7,
                                     truth[i % 2] + r4.normal(0, 1.0, 3), 6.0, 'map')
        rows = conn.execute('SELECT x, y, z FROM detections WHERE session_id = ?', (sid2,)).fetchall()
        close = all(min(np.linalg.norm(np.array(r) - tp) for tp in truth) < 1.0 for r in rows)
        ok &= check('repeat sightings merge into one row per object', len(rows) == 2 and close,
                    f'{len(rows)} rows')

        # 7. directions with obstacle warnings (camera at the origin looking along +z, 5 units per meter)
        import guidance
        sid3 = new_session(conn)
        conn.execute('UPDATE sessions SET units_per_meter = 5.0 WHERE id = ?', (sid3,))

        def add(label, x, y, z):
            conn.execute("INSERT INTO detections (label, confidence, image_source, timestamp, x, y, z, "
                         "session_id, pos_source) VALUES (?, 0.8, 'selftest', '2026-01-01T00:00:00', ?, ?, ?, ?, 'map')",
                         (label, x, y, z, sid3))
        add('laptop', 1.5, 0.0, 10.0)          # 2 m ahead, a little to the right
        add('chair', 0.7, 0.0, 5.0)            # on the straight path, 1 m ahead
        add('cell phone', -3.0, 0.0, 6.0)      # to the left, nothing in between
        add('dining table', 6.0, 0.0, 4.0)     # far to the right
        conn.commit()
        write_position(sid3, np.eye(3), np.zeros((3, 1)), 'tracking')
        ok &= check('"the phone" finds the cell phone', guidance.label_matches('take me to the phone', 'cell phone')
                    and not guidance.label_matches('take me to the phone', 'laptop'))
        a = guidance.describe('take me to the laptop', conn)
        ok &= check('directions name the obstacle on the path', a['found'] and 'chair' in a['text']
                    and 'Distance' in a['text'], 'no obstacle reported' if 'chair' not in a['text'] else '')
        b = guidance.describe('take me to the cell phone', conn)
        ok &= check('a clear path reports no obstacle', b['found'] and 'Turn left' in b['text']
                    and 'none known' in b['text'])
        c3 = guidance.describe('go to the table', conn)
        ok &= check('right turn is reported as right', c3['found'] and 'Turn right' in c3['text'])
        write_position(sid3, np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]), np.zeros((3, 1)), 'tracking')
        ok &= check('a target behind you says turn around', 'Turn around' in guidance.describe('laptop', conn)['text'])
        write_position(sid3, np.eye(3), np.zeros((3, 1)), 'tracking')
        near = guidance.obstacles_ahead([{'label': 'chair', 'position': np.array([0.5, 0, 3.0])},
                                         {'label': 'chair', 'position': np.array([0.5, 0, 9.0])},
                                         {'label': 'bottle', 'position': np.array([0.1, 0, 1.0])}],
                                        np.zeros(3), np.eye(3), 5.0)
        ok &= check('only a close, large object in front triggers a warning',
                    len(near) == 1 and near[0][0] == 'chair', f'{len(near)} warnings')
        # 360 degrees: 1 m = 5 map units; the camera looks along +z, x is to the right
        around = guidance.obstacles_ahead(
            [{'label': 'chair', 'position': np.array([0.0, 0, 4.0])},     # 0.8 m straight ahead
             {'label': 'couch', 'position': np.array([4.0, 0, 0.5])},     # 0.8 m to the right
             {'label': 'bed', 'position': np.array([-3.0, 0, -3.0])},     # 0.85 m behind, to the left
             {'label': 'chair', 'position': np.array([0.0, 0, -9.0])},    # 1.8 m behind: too far
             {'label': 'chair', 'position': np.array([5.0, 0, 5.0])}],    # 1.4 m diagonal: too far
            np.zeros(3), np.eye(3), 5.0)
        ok &= check('warnings cover all directions within 1 m: ahead, beside and behind, nearest first',
                    [a[0] for a in around] == ['chair', 'couch', 'bed'] and abs(around[2][2]) > guidance.BEHIND_DEG,
                    str([(a[0], round(a[2])) for a in around]))
        spun = guidance.obstacles_ahead([{'label': 'chair', 'position': np.array([0.0, 0, 4.0])}],
                                        np.zeros(3), np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]), 5.0)
        ok &= check('turning around does not hide an object (it is now behind you)',
                    len(spun) == 1 and abs(spun[0][2]) > guidance.BEHIND_DEG)
        ok &= check('no scale means no warnings (distance unknown)',
                    guidance.obstacles_ahead([{'label': 'chair', 'position': np.array([0.0, 0, 1.0])}],
                                             np.zeros(3), np.eye(3), None) == [])

        # 8. two whole runs through live_run: the second one joins the first one's map
        import live_run
        tex = cv2.GaussianBlur(np.random.default_rng(3).integers(0, 255, (900, 2400)).astype(np.uint8), (0, 0), 1.2)

        def view(start, n, step):
            out = []
            for i in range(n):
                x0 = int(300 + start + min(i, 60) * step)
                out.append(cv2.cvtColor(tex[200:680, x0:x0 + 640], cv2.COLOR_GRAY2BGR))
            return out

        class Frames:                                  # a camera made of ready-made pictures
            def __init__(self, frames):
                self.frames, self.i = frames, 0

            def read(self):
                self.i += 1
                return (True, self.frames[self.i - 1].copy()) if self.i <= len(self.frames) else (False, None)

            def release(self):
                pass

        messages, flags = [], []

        def collect(text, speak=True):
            messages.append(text)
            flags.append((text, speak))

        detector = lambda frame: [('chair', 0.9, (250., 150., 400., 330.)), ('person', 0.9, (100., 100., 300., 400.))]
        run_a = live_run.run(save_map=True, detect=True, source=Frames(view(0, 110, 4.0)), show=False,
                             detector=detector, K=K, notify=collect)
        conn.execute('UPDATE sessions SET units_per_meter = 5.0 WHERE id = ?', (run_a,))
        conn.commit()
        run_b = live_run.run(save_map=True, detect=True, source=Frames(view(120, 100, 1.0)), show=False,
                             detector=detector, K=K, notify=collect)
        ok &= check('a second run joins the first run\'s map and keeps its scale',
                    run_b == run_a and any('Saved map found' in m for m in messages),
                    f'first run session {run_a}, second run session {run_b}')
        ok &= check('limited speech: "saved" and map messages are shown but not spoken',
                    any('saved' in t and not sp for t, sp in flags)
                    and all(not sp for t, sp in flags if 'Saved map found' in t or t.endswith(' saved')),
                    f'{sum(1 for _, sp in flags if not sp)} quiet of {len(flags)} messages')
        labels = {r[0] for r in conn.execute('SELECT label FROM detections WHERE session_id = ?', (run_a,))}
        ok &= check('people are not saved as landmarks', labels == {'chair'}, f'saved: {sorted(labels)}')

        # 9. people memory and "what is ahead" (camera at the origin looking along +z)
        import people
        guide = guidance.describe
        ok &= check('"remember this person as ..." stores the name, who they are and asks for a face',
                    'Ahmed' in guide('remember this person as Ahmed, my brother', conn)['text']
                    and people.pending_enrollment(conn) is not None)
        ok &= check('"remember that ... is ..." and "who is ..." work',
                    'your friend' in guide('remember that Sara is my friend', conn)['text']
                    and 'Ahmed is your brother' in guide('who is Ahmed', conn)['text'])
        ok &= check('an unknown name is not invented', not guide('who is Zoya', conn)['found'])
        people_rng = np.random.default_rng(1)
        face_a, face_b = people_rng.normal(size=128), people_rng.normal(size=128)
        pid_a, pid_b = people.find_in_text(conn, 'Ahmed')[0], people.find_in_text(conn, 'Sara')[0]
        for _ in range(3):
            people.save_sample(conn, pid_a, face_a + people_rng.normal(0, .15, 128))
            people.save_sample(conn, pid_b, face_b + people_rng.normal(0, .15, 128))
        known = people.load_known(conn)
        m_a = people.best_match(face_a + people_rng.normal(0, .2, 128), known)
        ok &= check('a known face is recognised', m_a is not None and m_a[1] == 'Ahmed')
        ok &= check('a stranger and a blend of two people are not named',
                    people.best_match(people_rng.normal(size=128), known) is None
                    and people.best_match((face_a + face_b) / 2, known) is None)
        sid4 = new_session(conn)
        write_position(sid4, np.eye(3), np.zeros((3, 1)), 'tracking')
        for label, x, z in [('chair', 0.5, 3.0), ('laptop', 1.0, 12.0)]:
            conn.execute("INSERT INTO detections (label, confidence, image_source, timestamp, x, y, z, session_id, "
                         "pos_source) VALUES (?, 0.8, 'selftest', '2026-01-01T00:00:00', ?, 0, ?, ?, 'map')",
                         (label, x, z, sid4))
        conn.commit()
        people.note_seen(conn, pid_a, sid4, np.array([-0.4, 0.0, 4.0]))
        write_live_scene(sid4, [{'label': 'person', 'x': 0.0, 'y': 0.0, 'z': 4.5}])
        ahead = guide('what is ahead', conn)['text']
        ok &= check('"what is ahead" lists things within about 1 m, people included, and not far ones',
                    'chair' in ahead and 'Ahmed' in ahead and 'person' in ahead and 'laptop' not in ahead, ahead[:80])
        ok &= check('"where is Ahmed" gives directions to the last seen position',
                    'Straight ahead' in guide('where is Ahmed', conn)['text'])
        guide('forget Ahmed', conn)
        ok &= check('"forget" removes the person and their face samples',
                    people.find_in_text(conn, 'Ahmed') is None
                    and conn.execute('SELECT COUNT(*) FROM face_samples WHERE person_id = ?', (pid_a,)).fetchone()[0] == 0)

        # 10. a whole run that learns a face and then recognises it (with a stand-in face engine)
        import live_run
        pid_n = people.add_person(conn, 'Nadia', 'my sister')
        people.request_enrollment(conn, pid_n)
        nadia = people_rng.normal(size=128)

        class FakeFaces:
            def detect(self, frame):
                return [(160., 120., 80., 90., None)]

            def embed(self, frame, face):
                return nadia + np.random.default_rng().normal(0, .15, 128)

        sees_person = lambda frame: [('person', 0.9, (100., 100., 300., 400.)), ('chair', 0.9, (350., 150., 500., 330.))]
        said = []
        run_c = live_run.run(save_map=True, detect=True, source=Frames(view(0, 110, 4.0)), show=False,
                             detector=sees_person, K=K, notify=said.append, face_engine=FakeFaces(), resume='new')
        ok &= check('a run learns the face it is asked to learn',
                    any('learned Nadia' in m for m in said) and people.pending_enrollment(conn) is None
                    and len([k for k in people.load_known(conn) if k[1] == 'Nadia']) == 3)
        live_run.run(save_map=True, detect=True, source=Frames(view(120, 100, 1.0)), show=False,
                     detector=sees_person, K=K, notify=said.append, face_engine=FakeFaces())
        seen = people.last_seen(conn, pid_n, run_c)
        ok &= check('the next run recognises that person and notes where they are', seen is not None)
        ok &= check('people are never stored as objects',
                    conn.execute("SELECT COUNT(*) FROM detections WHERE label = 'person' AND session_id = ?",
                                 (run_c,)).fetchone()[0] == 0)
        conn.close()
    finally:
        tracker.DB_PATH, tracker.POSITION_FILE, tracker.LIVE_SCENE_FILE = old_db, old_pos, old_scene
    return ok


def main():
    results = [run_scenario('turn'), run_scenario('swing'), run_logic_checks()]
    print("PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
