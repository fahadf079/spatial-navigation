"""Shared tracking, database and position helpers for the Spatial Navigation System.

World frame: the camera pose at the start of a run (x right, y down, z forward).
Every run is its own session, so coordinates are only comparable inside one session.
"""
import datetime
import json
import os
import sqlite3

import cv2
import numpy as np

DB_PATH = 'spatial_memory.db'
POSITION_FILE = 'current_position.json'
CALIB_FILE = 'camera_calibration.npz'
LIVE_SCENE_FILE = 'live_scene.json'
# Used only when a session has no measured scale. The scales measured so far ranged from about
# 3 to 10 map units per meter, so distances from this default are rough (up to about 2x off).
DEFAULT_UNITS_PER_METER = 5.0

MIN_PARALLAX_PX = 12.0      # median pixel motion needed before a new keyframe
MIN_PNP_POINTS = 8          # known 3D points needed to track with PnP
MIN_PNP_INLIERS = 8         # of those, how many must agree on one pose
MIN_PNP_INLIER_RATIO = 0.4  # and at least this share of them ...
PNP_ENOUGH_INLIERS = 25     # ... unless this many agree, which is enough on its own
PNP_REPROJ_PX = 4.0         # pixel error allowed for a point to agree with a pose
TRACKED_REPROJ_PX = 6.0     # looser limit for points reused to place detected objects
MIN_E_INLIERS = 30          # inliers needed to trust an essential-matrix pose
MIN_NEW_POINTS = 20         # triangulated points needed to create a keyframe
MIN_RAY_ANGLE_DEG = 1.0     # reject points with almost no parallax
MAX_REPROJ_PX = 2.0         # reject badly triangulated points
LOST_FRAMES_BEFORE_RESEED = 90        # about 3 s of video: keep trying to recover first
LOST_FRAMES_BEFORE_RESEED_INIT = 5    # while still initializing there is nothing to recover
MAX_OLD_KEYFRAMES = 6                 # earlier views kept for recovery after a loss
RELOC_MIN_INLIERS = 30      # a saved view must agree with this many points to be trusted
RELOC_MIN_RATIO = 0.35      # and with at least this share of its matched points
RELOC_KFS_PER_TRY = 8       # saved views tried per attempt (they are cycled through)
RELOC_EVERY_FRAMES = 3      # attempt every few frames while lost
RELOC_AGREE_UNITS = 0.5     # two attempts must agree on position within this many map units ...
RELOC_AGREE_DEG = 5.0       # ... and on heading within this many degrees
MAX_MAP_KEYFRAMES = 400     # keyframes kept in memory for one world frame


# ----------------------------------------------------------------- database
def _columns(conn, table):
    return {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}


def ensure_schema(conn):
    """Create missing tables and columns. Safe to call on every start."""
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, confidence REAL,
        image_source TEXT, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS point_cloud (
        id INTEGER PRIMARY KEY AUTOINCREMENT, x REAL, y REAL, z REAL, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, units_per_meter REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE COLLATE NOCASE, who TEXT,
        created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS face_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, embedding BLOB, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS people_seen (
        id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, session_id INTEGER,
        x REAL, y REAL, z REAL, timestamp TEXT, UNIQUE (person_id, session_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS enroll_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, samples_needed INTEGER,
        created_at TEXT, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS keyframes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, created_at TEXT, n INTEGER,
        pts BLOB, des BLOB, pts3d BLOB, R BLOB, t BLOB)''')
    wanted = {
        'detections': [('x', 'REAL DEFAULT 0.0'), ('y', 'REAL DEFAULT 0.0'),
                       ('z', 'REAL DEFAULT 0.0'), ('session_id', 'INTEGER DEFAULT 0'),
                       ('pos_source', "TEXT DEFAULT 'camera'")],
        'point_cloud': [('session_id', 'INTEGER DEFAULT 0')],
    }
    for table, cols in wanted.items():
        have = _columns(conn, table)
        for name, ddl in cols:
            if name not in have:
                c.execute(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}')
    conn.commit()


def connect():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    return conn


def new_session(conn):
    cur = conn.execute('INSERT INTO sessions (started_at) VALUES (?)',
                       (datetime.datetime.now().isoformat(),))
    conn.commit()
    return cur.lastrowid


def get_units_per_meter(conn, session_id):
    row = conn.execute('SELECT units_per_meter FROM sessions WHERE id = ?',
                       (session_id,)).fetchone()
    return row[0] if row and row[0] else None


def effective_scale(conn, session_id):
    """(map units per meter, measured?). Falls back to a rough default when none was measured."""
    measured = get_units_per_meter(conn, session_id) if session_id else None
    return (measured, True) if measured else (DEFAULT_UNITS_PER_METER, False)


# ------------------------------------------------------------ saved map views
def save_keyframe(conn, session_id, kf):
    """Store one keyframe (keypoints, descriptors, 3D points, pose) so a later run can
    recognise the same place and continue in the same coordinates."""
    conn.execute(
        'INSERT INTO keyframes (session_id, created_at, n, pts, des, pts3d, R, t) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (int(session_id), datetime.datetime.now().isoformat(), int(len(kf['pts'])),
         np.asarray(kf['pts'], np.float32).tobytes(), np.asarray(kf['des'], np.uint8).tobytes(),
         np.asarray(kf['pts3d'], np.float32).tobytes(),
         np.asarray(kf['R'], np.float64).tobytes(), np.asarray(kf['t'], np.float64).tobytes()))
    conn.commit()


def load_keyframes(conn, session_id):
    out = []
    for pts, des, pts3d, R, t in conn.execute(
            'SELECT pts, des, pts3d, R, t FROM keyframes WHERE session_id = ? ORDER BY id',
            (int(session_id),)):
        out.append({'pts': np.frombuffer(pts, np.float32).reshape(-1, 2).copy(),
                    'des': np.frombuffer(des, np.uint8).reshape(-1, 32).copy(),
                    'pts3d': np.frombuffer(pts3d, np.float32).reshape(-1, 3).astype(np.float64),
                    'R': np.frombuffer(R, np.float64).reshape(3, 3).copy(),
                    't': np.frombuffer(t, np.float64).reshape(3, 1).copy()})
    return out


def latest_map_session(conn):
    """Session of the most recently saved map, or None."""
    row = conn.execute('SELECT session_id FROM keyframes GROUP BY session_id '
                       'ORDER BY MAX(id) DESC LIMIT 1').fetchone()
    return row[0] if row else None


# ----------------------------------------------------------------- position
def camera_center(R, t):
    return (-np.asarray(R).T @ np.asarray(t).reshape(3, 1)).ravel()


def write_position(session_id, R, t, status):
    c = camera_center(R, t)
    data = {'session_id': int(session_id), 'x': float(c[0]), 'y': float(c[1]),
            'z': float(c[2]), 'rotation': np.asarray(R).tolist(), 'status': status,
            'timestamp': datetime.datetime.now().isoformat()}
    tmp = POSITION_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, POSITION_FILE)
    except OSError:
        pass  # a reader had the file open; the next frame will write it again


def write_live_scene(session_id, items):
    """What the camera sees right now (people included), for 'what is ahead' questions."""
    data = {'session_id': int(session_id), 'timestamp': datetime.datetime.now().isoformat(),
            'items': items}
    tmp = LIVE_SCENE_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, LIVE_SCENE_FILE)
    except OSError:
        pass


def read_live_scene(max_age_seconds=5.0):
    """Items seen in the last few seconds in the current session, or an empty list."""
    try:
        with open(LIVE_SCENE_FILE) as f:
            d = json.load(f)
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(d['timestamp'])).total_seconds()
    except (OSError, ValueError, KeyError):
        return {'session_id': None, 'items': []}
    return d if age <= max_age_seconds else {'session_id': d.get('session_id'), 'items': []}


def read_position():
    """Return the latest position dict, or None if odometry never ran."""
    if not os.path.exists(POSITION_FILE):
        return None
    try:
        with open(POSITION_FILE) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    d.setdefault('session_id', 0)
    d.setdefault('rotation', np.eye(3).tolist())
    return d


# ----------------------------------------------------------------- odometry
class VisualOdometry:
    """Monocular visual odometry with a growing map.

    Pose is stored world-to-camera: x_cam = R @ x_world + t.
    The first baseline sets the scale (1 unit). Later frames are tracked against
    the map with PnP, so the scale stays consistent and rotation is included.
    """

    def __init__(self, K):
        self.K = np.asarray(K, dtype=np.float64)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.kf = None
        self.has_map = False
        self.scale_known = False
        self.last_step = 1.0
        self.lost = 0
        self.status = 'initializing'
        self.reason = ''
        self.pnp_note = ''
        self.old_kfs = []        # earlier keyframes (same world frame), newest first
        self.generation = 0      # increases when the world frame is reset after a long loss
        self.map_kfs = []        # every keyframe of this world frame (for recognising places)
        self._unsaved = []       # keyframes not yet handed to the caller for saving
        self._reloc_cursor = 0
        self._reloc_prev = None
        self.last_inliers = 0
        self.new_points = np.empty((0, 3))
        self.tracked_px = np.empty((0, 2))
        self.tracked_xyz = np.empty((0, 3))

    @property
    def position(self):
        return camera_center(self.R, self.t)

    def _set_kf(self, pts, des, pts3d):
        if self.kf is not None and np.isfinite(self.kf['pts3d']).all(axis=1).any():
            self.old_kfs.insert(0, self.kf)
            del self.old_kfs[MAX_OLD_KEYFRAMES:]
        self.kf = {'pts': pts, 'des': des, 'pts3d': pts3d,
                   'R': self.R.copy(), 't': self.t.copy()}
        if np.isfinite(pts3d).all(axis=1).any():          # a real keyframe, not a placeholder
            self.map_kfs.append(self.kf)
            del self.map_kfs[:-MAX_MAP_KEYFRAMES]
            self._unsaved.append(self.kf)

    def drain_new_keyframes(self):
        """Keyframes created since the last call, so the caller can save them."""
        out, self._unsaved = self._unsaved, []
        return out

    def _use_keyframe(self, old):
        """Make an earlier keyframe the current one (it is in the same world frame)."""
        if self.kf is not None and np.isfinite(self.kf['pts3d']).all(axis=1).any():
            self.old_kfs.insert(0, self.kf)
        self.old_kfs = [k for k in self.old_kfs if k is not old][:MAX_OLD_KEYFRAMES]
        self.kf = old

    def _match(self, kf, pts, des):
        """Match a keyframe against the current frame. Returns (match data or None, count)."""
        matches = self.bf.match(kf['des'], des)
        if len(matches) < 20:
            return None, len(matches)
        qi = np.array([m.queryIdx for m in matches])
        ti = np.array([m.trainIdx for m in matches])
        return (qi, ti, kf['pts'][qi].astype(np.float64), pts[ti].astype(np.float64)), len(matches)

    def update(self, pts, des):
        """Feed ORB keypoint pixels (N,2) and descriptors (N,32). Returns status."""
        self.new_points = np.empty((0, 3))
        self.tracked_px = np.empty((0, 2))
        self.tracked_xyz = np.empty((0, 3))
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if des is None or len(pts) < 20:
            self.reason = f'only {len(pts)} keypoints: image too dark, blurry or blank'
            return self._lost(None, None)
        if self.kf is None:
            self._set_kf(pts, des, np.full((len(pts), 3), np.nan))
            return self.status

        pose, inliers, note, parallax = None, None, '', 0.0
        m, n_matches = self._match(self.kf, pts, des)
        if m is not None:
            qi, ti, p_kf, p_cur = m
            parallax = float(np.median(np.linalg.norm(p_cur - p_kf, axis=1)))
            if self.has_map:
                pose, note = self._track_pnp(self.kf, qi, p_cur)

        # Lost against the newest keyframe: try the earlier views before giving up.
        tried_old = 0
        if pose is None and self.has_map:
            for old in list(self.old_kfs):
                mo, _ = self._match(old, pts, des)
                tried_old += 1
                if mo is None:
                    continue
                pose_old, _ = self._track_pnp(old, mo[0], mo[3])
                if pose_old is not None:
                    pose = pose_old
                    qi, ti, p_kf, p_cur = mo
                    self._use_keyframe(old)
                    break

        # Still lost: look for the place among ALL saved views of this map, not just recent ones.
        if (pose is None and self.has_map and self.lost >= RELOC_EVERY_FRAMES
                and self.lost % RELOC_EVERY_FRAMES == 0 and len(self.map_kfs) > 1):
            hit = self.relocalize(pts, des)
            if hit is not None:
                self.adopt_pose(*hit)
                return self.status

        if pose is None and m is not None and parallax >= MIN_PARALLAX_PX:
            pose, inliers = self._track_essential(p_kf, p_cur)
        self.pnp_note = note
        if pose is None:
            if not self.has_map and m is not None and parallax < MIN_PARALLAX_PX:
                self.status = 'initializing'   # waiting for the camera to move
                return self.status
            if m is None:
                self.reason = f'only {n_matches} matches with the last keyframe: moved too fast or view changed'
            else:
                self.reason = f'{n_matches} matches, parallax {parallax:.0f}px, {note or "no map yet"}'
            if tried_old:
                self.reason += f'; {tried_old} earlier keyframes also failed'
            return self._lost(pts, des)

        # Real parallax: pixel motion left after removing the camera's rotation.
        Rrel = pose[0] @ self.kf['R'].T
        H = self.K @ Rrel @ np.linalg.inv(self.K)
        pred = cv2.perspectiveTransform(p_kf.reshape(-1, 1, 2), H).reshape(-1, 2)
        true_parallax = float(np.median(np.linalg.norm(p_cur - pred, axis=1)))
        if not self.has_map and true_parallax < MIN_PARALLAX_PX:
            self.status = 'initializing'   # turning in place: no usable baseline yet
            return self.status

        self.R, self.t = pose
        self.lost = 0
        self.status = 'tracking' if self.has_map else 'initializing'
        if true_parallax >= MIN_PARALLAX_PX:
            self._new_keyframe(pts, des, qi, ti, p_kf, p_cur, inliers)
        return self.status

    def _track_pnp(self, kf, qi, p_cur, strict=False):
        """Pose from known 3D points of keyframe kf. Returns (pose or None, note)."""
        world = kf['pts3d'][qi]
        ok = np.isfinite(world).all(axis=1)
        if ok.sum() < MIN_PNP_POINTS:
            return None, f'only {int(ok.sum())} matches have a known 3D point (need {MIN_PNP_POINTS})'
        world_ok = world[ok].astype(np.float64)
        px_ok = p_cur[ok]
        try:
            success, rvec, tvec, inl = cv2.solvePnPRansac(
                world_ok, px_ok, self.K, None, iterationsCount=200,
                reprojectionError=PNP_REPROJ_PX, confidence=0.99)
        except cv2.error:
            success, inl = False, None
        n_in = 0 if inl is None else len(inl)
        self.last_inliers = n_in
        if strict:
            weak = n_in < RELOC_MIN_INLIERS or n_in < RELOC_MIN_RATIO * len(world_ok)
        else:
            weak = n_in < PNP_ENOUGH_INLIERS and n_in < MIN_PNP_INLIER_RATIO * len(world_ok)
        if not success or n_in < MIN_PNP_INLIERS or weak:
            return None, (f'only {n_in} of {len(world_ok)} mapped matches agree on a pose '
                          f'(need {PNP_ENOUGH_INLIERS}, or {MIN_PNP_INLIERS} and {MIN_PNP_INLIER_RATIO:.0%})')
        idx = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(world_ok[idx], px_ok[idx], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        # Points that agree with the final pose (looser limit) are used to place objects.
        proj, _ = cv2.projectPoints(world_ok, rvec, tvec, self.K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - px_ok, axis=1)
        keep = err < TRACKED_REPROJ_PX
        self.tracked_px = px_ok[keep]
        self.tracked_xyz = world_ok[keep]
        return (R, tvec.reshape(3, 1)), ''

    # ------------------------------------------------------------ recognising places
    def _reloc_view(self, kf):
        """Keyframe reduced to its keypoints that have a 3D point (faster to match)."""
        v = kf.get('_reloc')
        if v is None:
            ok = np.isfinite(kf['pts3d']).all(axis=1)
            v = {'pts': kf['pts'][ok], 'des': kf['des'][ok], 'pts3d': kf['pts3d'][ok],
                 'R': kf['R'], 't': kf['t']}
            kf['_reloc'] = v
        return v

    @staticmethod
    def _poses_agree(a, b):
        da = np.linalg.norm(camera_center(*a) - camera_center(*b))
        ang = np.degrees(np.arccos(np.clip((np.trace(a[0] @ b[0].T) - 1) / 2, -1, 1)))
        return da < RELOC_AGREE_UNITS and ang < RELOC_AGREE_DEG

    def relocalize(self, pts, des, kfs=None):
        """Try to find the camera in a set of saved views (default: this map's keyframes).

        Each call tries a few views, cycling through all of them. A hit is only returned when
        two attempts in a row agree on the pose, so a wrong match cannot move the camera.
        Returns (keyframe, (R, t)) or None.
        """
        kfs = self.map_kfs if kfs is None else kfs
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if not kfs or des is None or len(pts) < 20:
            self._reloc_prev = None
            return None
        keep = (self.tracked_px, self.tracked_xyz)
        best = None
        for _ in range(min(RELOC_KFS_PER_TRY, len(kfs))):
            kf = kfs[self._reloc_cursor % len(kfs)]
            self._reloc_cursor += 1
            m, _ = self._match(self._reloc_view(kf), pts, des)
            if m is None:
                continue
            pose, _ = self._track_pnp(self._reloc_view(kf), m[0], m[3], strict=True)
            if pose is not None and (best is None or self.last_inliers > best[2]):
                best = (kf, pose, self.last_inliers)
        self.tracked_px, self.tracked_xyz = keep
        if best is None:
            self._reloc_prev = None
            return None
        kf, pose, _ = best
        prev, self._reloc_prev = self._reloc_prev, (pose, kf)
        if prev is not None and self._poses_agree(prev[0], pose):
            self._reloc_prev = None
            return kf, pose
        return None

    def adopt_pose(self, kf, pose):
        """Continue tracking from a pose found by relocalize()."""
        self.R, self.t = pose
        self._use_keyframe(kf)
        self.old_kfs = [k for k in self.map_kfs[-MAX_OLD_KEYFRAMES - 1:] if k is not kf][-MAX_OLD_KEYFRAMES:]
        self.has_map = True
        self.scale_known = True
        self.lost = 0
        self.status = 'tracking'

    def load_map(self, kfs, pose, kf):
        """Switch to a saved map (from an earlier run) at a pose found by relocalize()."""
        self.map_kfs = list(kfs)
        self._unsaved = []
        centers = [camera_center(k['R'], k['t']) for k in kfs]
        steps = [float(np.linalg.norm(b - a)) for a, b in zip(centers, centers[1:])]
        steps = [x for x in steps if x > 1e-9]
        self.last_step = float(np.median(steps)) if steps else 1.0
        self.kf = None
        self.adopt_pose(kf, pose)

    def _track_essential(self, p_kf, p_cur):
        E, mask = cv2.findEssentialMat(p_kf, p_cur, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return None, None
        E = E[:3]
        n, Rr, tr, mpose = cv2.recoverPose(E, p_kf, p_cur, self.K, mask=mask.copy())
        if n < MIN_E_INLIERS:
            return None, None
        scale = self.last_step if self.scale_known else 1.0
        # recoverPose gives previous-to-current motion: x_cur = Rr @ x_kf + tr
        R = Rr @ self.kf['R']
        t = Rr @ self.kf['t'] + tr * scale
        return (R, t), mpose.ravel() > 0

    def _good_points(self, X, p1, p2, R1, t1, R2, t2):
        c1, c2 = camera_center(R1, t1), camera_center(R2, t2)
        with np.errstate(all='ignore'):
            ok = np.isfinite(X).all(axis=1)
            Xc1 = (R1 @ X.T + t1).T
            Xc2 = (R2 @ X.T + t2).T
            ok &= (Xc1[:, 2] > 0) & (Xc2[:, 2] > 0)

            def reproj(Xc, p):
                uvw = (self.K @ Xc.T).T
                return np.linalg.norm(uvw[:, :2] / uvw[:, 2:3] - p, axis=1)

            ok &= (reproj(Xc1, p1) < MAX_REPROJ_PX) & (reproj(Xc2, p2) < MAX_REPROJ_PX)
            r1, r2 = X - c1, X - c2
            cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
            ok &= np.degrees(np.arccos(np.clip(cos, -1, 1))) >= MIN_RAY_ANGLE_DEG
        return ok

    def _new_keyframe(self, pts, des, qi, ti, p_kf, p_cur, inliers):
        mask = np.ones(len(qi), bool) if inliers is None else inliers
        if mask.sum() < MIN_NEW_POINTS:
            return
        R1, t1 = self.kf['R'], self.kf['t']
        P1 = self.K @ np.hstack((R1, t1))
        P2 = self.K @ np.hstack((self.R, self.t))
        X4 = cv2.triangulatePoints(P1, P2, p_kf[mask].T, p_cur[mask].T)
        with np.errstate(all='ignore'):
            X = (X4[:3] / X4[3]).T
        good = self._good_points(X, p_kf[mask], p_cur[mask], R1, t1, self.R, self.t)
        if good.sum() < MIN_NEW_POINTS:
            return
        sel = np.flatnonzero(mask)[good]
        prev3d = self.kf['pts3d'][qi[sel]]
        had_prev = np.isfinite(prev3d).all(axis=1)
        chosen = np.where(had_prev[:, None], prev3d, X[good])
        pts3d = np.full((len(pts), 3), np.nan)
        pts3d[ti[sel]] = chosen
        self.new_points = X[good][~had_prev]
        step = float(np.linalg.norm(self.position - camera_center(R1, t1)))
        if step > 1e-9:
            self.last_step = step
        self._set_kf(pts, des, pts3d)
        self.has_map = True
        self.scale_known = True
        self.status = 'tracking'

    def _lost(self, pts, des):
        self.lost += 1
        had_map = self.has_map
        self.status = 'lost' if had_map else 'initializing'
        limit = LOST_FRAMES_BEFORE_RESEED if had_map else LOST_FRAMES_BEFORE_RESEED_INIT
        if self.lost >= limit and pts is not None and des is not None and len(pts) >= 20:
            self._reseed(pts, des, had_map)
        return self.status

    def _reseed(self, pts, des, had_map):
        """Start over from the current frame with a fresh world frame and scale.

        The held pose is stale after a long loss, so keeping it would put new points in the
        wrong place inside the old map. generation tells the caller to open a new session.
        """
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.old_kfs = []
        self.map_kfs = []
        self._unsaved = []
        self._reloc_prev = None
        self.kf = None
        self._set_kf(pts, des, np.full((len(pts), 3), np.nan))
        self.has_map = False
        self.scale_known = False
        self.last_step = 1.0
        self.lost = 0
        self.status = 'initializing'
        if had_map:
            self.generation += 1
