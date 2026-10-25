# Spatial Navigation System

A single-camera system that works out where the camera is, remembers where it saw objects and people, and guides you with spoken directions and obstacle warnings. It is a pilot built for blind users who carry a phone in a front pocket with the camera facing forward. This file is the only documentation in the repo.

**Pilot, not a safety device.** It is advisory only. Do not rely on it to avoid danger. Accuracy has not been reliably measured yet, see the results table (the pilot target is about 1 m), and it only knows what the camera has already seen.

From one webcam (or a phone) it can:

1. **Track the camera** in 3D: position (X/Y/Z) and rotation, live on screen.
2. **Detect objects** with YOLOv8n (COCO classes). Landmarks come from the model, not from a hand-written list.
3. **Remember** each object and where it was seen, in a SQLite database, saved as the camera runs.
4. **Join separate runs:** the map is saved, so a later run that sees a place you scanned before continues in the same coordinates.
5. **Guide you** to an object ("take me to the laptop") with a turn angle, a distance and a warning about obstacles on the way, by text and by voice.
6. **Say what is ahead:** "what is ahead" lists what is within about 1 m in a straight line in front of the camera, people included.
7. **Remember people:** their names and who they are ("Ahmed, my brother"), recognise them by face, and say when one is near.
8. **Speak and listen:** warnings and answers are spoken, and questions can be asked by voice.
9. **Build a 3D point cloud** of the room and plot it as a density heatmap.

## Project decisions (from the project owner)

| Question | Decision |
|---|---|
| Sound | Spoken warnings and voice questions only (section 1.6). |
| Amount of speech | Keep it limited: one short line with the object name and distance, then silence. **Implemented** (section 1.6). |
| Consent | Faces are of friends, for pilot testing only. Data stays local and is archived when the pilot ends (see Privacy). |
| Tracking | Expected to stay weak in the pilot. Scan carefully with clear corners and rely on the saved maps. |
| Identical rooms | Test at least 2 pairs of look-alike rooms (section 5). |
| Motion sensors | The phone's motion sensors are used in a learning mode at the start, which sets the scale without a tape (section 3.6). **Implemented and tested on simulated walks only; not yet tested on a real phone.** |
| Obstacles | Approved as built. One addition: memory of people's names and who they are (section 1.7). |
| What "1 m" means | What comes next within about 1 m in a straight line in front of the camera, relative to the camera's rotation and angle. |
| Camera | The webcam for development. The target user is a blind person with the phone in a front pocket, camera facing out. |
| Phone trial | Android with Chrome, because it is the easiest (section 3.5). |
| Separate runs | They are joined, so the map is saved and reused (section 1.3). |
| Saved maps | Kept for now. Nothing is deleted. |
| Test entries | Kept for the pilot. Old rows are only hidden by default in `view_memory.py` (use `--all`). |
| Scope | The pilot stops at the phone. Whatever is built on top of it (apps, hardware) is up to the client. |

Pilot scope: one room, one camera, phone level only.

---

## 1. How it works

### 1.1 Coordinates and sessions
- The **world frame is the camera's pose at the start of a map**: x right, y down, z forward.
- A **session** is one map. It has its own origin and its own scale. Numbers are only comparable inside one session, and tools warn when you mix sessions.
- A run continues the previous session when it recognises the place (section 1.3). Otherwise it starts a new session.
- Rows from before sessions existed are marked `legacy`. Their positions are placeholders and mean nothing.

### 1.2 Scale
A single camera cannot see real-world size. The first sideways movement of a map defines **1 map unit**. To get meters, run `set_scale.py` once per session: move the camera a distance you measured with a tape and it stores *map units per meter* for that session. Tools then print meters and the obstacle and distance logic use real distances. A session that is continued by a later run keeps its scale.

### 1.3 Tracker and saved maps (`tracker.py`)
- Finds ORB features in each frame and matches them against stored keyframes.
- **Initialization:** the first sideways movement is solved with the essential matrix and sets the scale.
- **Tracking:** after that the camera pose is solved with PnP (RANSAC) against known 3D map points, so scale stays consistent and rotation is included.
- **Keyframes** are added when there is real parallax: at least 12 px of motion left after removing the camera's rotation.
- **New map points** are triangulated from two views and kept only if they have positive depth in both views, reprojection error under 2 px and a viewing angle of at least 1 degree. At least 20 must survive.
- **Every keyframe is saved** in the database (keypoints, descriptors, 3D points, pose).
- **Recovery inside a run:** if tracking is lost, the tracker tries recent keyframes, and every few frames it tries all saved views of the map. Two attempts in a row must agree on the pose before it is accepted, so a wrong match cannot move the camera.
- **Joining runs:** at the start of a run, `live_run.py` loads the most recently saved map and looks for it for about 120 frames while you aim at a place you scanned before. If it is recognised, the run continues that session: same coordinates, same scale, same objects. If not, it starts a new session and keeps what it saw. `--new` skips the search and `--map 17` picks a map.
- **Restart:** after about 90 lost frames with no recognised view it starts a fresh world frame and a new session, so old and new points are never mixed.
- The pose is stored world-to-camera: `x_cam = R @ x_world + t`.

### 1.4 Objects (`live_run.py`)
- YOLO runs on every 10th frame at confidence 0.6 by default.
- An object's position is the **median of the nearest tracked map points inside its box** (the box grows by 20% if it has too few points). Using the nearest points keeps the object from being placed on the wall behind it. At least 3 points are needed. With fewer, the object is **not saved**, because a guess would be wrong.
- Objects are saved only while status is `tracking`. Labels in the ignore list (default `person`) are never saved, because people move and are not landmarks.
- **One row per object.** A repeat sighting of the same label within 0.75 m (once the scale is known) updates that row with the median of the sightings so far. Two objects with the same label closer than that are treated as one.

### 1.5 Guidance, "what is ahead" and obstacles (`guidance.py`)
- **"Ahead"** means in a straight line in front of the camera, within about 1 m, within 25 degrees of the camera's forward direction. It follows the camera's rotation and angle, so for a phone in a pocket it is whatever the phone is facing.
- Your question is matched against the stored labels (whole phrases or single words, any case, plurals and a few synonyms such as "sofa" and "monitor").
- It uses the newest match from the **current session** and gives a turn angle and distance relative to the camera's heading. If the object was only seen in an older session it says so.
- **Obstacles:** stored objects of obstacle-sized classes (chair, couch, table, tv, plant, bed, laptop and similar) that stand within 0.5 m of the straight path to the target are reported with their distance and side.
- **Live warning:** while running with sound, a large object or a person less than 1 m ahead is announced ("Obstacle ahead: chair, 0.8 meters"). People seen by the camera right now count, even strangers.
- **"What is ahead":** say or type "what is ahead", "what is in front of me" or "is the way clear". The answer lists stored objects, remembered people seen a moment ago, and people the camera sees right now.
- **Scale:** distances need a scale. Without a measured one a rough default of 5 map units per meter is used, and answers say it is estimated. It can be up to about 2 times off, so measure it with `set_scale.py` when you can.
- Every answer ends with a reminder that positions are approximate (about 1 m).

### 1.6 Sound (`sound.py`)
- **Speaking** needs no extra packages on Windows (it uses the speech engine built into Windows). On macOS it uses `say`, on Linux `espeak` or `spd-say`.
- **Listening** on Windows uses the built-in recognizer (needs a microphone and Windows speech recognition for your language). Elsewhere it uses the `SpeechRecognition` package if installed. The web page can also listen through the browser (Chrome, Edge or Safari).
- If a computer has no speech support, everything still works and messages are printed.
- Sound here means spoken warnings and answers plus voice questions, as the project owner asked. Locating things by sound itself is not part of the pilot.

**Limited speech.** Only one short line is spoken: the object and its distance (`Chair, 80 cm.`,
`Chair ahead, 0.8 meters`) or a short warning (`Tracking lost.`). Everything else (`chair saved`, map messages,
details) is printed or shown but not spoken. When no scale was measured the spoken distance says "about".
`python voice.py --full` speaks the whole answer, and `guidance.LIMITED_SPEECH = False` turns limited speech off.

**Speech cannot hang the program.** Every spoken line has a time limit (4 seconds plus 0.6 seconds per word, never more
than 20). At the limit the speech process and its helper processes are killed. After 3 time-outs within a minute, speech is
paused for a minute and messages are still printed, so a broken speech engine can never hold up a warning. Ctrl+C skips a
line. This was tested with stand-in processes; the Windows speech engine itself has not been tested on a real machine yet.

**When tracking is lost, nothing stale is said.** "What is ahead" and directions say only `Tracking lost.` (or
`Tracking is not ready.`) and give no objects or directions, because the last position can no longer be trusted. If the
camera program has stopped (the position file is more than 4 seconds old) they say `Camera is not running.`
People are not saved on the map, but "person ahead" warnings use people seen in the last 3 seconds (not yet tested with a
real person walking ahead).

### 1.7 People memory (`people.py`, `faces.py`)
Say or type:

| Say | What happens |
|---|---|
| `remember this person as Ahmed, my brother` | Saves the name and who they are, then learns the face the camera sees in the next minute (3 pictures). |
| `remember that Sara is my friend from university` | Saves the name and who they are, without a face. |
| `who is Ahmed` | Says who they are and when they were last seen. |
| `who do you know` | Lists everyone remembered. |
| `where is Ahmed` / `take me to Ahmed` | Directions to where Ahmed was last seen in this map, with a note that people move. |
| `forget Ahmed` | Removes the person, their face samples and their sightings. |

- When a remembered person is in view, the system says their name and who they are ("Ahmed, your brother, 2.0 meters" is what is spoken) and `what is ahead` names them.
- Face recognition uses two small OpenCV models (YuNet and SFace). They download once into the `.spatial_navigation/models` folder in your user folder and are checked against their published hashes. If they cannot be downloaded, people can still be remembered by name, but not recognised by face.
- A face is only accepted when it is clearly closest to one remembered person, so look-alikes are not confused. A wrong or missed recognition can still happen.
- **Privacy:** face samples are biometric data. They stay in the local database. Only do this with the person's agreement, and use `forget` to remove someone. Recognised people are not stored as ordinary objects.

---

## 2. Setup

Requires Python 3 and a webcam.

```powershell
cd spatial-navigation
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`) download automatically the first time detection runs.

Only for the web dashboard and the phone mode:
```powershell
pip install fastapi "uvicorn[standard]" cryptography
```

### Calibration
`camera_calibration.npz` holds the webcam's intrinsics (focal length about 387 px and optical center about (319.9, 235.9) for the author's webcam). It is created by `calibrate.py` from checkerboard photos. **A different webcam must be calibrated again**, otherwise distances and angles will be wrong. See the comments at the top of `calibrate.py`. The phone mode estimates its own approximate intrinsics from the picture size, so it needs no calibration (and is less accurate).

### Secrets
`.env` is gitignored. Never commit API keys.

---

## 3. Running it

All commands run **from the project folder** and start with `python`.

### 3.1 Live scan
```powershell
python live_run.py
```
- Aim at something with texture (shelves, posters, a cluttered desk). Blank walls and dark rooms give the tracker too few features.
- Hold still for a second and wait for the status `tracking`.
- If a map was saved before, the first seconds say `Looking for the saved map...`. Aim at a place you scanned before to continue it, or just move on and a new map starts.
- Move slowly and smoothly, with each view overlapping the last by about half. Do not lift the camera and put it somewhere else.
- Press `q` in the video window to stop.

| Flag | Meaning |
|---|---|
| `--no-detect` | tracking and map only |
| `--no-map` | tracking and detection only |
| `--video clip.mp4` | use a video file instead of the webcam |
| `--conf 0.5` | detection confidence (default 0.6) |
| `--model yolov8s.pt` | a bigger, more accurate YOLO model (slower) |
| `--ignore person,chair` | labels never saved (default `person`). `--ignore none` saves everything |
| `--speak` | say warnings out loud (tracking lost, obstacle ahead, object saved) |
| `--new` | do not look for the saved map; start a fresh one |
| `--map 17` | continue the saved map of session 17 |
| `--dshow` | open the webcam with DirectShow. Try this if the picture turns green or freezes on Windows |

**On-screen overlay:** X/Y/Z of the camera in yellow, the status (`tracking` green, `lost` red, `initializing` yellow), counts of points and objects, and boxes around detected objects (green = saved, orange = not saved). A bar at the bottom shows `LOST n/90` while tracking is lost.

**Console:** every status change is printed with a reason for each loss. Each saved object is printed. `Done.` summarises points saved, objects saved, repeat sightings merged and detections skipped.

If the camera picture is blank or frozen for about a second, the script says so and reopens the camera.

### 3.2 Setting the scale (second terminal)
Keep `live_run.py` running and showing `tracking`. In a second terminal, in the project folder:
```powershell
python set_scale.py
```
Type the distance you will move (0.1 to 3 m, for example `0.3`), press Enter at the start point, slide the camera that distance in a straight line along a tape, hold still and press Enter. Do this early in a new map. The measurement is discarded if the tracker restarts or stalls while you move.

### 3.3 Looking at the results
```powershell
python view_memory.py
python navigate.py "take me to the laptop"
python voice.py
python distance_diff.py 75 78
python check_accuracy.py 75 78 0.5
python visualize_pointcloud.py
```
- `view_memory.py` lists the objects of the **newest session**, with meters when the session has a scale. Use `--session 17`, `--label phone` or `--all` (everything, including legacy rows).
- `navigate.py` prints directions and obstacle warnings. Add `--speak` to hear them.
- `voice.py` is a small conversation: type a question (or use `--listen` to speak it) and hear the answer. Say or type `quit` to stop.
- `distance_diff.py` gives the distance between two detection IDs and warns when they come from different sessions.
- `check_accuracy.py ID1 ID2 TAPE_METERS` compares that distance with a tape measurement and says whether the error is within the 1 m target.
- `visualize_pointcloud.py` plots one session (the latest by default) and saves `pointcloud_filtered.png`.

### 3.4 Dashboard and API
```powershell
python api.py
```
Open http://127.0.0.1:8000. It shows the live status and the stored objects, and lets you type or speak a question and hear the answer. The same data is available as JSON: `/status`, `/sessions`, `/detections`, `/pointcloud`, `/navigate?query=...` and `/events` (interactive docs at `/docs`). The API only reads.

### 3.5 Phone camera in pocket mode (web-based pipeline)
Built and meant for Android with Chrome.
```powershell
python api.py --phone
```
1. The computer prints an `https://...` address with a key. Open it on the phone (same Wi-Fi) in Chrome.
2. Accept the certificate warning (the certificate is self-made and stays on this computer), then press **Start camera**.
3. The phone says it is a pilot and not a safety device. With learning mode on (the default) you hold the phone in front of you and walk about 8 steps so it can set the scale (section 3.6). Then it asks you to put the phone in your front pocket with the camera facing forward and waits 10 seconds. It streams to the computer, which runs the same tracking, detection, people recognition and saving as `live_run.py`.
4. Everything is voice, and speech is limited to one short line (section 1.6). Warnings are spoken, and the phone vibrates for obstacles where the browser can (Android Chrome). **iPhone browsers cannot vibrate**, so the page beeps instead (the "Beep on warnings" box is on by default when vibration is not available; "Test alert" shows which one you have). Say a command such as "what is ahead", "take me to the chair", "who is Ahmed", "remember this person as Ahmed, my brother" or "calibrate". Say "stop", or hold the black screen for 2 seconds.
5. If the connection drops it says so and reconnects.

Notes:
- A phone run first looks for the saved map and continues it if it recognises the place (`--phone-new` always starts fresh). Phone maps and webcam maps are separate.
- The phone sends fewer pictures per second than the webcam, and walking shakes the camera, so tracking is harder than at a desk. Walk slowly.
- The page keeps the screen on, but if the phone locks or Chrome is put in the background the camera stops.
- Chrome's speech recognition listens through the phone microphone and needs an internet connection.
- Use it on a network you trust, because anyone with the link and key can see the camera data.

---

### 3.6 Learning mode: setting the scale with the phone's motion sensors
A single camera cannot see real size, and a blind user cannot use a tape, so at the start the phone page runs a learning mode
(tick or untick "Learning mode at start"; type your height in cm for a better result).

1. Tap **Start**. An iPhone asks for permission to use motion sensors here; Android Chrome does not.
2. Hold the phone in front of you, camera forward. When tracking starts it says "Learning mode. Walk 8 steps, then stop."
3. Walk about 8 steps at a normal pace, then stand still. The accelerometer counts the steps (`motion.py`) and the camera measures how
   far it moved in map units. Scale = map units walked / (steps x step length). Step length is 0.415 x your height if you typed it,
   otherwise 0.70 m.
4. It says "Scale saved." (the details are shown on the page) or "Learning failed. Say calibrate to try again." Then it asks you to put the
   phone in your pocket. Say "calibrate" at any time to do it again.

Rules and limits:
- Nothing is saved if tracking was lost during the walk, a new map was started, fewer than 4 steps were seen, the rhythm was uneven, the pace
  was outside 1.1 to 3 steps a second (steps were probably missed), or the result is outside 0.5 to 200 map units per meter.
- A session that already has a measured scale (from `set_scale.py`, or a joined saved map) is left alone.
- **The scale is only as good as the step length.** In simulated walks the error was under 5% when the step length was right. A stride that
  is really 0.60 m or 0.85 m instead of 0.70 m gives an error about equal to the stride error (roughly -18% to +16%). That is still much
  better than the default guess (5 map units per meter), which was off by 2 times or more in the pilot, but it is not a tape measure.
- **Not yet tested on a real phone.** The step counting and scale were tested on simulated accelerometer and camera data, and the page logic
  against a stand-in browser. It still needs a real walk, compared with a tape, in the real room.
- The page also shows the round trip time of a camera frame (phone to computer and back), which is the system's own delay and the first part
  of the warning latency. It does not include the time the camera needs to see the object.

## 4. Files

| File | What it does |
|---|---|
| `tracker.py` | Shared code: visual odometry, saved maps, database, sessions, position file. |
| `live_run.py` | Tracking, mapping, detection, people recognition and warnings together, saving as it runs. |
| `guidance.py` | Matching, directions and obstacle warnings. |
| `sound.py` | Speech output and voice input. |
| `people.py` | People memory: names, who they are, face samples, sightings and spoken commands. |
| `faces.py` | Face detection and recognition (downloads two small models once). |
| `navigate.py` | Directions to an object, optionally spoken. |
| `voice.py` | Ask by typing or speaking, hear the answers. |
| `api.py` | Dashboard, JSON API and the phone camera pipeline. |
| `set_scale.py` | Stores units-per-meter for the live session from a measured move. |
| `check_accuracy.py` | Compares a measured distance with a tape measurement. |
| `distance_diff.py` | Distance between two stored detections. |
| `view_memory.py` | Prints stored detections (newest session by default, with filters). |
| `visualize_pointcloud.py` | Density plot of one session's point cloud. |
| `detect.py` | Single-image detection, tagged with session and camera position. |
| `odometry_test.py` | Tracking only, with the overlay (wrapper around `tracker.py`). |
| `triangulate_map.py` | Tracking plus map points (wrapper around `tracker.py`). |
| `calibrate.py` | Computes `camera_calibration.npz` from checkerboard photos. |
| `selftest.py` | Checks the tracker and the logic on a simulated room. No webcam needed. |
| `motion.py` | Learning mode: counts steps from the phone's accelerometer and sets the scale (section 3.6). |
| `selftest_phone.py` | Tests speech safety, limited speech, the tracking-lost behaviour, learning mode, the phone server and the phone page logic. |
| `page_test.js` | The phone page logic test that `selftest_phone.py` runs with Node (optional). |
| `requirements.txt` | Python dependencies. |

### Data files (created while running)
- `spatial_memory.db`: SQLite database. Tables are created and upgraded automatically.
  - `detections`: `label`, `confidence`, `image_source`, `timestamp`, `x`, `y`, `z`, `session_id`, `pos_source`.
  - `point_cloud`: `x`, `y`, `z`, `timestamp`, `session_id`.
  - `sessions`: `id`, `started_at`, `units_per_meter`.
  - `people`, `face_samples`, `people_seen`, `enroll_requests`: who is remembered, their face samples, where each was last seen per session, and pending requests to learn a face.
  - `keyframes`: the saved views that let a later run recognise a place (`session_id`, keypoints, descriptors, 3D points, pose). Each keyframe takes roughly 100 KB, so a long scan adds a few MB to the database.
- `current_position.json`: written on every frame with `session_id`, `x`, `y`, `z`, `rotation`, `status` and `timestamp`.
- `live_scene.json`: what the camera saw in the last detection pass (people included), used by `what is ahead`.
- `pointcloud_filtered.png`: output of `visualize_pointcloud.py`.
- The phone mode keeps its self-made certificate, and face recognition its two models, in `.spatial_navigation` in your user folder, outside the repo.

If `spatial_memory.db` is tracked by git, every commit stores a new copy of it. Keep it out of git (`git rm --cached spatial_memory.db` and add it to `.gitignore`) unless you want the database in the repo.

---

## 5. Testing

```powershell
python selftest.py
python selftest_phone.py
```
`selftest_phone.py` takes about 30 seconds and needs no camera. Its server and page checks are skipped, with a note, if `fastapi`/`httpx` or Node are not installed.

It takes a few minutes. It builds a synthetic room with a virtual camera that turns and swings, runs the tracker, and checks the result against the truth. It passes when the final status is `tracking`, path error is under 6% of path length, heading error is under 8 degrees and the median map error is under 0.4 units. It then checks the logic: saved keyframes are loaded and recognised in a new run, a different room is rejected, short losses recover without a new session, a long loss starts a new world frame, objects are placed on the object and not the wall behind it, repeat sightings merge into one row, directions and obstacle warnings are right, people can be remembered, recognised (with stand-in faces) and forgotten, and `what is ahead` lists the right things. It uses temporary files, so your database and position file are never touched. It uses random noise, so a run that misses a limit by a hair can happen. Run it again before assuming something is broken.

### Results so far (pilot, one developer, one room, webcam plus one iPhone)

Anything not listed here has **not been measured**. Scale values from different sessions cannot be compared with each other. All counts were taken by one person, so treat them as pilot numbers, not proof.

| Test | Result | Status |
|---|---|---|
| `selftest.py` (simulated room) | Passed twice earlier. Turn heading error 2.0 degrees, swing 1.4 degrees. Run again before each release. | Measured |
| Scale repeatability (0.5 m move, 3 runs per session) | Session 42: 3.775, 4.413, 4.182 (spread about 17%, pass). Session 43: 13.294, 10.097 (about 30% apart, fail). Session 62: 0.197, 0.289, 0.146 (the script flagged them as unusual, about 2 times apart, fail). Session 41: 4.363, 4.620 (a third attempt was discarded). | Measured. Passed in 1 session, failed in 2 |
| Early scale attempt at 0.1 m (session 39) | 10.971, 3.633, 23.471. The move was too short and the camera was rotated, so this method is not valid. | Measured, invalid method |
| Accuracy, first attempt (session 43) | Tape 0.4, 0.8, 1.2 m. System said 0.20, 0.20, 0.38 m. Errors 0.20, 0.60, 0.82 m. The scale in that session was not repeatable and the distances were short, so this is not conclusive. | Inconclusive |
| Accuracy, redo (session 62) | System distances 10.40, 8.77 and 2.85 m for laptop to chair, chair to phone and laptop to phone. Tape values entered: 0.3 m (first run) and 0.5 m (second run) for each pair. Errors 2.35 to 10.10 m, all failed. The scale in that session was 0.146 units per meter (flagged as unusual, and the three runs ranged from 0.146 to 0.289), so this is a scale failure, not a measure of tracker accuracy. | Failed, scale unreliable |
| Saved-map join | Found the saved map in 3 of 3 tries (session 51). The first map was small (283 points) and grew each try (3, 4, then 13 views). An earlier round gave 1 of 3. | Measured |
| Identical rooms |  | **Not measured** |
| Faces | Enrolled person named correctly 10 of 10. Strangers given a name 0 of 10. One enrolled person, normal indoor light, counted from the camera window. | Measured, small sample |
| Phone: tracking and voice | Connected and tracked (session 60: 16,253 map points). Voice commands reached the server and spoken output was heard. "What is head" was misheard again. | Measured |
| Phone: vibration | Not supported by iOS browsers, and only an iPhone was available. To be checked on Android. | Not testable on iPhone |
| Phone: pocket walk |  | **Not measured** |
| Obstacle warnings |  | **Not measured** |
| Tracking stability | Session 60 (phone): 3 lost events. Session 62 (laptop, handheld): 13 lost events. Tracking recovered each time without starting a new session. | Measured, weak |

**Planned tests and pass bars:**

1. **Scale repeatability:** 3 runs of `set_scale.py` in one session with 0.594 m (2 A4 sheets). All three within 20% of each other.
2. **Accuracy:** 6 object pairs over 2 sessions, objects at the same height, measured with `check_accuracy.py`. Every error under 1.0 m.
3. **Saved-map join:** 3 tries, at least 2 must print "Found the saved map".
4. **Identical rooms:** a run in room A2 must **not** join the saved map of room A1 (zero false joins), and `--map N` must join the right room. Repeat with pair B.
5. **Faces (with the friends' permission):** at least 8 of 10 correct in normal light and zero names given to a stranger.
6. **Phone:** spoken output (voice box on, media volume full), vibration near an obstacle, a 3 minute slow walk with the phone in a pocket, and a count of lost events.

A real tape check: use A4 sheets, long side 29.7 cm (1 sheet = 0.297 m, 2 = 0.594 m, 3 = 0.891 m, 5 = 1.485 m). Run `python check_accuracy.py ID1 ID2 TAPE_METERS` on two objects at the same height.

---

## 6. Known limitations

- **Accuracy has not been reliably measured.** One attempt was inconclusive and the redo failed because the scale in that session was unreliable. About 1 m is the pilot target, not a result. A single camera's depth is rough, and farther objects have noisier positions. A depth-capable device (for example an iPhone with LiDAR) would improve this at the source.
- **Drift.** There is no loop closure or bundle adjustment. Recognising a saved view corrects the position, but heading and scale can still drift slowly during long scans.
- **Fast motion and blur** cause `lost`. Move slowly, and use a well-lit room with texture.
- **A map is recognised only if the place looks similar.** Moving furniture or changing light can stop a later run from joining the old map. It then starts a new session.
- **Different cameras do not share maps.** A map made with the webcam is not used by the phone, and the reverse.
- **Object positions are approximate**, and objects of the same label within about 0.75 m are merged into one.
- **Obstacle warnings only cover objects the camera has already seen**, and only the obstacle-sized classes. They are advisory, not a safety system.
- **Detection is only as good as YOLOv8n.** Wrong labels still happen, even at confidence 0.6 (for example bed, dog or couch), and one object can get several labels.
- **Vibration alerts do not work on iPhone.** iOS browsers do not support web vibration, so the phone page beeps instead. Vibration is not yet tested on Android.
- **Saved-map joins vary between runs** (1 of 3 in an earlier round, 3 of 3 in the latest). See the results table for look-alike rooms.
- **Scale is only rough without a measurement.** A blind user cannot use a tape, so a helper should run `set_scale.py` first. Without it the default (5 map units per meter) can be up to about 2 times off, which moves the 1 m warning distance.
- **Walking and a pocket are hard for the tracker:** shaking, a low viewpoint and clothing that may cover the lens cause `lost`. Warnings stop while tracking is lost, and the phone says so.
- **Learning-mode scale is about as accurate as the step length** (roughly 15 to 25% off for an unusual stride) and has only been tested on simulated walks. A tape measure is still better.
- **Face recognition can be wrong or miss**, especially in poor light, side views or from far away.
- **Legacy rows** (the first 17 detections) have placeholder positions. They are hidden unless you use `--all`.

## 7. Troubleshooting

| Problem | What to do |
|---|---|
| `live_run.py : The term ... is not recognized` | Start commands with `python`, and run them from the project folder. |
| `set_scale.py` says the position is not updating | `live_run.py` is not running. Keep it running in another terminal. |
| `set_scale.py` says the tracking session changed | The tracker lost its place while you moved. Move more smoothly and try again. |
| Status stays `lost` | Move the camera back to something you scanned earlier and wait for `tracking`. |
| Picture is green, black or frozen | Close other apps using the camera, unplug and replug it, or run `python live_run.py --dshow`. |
| `navigate.py` says the object is only from an earlier session | Scan the object again in the current run, or continue the map that has it with `--map N`. |
| Objects are not being saved | They need at least 3 tracked map points on them. Pause on each object and move slowly. |
| The saved map is not recognised | Aim at a part of the room you scanned before, with the same furniture and light. Otherwise it starts a new map. |
| No sound | The computer has no speech engine, or the volume is off. Messages are still printed. On Windows, check Settings > Speech. |
| `Face recognition is not available` | The two face models could not be downloaded. Check the internet connection, or download them by hand (see the top of `faces.py`). Names can still be remembered. |
| The phone page does not speak or listen | Use Chrome on Android, allow the microphone, and tap Start camera once (browsers need a tap before speaking). |
| `voice.py --listen` hears nothing | Check the microphone and Windows speech recognition for your language, or type your questions. |
| Phone page says the camera needs https | Use the `https://` address printed by `python api.py --phone`, and accept the certificate warning. |
| `api.py` says it needs a package | Run `pip install fastapi "uvicorn[standard]" cryptography`. |

## 8. Roadmap

- Test learning mode on a real phone against a tape, check the beep on iPhone and vibration on Android, and measure the warning latency and the "person ahead" warning.
- Measure accuracy against a tape in a real room and report it here.
- Test the join across two real runs and the phone pipeline on real devices.
- A scale that needs no tape measure (for example from the phone's motion sensors or known object sizes), so a blind user can use it alone.
- Walking tests with a phone in a pocket, and tuning of the tracker for that.
- A depth-capable device (iPhone with LiDAR) for better accuracy.
- Anything beyond the phone is for the client to build.

## Privacy
Face samples are only of friends who agreed, for pilot testing. They stay in the local database (never in git). When the pilot ends, the data is archived locally. `forget NAME` removes a person's name and face. The repo ignores `.env`, databases, images, model files, calibration images and live position files (see `.gitignore`), so keys, photos and face data are never committed.
