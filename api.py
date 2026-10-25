"""Web side of the system: API, dashboard with voice, and the phone camera pipeline.

    pip install fastapi "uvicorn[standard]"                 (once)
    python api.py                  dashboard on this computer: http://127.0.0.1:8000
    python api.py --phone          also accept a phone camera over your Wi-Fi (needs: pip install cryptography)

Dashboard: live status, stored objects, and a box to ask "take me to the laptop". Answers are
shown and spoken by the browser, and the microphone button lets you ask by voice (Chrome/Edge/Safari).

Phone mode: run `python api.py --phone`, open the https address it prints on your phone (same
Wi-Fi), accept the certificate warning, allow the camera and press Start. The phone streams its
camera to this computer, which runs the same tracking, detection and saving as live_run.py.
Use it on a network you trust: anyone with the link and key can see the camera data.

Endpoints:
    GET /status      latest position, rotation and tracking status
    GET /sessions    every session with its scale and row counts
    GET /detections  objects of the newest session (?session=17, ?label=phone, ?all=true)
    GET /pointcloud  map points of the newest session (?session=17&limit=5000)
    GET /navigate    answer a text query (?query=take me to the laptop)
    GET /events      state and messages of the phone pipeline

Learning mode (phone): at the start the phone page asks you to walk about eight steps. The phone's
motion sensors count the steps and the camera measures how far it moved, which sets the map scale
without a tape measure (motion.py). Say "calibrate" later to do it again.
"""
import argparse
import collections
import os
import queue
import secrets
import socket
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

import guidance
import motion
from tracker import connect, get_units_per_meter, read_position

PHONE_FOCAL_FACTOR = 0.8   # focal length ~ 0.8 x the longer image side (typical phone main camera)
CERT_DIR = os.path.join(os.path.expanduser('~'), '.spatial_navigation')

app = FastAPI(title='Spatial Navigation')
TOKEN = None               # set in phone mode: pages and the camera socket need ?key=...


# ------------------------------------------------------------------ data endpoints
def _newest_session(conn):
    row = conn.execute('SELECT MAX(id) FROM sessions').fetchone()
    return row[0] if row and row[0] else None


@app.middleware('http')
async def guard(request, call_next):
    if not TOKEN:
        return await call_next(request)
    given = request.query_params.get('key') or request.cookies.get('key')
    if given != TOKEN:
        return PlainTextResponse('Open the link printed in the computer window (it includes ?key=...).',
                                 status_code=403)
    response = await call_next(request)
    if request.query_params.get('key') == TOKEN:
        response.set_cookie('key', TOKEN, httponly=True, samesite='lax')
    return response


@app.get('/status')
def status():
    pos = read_position()
    if pos is None:
        raise HTTPException(404, 'No position yet. Start live_run.py first.')
    return pos


@app.get('/sessions')
def sessions():
    conn = connect()
    rows = conn.execute(
        'SELECT s.id, s.started_at, s.units_per_meter, '
        '(SELECT COUNT(*) FROM detections d WHERE d.session_id = s.id), '
        '(SELECT COUNT(*) FROM point_cloud p WHERE p.session_id = s.id), '
        '(SELECT COUNT(*) FROM keyframes k WHERE k.session_id = s.id) '
        'FROM sessions s ORDER BY s.id').fetchall()
    conn.close()
    return [{'session_id': r[0], 'started_at': r[1], 'units_per_meter': r[2],
             'detections': r[3], 'map_points': r[4], 'saved_views': r[5]} for r in rows]


@app.get('/detections')
def detections(session: int | None = None, label: str | None = None, all: bool = False,
               limit: int = Query(200, ge=1, le=5000)):
    conn = connect()
    query = ('SELECT id, label, confidence, x, y, z, pos_source, session_id, timestamp '
             'FROM detections')
    where, args = [], []
    if not all:
        session = session if session is not None else (read_position() or {}).get('session_id') or _newest_session(conn)
        where.append('session_id = ?')
        args.append(session)
    if label:
        where.append('label LIKE ?')
        args.append(f'%{label}%')
    if where:
        query += ' WHERE ' + ' AND '.join(where)
    rows = conn.execute(query + ' ORDER BY id LIMIT ?', args + [limit]).fetchall()
    out = []
    for id_, lab, conf, x, y, z, source, sid, ts in rows:
        scale = get_units_per_meter(conn, sid) if sid else None
        item = {'id': id_, 'label': lab, 'confidence': conf, 'x': x, 'y': y, 'z': z,
                'pos_source': source, 'session_id': sid, 'timestamp': ts,
                'legacy': not sid, 'units_per_meter': scale}
        if scale:
            item['meters'] = {'x': x / scale, 'y': y / scale, 'z': z / scale}
        out.append(item)
    conn.close()
    return out


@app.get('/pointcloud')
def pointcloud(session: int | None = None, limit: int = Query(5000, ge=1, le=100000)):
    conn = connect()
    session = session if session is not None else _newest_session(conn)
    rows = conn.execute('SELECT x, y, z FROM point_cloud WHERE session_id = ? ORDER BY id LIMIT ?',
                        (session, limit)).fetchall()
    scale = get_units_per_meter(conn, session) if session else None
    conn.close()
    return {'session_id': session, 'units_per_meter': scale, 'count': len(rows),
            'points': [list(r) for r in rows]}


@app.get('/navigate')
def navigate(query: str = Query(..., min_length=1, max_length=200)):
    """Directions with obstacle warnings (same logic as navigate.py and voice.py)."""
    answer = guidance.describe(query)
    return {'query': query, **answer}


# --------------------------------------------------------------- phone camera pipeline
class FrameQueueSource:
    """Frames from the phone, shaped like a camera so live_run.py can read them."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.closed = False

    def put(self, frame):
        while True:
            try:
                self.queue.put_nowait(frame)
                return
            except queue.Full:                    # drop the oldest frame, keep the newest
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def read(self):
        while not self.closed:
            try:
                return True, self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
        return False, None

    def release(self):
        self.closed = True


def _measured_scale(session_id):
    conn = connect()
    try:
        return get_units_per_meter(conn, session_id)
    finally:
        conn.close()


def _store_scale(session_id, units_per_meter):
    conn = connect()
    try:
        conn.execute('UPDATE sessions SET units_per_meter = ? WHERE id = ?', (units_per_meter, session_id))
        conn.commit()
    finally:
        conn.close()


class PhonePipeline:
    """One live_run.py session fed by a phone, running in a background thread."""

    def __init__(self):
        self.source = None
        self.thread = None
        self.state = {'status': 'idle'}
        self.events = collections.deque(maxlen=50)
        self.next_event = 1
        self._event_lock = threading.Lock()
        self.error = None
        self.resume = 'auto'       # continue the latest saved map if the phone recognises the place
        self.learner = motion.LearningController(self.notify, read_position, _measured_scale, _store_scale)

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def notify(self, text, speak=True, learn=None):
        """A short message for the phone. speak=False: show it but do not say it (limited speech).
        learn: 'done', 'failed' or 'skipped' on the message that ends learning mode."""
        with self._event_lock:
            event = {'text': text, 'speak': bool(speak)}
            if learn:
                event['learn'] = learn
            self.events.append((self.next_event, event))
            self.next_event += 1

    def events_since(self, last_id):
        items = [(i, e) for i, e in list(self.events) if i > last_id]
        return [e for _, e in items], (items[-1][0] if items else last_id)

    def handle_text(self, raw):
        """A JSON text message from the phone: motion samples or a learning-mode command."""
        msg = motion.parse_message(raw)
        if msg is None:
            return
        kind = msg.get('type')
        if kind == 'motion':
            samples = msg.get('s')
            if isinstance(samples, list):
                self.learner.on_motion(samples[:200])
        elif kind == 'learn':
            action = msg.get('action')
            if action == 'start':
                if self.state.get('searching'):          # the session can still change when a saved map is found
                    self.notify('Not ready. Still looking for the saved map.', True, 'failed')
                    return
                step = msg.get('step_m')
                self.learner.start(step if isinstance(step, (int, float)) and not isinstance(step, bool) else None)
            elif action == 'cancel':
                self.learner.cancel()
        elif kind == 'motion_unavailable':
            self.learner.on_unavailable()

    def start(self, first_frame):
        import live_run                                     # heavy imports only when needed
        h, w = first_frame.shape[:2]
        f = PHONE_FOCAL_FACTOR * max(w, h)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        self.source = FrameQueueSource()
        self.state = {'status': 'starting'}
        self.error = None
        self.learner.reset()

        def target():
            try:
                live_run.run(save_map=True, detect=True, source=self.source, show=False, K=K,
                             state=self.state, notify=self.notify, resume=self.resume)
            except Exception as exc:                        # show the reason on the phone
                self.error = f'{type(exc).__name__}: {exc}'
                self.state['status'] = 'error'
                print('Phone pipeline stopped:', self.error)

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self):
        if self.source is not None:
            self.source.release()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.state = {'status': 'idle'}
        self.learner.reset()


PIPE = PhonePipeline()


@app.get('/events')
def events(since: int = 0):
    texts, last = PIPE.events_since(since)
    return {'state': PIPE.state, 'events': texts, 'last': last, 'error': PIPE.error}


@app.websocket('/ws/camera')
async def camera(ws: WebSocket):
    import asyncio
    await ws.accept()
    if TOKEN and ws.query_params.get('key') != TOKEN and ws.cookies.get('key') != TOKEN:
        await ws.close(code=1008)
        return
    if PIPE.running and PIPE.source is not None and not PIPE.source.closed:
        await ws.send_json({'error': 'Another phone is already connected.'})
        await ws.close()
        return
    last_event = 0
    try:
        while True:
            message = await ws.receive()
            if message.get('type') == 'websocket.disconnect':
                break
            if message.get('text') is not None:             # motion samples or a learning-mode command
                PIPE.handle_text(message['text'])
                continue
            data = message.get('bytes')
            if data is None:
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if not PIPE.running:
                PIPE.start(frame)
            PIPE.source.put(frame)
            for _ in range(20):                             # wait until processing has caught up
                if PIPE.source.queue.qsize() == 0:
                    break
                await asyncio.sleep(0.05)
            PIPE.learner.tick()
            texts, last_event = PIPE.events_since(last_event)
            await ws.send_json({'state': PIPE.state, 'events': texts, 'error': PIPE.error})
    except WebSocketDisconnect:
        pass
    finally:
        PIPE.stop()


# ------------------------------------------------------------------------- the page
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial Navigation</title>
<style>
body{font-family:system-ui,sans-serif;margin:0 auto;padding:12px;max-width:720px;background:#14161a;color:#eaeaea}
h2,h3{margin:.4em 0} button{font-size:1.1rem;padding:.7em 1em;margin:.2em .2em .2em 0;border-radius:8px;border:0;background:#2f6fed;color:#fff}
button.off{background:#444} input[type=text]{font-size:1rem;padding:.5em;width:60%;border-radius:8px;border:1px solid #555;background:#1f2228;color:#eee}
#status{font-size:1.1rem;margin:.6em 0;padding:.6em;border-radius:8px;background:#1f2228}
.tracking{color:#4ade80}.lost{color:#f87171}.initializing,.starting{color:#facc15}
pre{white-space:pre-wrap;background:#1f2228;padding:.6em;border-radius:8px;min-height:3em}
table{width:100%;border-collapse:collapse}td,th{padding:.25em .4em;border-bottom:1px solid #333;text-align:left;font-size:.9rem}
video{width:100%;border-radius:8px;background:#000}
#pocket{display:none;position:fixed;inset:0;background:#000;color:#777;align-items:center;justify-content:center;text-align:center;z-index:10;font-size:1.1rem;padding:1em;user-select:none;-webkit-user-select:none}
.note{font-size:.85rem;color:#aaa}
</style></head><body>
<h2>Spatial Navigation</h2>
<div id="cam" style="display:none">
  <video id="v" playsinline muted></video><canvas id="c" hidden></canvas>
  <div><button id="start">Start camera (pocket mode)</button><button id="stop" class="off">Stop</button></div>
  <div class="note">Pilot test, not a safety device. Starting: with learning mode on, hold the phone in front of you and walk about
  eight steps so the phone can set the map scale from its motion sensors (no tape needed). Then the phone speaks, waits 10 seconds so
  you can put it in your front pocket with the camera facing forward, and streams to the computer. Keep this page open; the screen
  stays on. Say a command (say "calibrate" to learn the scale again), or hold the black screen for 2 seconds to stop.</div>
  <div class="note">
    <label><input type="checkbox" id="learnmode" checked> Learning mode at start</label>
    <label>Your height in cm (optional) <input type="text" id="height" inputmode="numeric" style="width:4em"></label>
    <label><input type="checkbox" id="beep"> Beep on warnings</label>
    <button id="alerttest" class="off">Test alert</button> <span id="rtt"></span>
  </div>
  <div id="lastmsg" class="note"></div>
  <div id="warn" class="lost"></div>
</div>
<div id="status">Waiting for data...</div>
<div>
  <input type="text" id="q" placeholder="take me to the laptop">
  <button id="ask">Ask</button><button id="mic">Speak</button>
  <label><input type="checkbox" id="voice" checked> Speak answers</label>
</div>
<pre id="answer"></pre>
<div class="note">Try: "what is ahead", "take me to the chair", "who is Ahmed", "remember this person as Ahmed, my brother",
"remember that Sara is my friend", "who do you know", "forget Ahmed".</div>
<h3>Objects in this session</h3>
<table id="objs"><tr><th>Object</th><th>Where (m, or map units)</th><th>Conf</th></tr></table>
<div id="pocket">Pocket mode is on.<br>Say a command, for example "what is ahead".<br>Hold here for 2 seconds to stop.</div>
<script>
const CAMERA = __CAMERA__;
const $ = id => document.getElementById(id);
const COMMAND = /\b(where|take me|guide|find|who|what|remember|learn|learning mode|calibrat\w*|forget|list|ahead|front|next|clear|anyone|anything|this person|stop)\b/i;
let pocket = false, speaking = false, recog = null, wakeLock = null, ws = null, stream = null, running = false, stopping = false, lastSent = 0;

let audioCtx = null, trackStatus = 'idle', trackSearching = false, learnActive = false, learnState = '', motionAvail = false, motionOn = false;
let motionBuf = [], sentAt = 0, rttAvg = 0, rttMax = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function waitFor(test, ms){
  const end = performance.now() + ms;
  while(performance.now() < end){ if(test()) return true; await sleep(200); }
  return test();
}
function ensureAudio(){
  try{ audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); if(audioCtx.state === 'suspended') audioCtx.resume(); }catch(e){}
}
function beep(){
  ensureAudio(); if(!audioCtx) return;
  [0, 0.25].forEach(d => {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain(); o.frequency.value = 880; g.gain.value = 0.25;
    o.connect(g); g.connect(audioCtx.destination); o.start(audioCtx.currentTime + d); o.stop(audioCtx.currentTime + d + 0.15);
  });
}
function alertNow(){
  if(navigator.vibrate){ navigator.vibrate([200, 100, 200]); }
  if($('beep').checked) beep();
}
function handleEvent(ev){
  const t = (typeof ev === 'string') ? ev : ev.text;
  if(ev.learn){ learnState = ev.learn; }
  $('lastmsg').textContent = t;
  if(ev.speak === false) return;                       // limited speech: shown, not spoken
  say(t, /ahead|obstacle|lost/i.test(t));
}
function onMotion(e){
  const a = e.accelerationIncludingGravity;
  if(!a || a.x === null || a.x === undefined) return;
  motionAvail = true;
  if(!learnActive) return;
  motionBuf.push([Math.round(performance.now() * 10) / 10, +a.x.toFixed(3), +a.y.toFixed(3), +a.z.toFixed(3)]);
  if(motionBuf.length >= 10) flushMotion();
}
function flushMotion(){
  if(ws && ws.readyState === 1 && motionBuf.length){ ws.send(JSON.stringify({type:'motion', s: motionBuf})); }
  motionBuf = [];
}
async function enableMotion(){
  try{
    if(typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function'){
      const r = await DeviceMotionEvent.requestPermission();      // iPhone: must be asked inside the tap
      if(r !== 'granted') return false;
    }
    if(!motionOn){ window.addEventListener('devicemotion', onMotion); motionOn = true; }
    await waitFor(() => motionAvail, 1500);
    return motionAvail;
  }catch(e){ return false; }
}
function stepLength(){
  const h = parseFloat($('height').value);
  return (h > 100 && h < 230) ? +(0.415 * h / 100).toFixed(3) : null;      // null: the computer uses its default
}
async function startLearning(){
  if(!ws || ws.readyState !== 1){ say('The camera is not running.'); return; }
  if(!motionAvail && !(await enableMotion())){ ws.send(JSON.stringify({type:'motion_unavailable'})); return; }
  learnState = 'running'; learnActive = true; motionBuf = [];
  ws.send(JSON.stringify({type:'learn', action:'start', step_m: stepLength()}));
  await waitFor(() => learnState !== 'running', 50000);
  learnActive = false; flushMotion();
  if(learnState === 'running' && ws && ws.readyState === 1){ ws.send(JSON.stringify({type:'learn', action:'cancel'})); }
}

function say(text, urgent){
  if(urgent) alertNow();
  if(!$('voice').checked || !('speechSynthesis' in window)) return;
  if(urgent) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.onstart = () => { speaking = true; stopListening(); };
  u.onend = u.onerror = () => { speaking = false; if(pocket) setTimeout(startListening, 300); };
  speechSynthesis.speak(u);
}
function startListening(){
  if(!pocket || speaking || recog) return;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR) return;
  recog = new SR(); recog.lang = 'en-US'; recog.continuous = true; recog.interimResults = false;
  recog.onresult = e => {
    for(let i = e.resultIndex; i < e.results.length; i++){
      if(e.results[i].isFinal){ const t = e.results[i][0].transcript.trim(); if(COMMAND.test(t)) ask(t); }
    }
  };
  recog.onend = () => { recog = null; if(pocket && !speaking) setTimeout(startListening, 300); };
  recog.onerror = () => {};
  try{ recog.start(); }catch(e){ recog = null; }
}
function stopListening(){ if(recog){ const r = recog; recog = null; try{ r.abort(); }catch(e){} } }

function showState(s){
  const st = s.status || 'idle'; trackStatus = st; trackSearching = !!s.searching;
  $('status').innerHTML = 'Tracking: <b class="'+st+'">'+st+'</b>'
    + (s.x!==undefined ? ' &nbsp; X '+s.x.toFixed(1)+' Y '+s.y.toFixed(1)+' Z '+s.z.toFixed(1) : '')
    + (s.points!==undefined ? ' &nbsp; points '+s.points : '')
    + (s.searching ? ' &nbsp; (looking for saved map)' : '');
}
async function poll(){
  try{
    if(!CAMERA || !ws){ const r = await fetch('/status'); if(r.ok){ const p = await r.json(); showState({status:p.status,x:p.x,y:p.y,z:p.z}); } }
    const d = await (await fetch('/detections?limit=100')).json();
    const rows = d.map(o => '<tr><td>'+o.label+'</td><td>'+(o.meters ? o.meters.x.toFixed(2)+', '+o.meters.y.toFixed(2)+', '+o.meters.z.toFixed(2)+' m' : o.x.toFixed(1)+', '+o.y.toFixed(1)+', '+o.z.toFixed(1))+'</td><td>'+o.confidence.toFixed(2)+'</td></tr>').join('');
    $('objs').innerHTML = '<tr><th>Object</th><th>Where (m, or map units)</th><th>Conf</th></tr>' + rows;
  }catch(e){}
}
setInterval(poll, 1500); poll();

async function ask(text){
  text = (text || $('q').value).trim(); if(!text) return;
  $('q').value = text;
  if(/^(stop|stop listening|goodbye|stop camera)\W*$/i.test(text)){ say('Stopping.'); stopAll(); return; }
  if(/\b(calibrat\w*|learning mode)\b/i.test(text)){ startLearning(); return; }
  try{
    const a = await (await fetch('/navigate?query='+encodeURIComponent(text))).json();
    $('answer').textContent = a.text; say(a.spoken, /warning|ahead/i.test(a.spoken) && /obstacle/i.test(a.spoken));
  }catch(e){ say('I could not reach the computer.'); }
}
$('ask').onclick = () => ask();
$('q').addEventListener('keydown', e => { if(e.key==='Enter') ask(); });
$('mic').onclick = () => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){ $('answer').textContent = 'This browser has no speech recognition. Try Chrome, Edge or Safari, or type.'; return; }
  const r = new SR(); r.lang = 'en-US'; r.onresult = e => ask(e.results[0][0].transcript);
  r.onerror = e => { $('answer').textContent = 'Could not hear you ('+e.error+').'; }; r.start();
};

if(CAMERA){
  $('cam').style.display = 'block';
  if(!window.isSecureContext) $('warn').textContent = 'The camera needs the https:// address. Use the link printed on the computer.';
  function sendFrame(){
    if(!running || !ws || ws.readyState !== 1) return;
    const v = $('v'), c = $('c'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob(b => { if(b && ws && ws.readyState === 1){ sentAt = performance.now(); ws.send(b); flushMotion(); } }, 'image/jpeg', 0.7);
  }
  function connect(){
    ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/camera');
    ws.onopen = sendFrame;
    ws.onmessage = e => {
      const m = JSON.parse(e.data);
      if(m.error){ $('warn').textContent = m.error; }
      if(m.state) showState(m.state);
      if(sentAt){
        const r = performance.now() - sentAt; rttAvg = rttAvg ? 0.8 * rttAvg + 0.2 * r : r; rttMax = Math.max(rttMax, r); sentAt = 0;
        $('rtt').textContent = 'round trip ' + Math.round(rttAvg) + ' ms (max ' + Math.round(rttMax) + ')';
      }
      (m.events || []).forEach(handleEvent);
      const wait = Math.max(0, 90 - (performance.now() - lastSent)); lastSent = performance.now() + wait;
      setTimeout(sendFrame, wait);
    };
    ws.onclose = () => { if(running && !stopping){ say('Connection lost. Trying again.'); setTimeout(connect, 2000); } };
  }
  async function enterPocket(){
    pocket = true; $('pocket').style.display = 'flex';
    try{ wakeLock = await navigator.wakeLock.request('screen'); }catch(e){}
    startListening();
  }
  function stopAll(){
    stopping = true; running = false; pocket = false; $('pocket').style.display = 'none';
    learnActive = false; motionBuf = [];
    if(motionOn){ window.removeEventListener('devicemotion', onMotion); motionOn = false; }
    stopListening(); if(ws) ws.close(); ws = null;
    if(stream) stream.getTracks().forEach(t => t.stop());
    if(wakeLock){ try{ wakeLock.release(); }catch(e){} wakeLock = null; }
  }
  $('beep').checked = !navigator.vibrate;
  $('alerttest').onclick = () => {
    ensureAudio();
    $('lastmsg').textContent = navigator.vibrate
      ? (navigator.vibrate([300]) ? 'Vibration sent.' : 'Vibration was blocked by the browser.')
      : 'Vibration is not supported by this browser (iPhone browsers cannot vibrate), so beeps are used.';
    alertNow();
  };
  $('start').onclick = async () => {
    stopping = false; ensureAudio();
    const wantLearn = $('learnmode').checked;
    const gotMotion = wantLearn ? await enableMotion() : false;       // first thing in the tap: iPhones ask permission here
    try{
      stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment', width:{ideal:640}, height:{ideal:480}}, audio:false});
    }catch(e){ $('warn').textContent = 'Camera not available: '+e.message; return; }
    $('v').srcObject = stream; await $('v').play();
    if(wantLearn && gotMotion){
      say('Pilot test. This is not a safety device. Hold the phone in front of you, camera forward. Learning mode starts in five seconds.', true);
      await sleep(5000);
      if(stopping) return;
      running = true; connect();
      const ready = await waitFor(() => trackStatus === 'tracking' && !trackSearching, 45000);   // not while it still looks for the saved map
      if(stopping) return;
      if(ready){ await startLearning(); } else { say('Tracking did not start, so learning mode is skipped.'); }
      if(stopping) return;
      say('Now put the phone in your front pocket with the camera facing forward. Starting in ten seconds.', true);
      await sleep(10000);
      if(stopping) return;
      enterPocket();
      say('Started. Walk slowly. Ask what is ahead at any time.');
    } else {
      if(wantLearn) say('Motion sensors are not available, so learning mode is off.');
      say('Pilot test. This is not a safety device. Put the phone in your front pocket with the camera facing forward. Starting in ten seconds.', true);
      await sleep(10000);
      if(stopping) return;
      running = true; connect(); enterPocket();
      say('Started. Walk slowly. Ask what is ahead at any time.');
    }
  };
  $('stop').onclick = stopAll;
  let holdTimer = null;
  $('pocket').addEventListener('touchstart', e => { e.preventDefault(); holdTimer = setTimeout(() => { say('Stopping.'); stopAll(); }, 2000); }, {passive:false});
  $('pocket').addEventListener('touchend', () => clearTimeout(holdTimer));
  $('pocket').addEventListener('contextmenu', e => e.preventDefault());
  document.addEventListener('visibilitychange', async () => {
    if(document.visibilityState === 'visible' && pocket && !wakeLock){ try{ wakeLock = await navigator.wakeLock.request('screen'); }catch(e){} }
  });
}
</script></body></html>"""


@app.get('/', response_class=HTMLResponse)
def dashboard():
    return PAGE.replace('__CAMERA__', 'false')


@app.get('/phone', response_class=HTMLResponse)
def phone_page():
    return PAGE.replace('__CAMERA__', 'true')


# ---------------------------------------------------------------------- start-up
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'


def make_certificate(ip):
    """Self-signed certificate (phones only allow the camera over https)."""
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    os.makedirs(CERT_DIR, exist_ok=True)
    cert_path, key_path = os.path.join(CERT_DIR, f'cert_{ip}.pem'), os.path.join(CERT_DIR, f'key_{ip}.pem')
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'spatial-navigation')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone', action='store_true', help='accept a phone camera over Wi-Fi (https)')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--phone-new', action='store_true',
                        help='phone runs always start a fresh map instead of looking for the saved one')
    args = parser.parse_args()
    if not args.phone:
        print('Dashboard: http://127.0.0.1:8000   (add --phone to use a phone camera)')
        uvicorn.run(app, host='127.0.0.1', port=args.port or 8000)
    else:
        try:
            ip = lan_ip()
            cert, key = make_certificate(ip)
        except ImportError:
            raise SystemExit('Phone mode needs one more package. Run: pip install cryptography')
        TOKEN = secrets.token_urlsafe(6)
        if args.phone_new:
            PIPE.resume = 'new'
        port = args.port or 8443
        print('\nOn your phone (same Wi-Fi), open this address, accept the certificate warning, '
              'allow the camera and press Start:')
        print(f'    https://{ip}:{port}/phone?key={TOKEN}')
        print(f'Dashboard on this computer: https://127.0.0.1:{port}/?key={TOKEN}')
        print('Anyone on your network with this link can see the camera data. Stop with Ctrl+C.\n')
        uvicorn.run(app, host='0.0.0.0', port=port, ssl_keyfile=key, ssl_certfile=cert)
