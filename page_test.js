const vm = require('vm'), fs = require('fs');
// Runs the phone page's JavaScript against a stand-in browser. Used by selftest_phone.py:
//   node page_test.js path/to/page.js
const code = fs.readFileSync(process.argv[2], 'utf8');
const SPEED = 50, t0 = Date.now();
function makeSandbox(opts) {
  const els = {}, spoken = [], sockets = [], beeps = [], vibrations = [], listeners = {};
  const mk = id => ({ id, style: {}, checked: (id === 'voice' || id === 'learnmode'), value: '', textContent: '', innerHTML: '', onclick: null,
    addEventListener() {}, getContext() { return { drawImage() {} }; }, srcObject: null, play: async () => {},
    videoWidth: 640, videoHeight: 480, toBlob(cb) { cb({ size: 1 }); } });
  const sb = {
    els, spoken, sockets, beeps, vibrations, listeners,
    document: { getElementById: id => els[id] || (els[id] = mk(id)), addEventListener() {}, visibilityState: 'visible' },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) }, wakeLock: { request: async () => ({ release() {} }) } },
    location: { protocol: 'https:', host: 'x' },
    performance: { now: () => (Date.now() - t0) * SPEED },
    setTimeout: (fn, ms) => setTimeout(fn, Math.max(1, ms / SPEED)),
    clearTimeout, setInterval: () => 0, console,
    fetch: async () => ({ ok: true, json: async () => [] }),
    speechSynthesis: { cancel() {}, speak(u) { spoken.push(u.text); if (u.onend) setTimeout(u.onend, 1); } },
    SpeechSynthesisUtterance: class { constructor(t) { this.text = t; } },
    DeviceMotionEvent: opts.iosPermission ? { requestPermission: async () => opts.iosPermission } : (opts.noMotion ? undefined : {}),
    AudioContext: class {
      constructor() { this.currentTime = 0; this.state = 'running'; this.destination = {}; }
      createOscillator() { const o = { frequency: {}, connect() {}, start() { beeps.push(1); }, stop() {} }; return o; }
      createGain() { return { gain: {}, connect() {} }; }
    },
    WebSocket: class { constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); setTimeout(() => this.onopen && this.onopen(), 1); }
      send(x) { this.sent.push(x); } close() { this.readyState = 3; } },
    window: null,
  };
  if (opts.vibrate) sb.navigator.vibrate = p => { vibrations.push(p); return true; };
  sb.window = { listeners, isSecureContext: true, addEventListener(type, fn) { listeners[type] = fn; if (type === 'devicemotion' && !opts.noMotionEvents) setTimeout(() => fn({ accelerationIncludingGravity: { x: 0, y: 0, z: 9.8 } }), 1); },
    removeEventListener(type) { delete listeners[type]; }, SpeechRecognition: undefined, AudioContext: sb.AudioContext, speechSynthesis: sb.speechSynthesis };
  ['pocket','start','alerttest','beep','lastmsg','rtt','v','height'].forEach(id => sb.document.getElementById(id));
  vm.createContext(sb);
  vm.runInContext(code.replace('const CAMERA = true;', 'const CAMERA = true;'), sb);
  return sb;
}
const run = (sb, js) => vm.runInContext(js, sb);
const results = [];
function check(name, ok, detail) { results.push(ok); console.log(`[${ok ? 'ok' : 'FAIL'}] ${name}${detail ? ' (' + detail + ')' : ''}`); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  // A. iPhone-like: no vibration -> beep is on by default; the alert test explains it
  let sb = makeSandbox({ iosPermission: 'granted' });
  check('no vibration support -> beep is on by default', sb.els.beep.checked === true);
  sb.els.alerttest.onclick();
  check('"Test alert" explains that iPhone browsers cannot vibrate and beeps', /not supported/.test(sb.els.lastmsg.textContent) && sb.beeps.length >= 2, sb.els.lastmsg.textContent.slice(0, 40));

  // B. Android-like: vibration exists -> beep off by default, urgent events vibrate
  let sb2 = makeSandbox({ vibrate: true });
  check('vibration support -> beep off by default', sb2.els.beep.checked === false);
  run(sb2, "handleEvent({text:'Chair ahead, 0.8 meters', speak:true})");
  check('an urgent warning vibrates', sb2.vibrations.length === 1);
  run(sb2, "handleEvent({text:'Saved map found. Continuing where you left off.', speak:false})");
  check('a quiet event is shown but not spoken or vibrated', sb2.vibrations.length === 1 && !sb2.spoken.some(t => /Saved map/.test(t)) && /Saved map/.test(sb2.els.lastmsg.textContent));
  run(sb2, "handleEvent({text:'Tracking lost.', speak:true})");
  check('"Tracking lost." is spoken and vibrates', sb2.vibrations.length === 2 && sb2.spoken.includes('Tracking lost.'));

  // C. the whole start flow with learning mode
  sb.document.getElementById('height').value = '175';
  const startP = sb.els.start.onclick();
  await sleep(300);
  check('iPhone motion permission asked and listener added', !!sb.listeners.devicemotion);
  check('the page opened the camera socket', sb.sockets.length === 1 && /\/ws\/camera$/.test(sb.sockets[0].url));
  const ws = sb.sockets[0];
  check('learning mode waits for tracking before starting', !ws.sent.some(x => typeof x === 'string' && /"learn"/.test(x)));
  ws.onmessage({ data: JSON.stringify({ state: { status: 'tracking', searching: true }, events: [] }) });
  await sleep(300);
  check('learning does not start while the pipeline still looks for the saved map', !ws.sent.some(x => typeof x === 'string' && /"learn"/.test(x)));
  ws.onmessage({ data: JSON.stringify({ state: { status: 'tracking', searching: false }, events: [] }) });
  await sleep(300);
  const learnMsg = ws.sent.map(x => { try { return JSON.parse(x); } catch (e) { return null; } }).find(m => m && m.type === 'learn');
  check('then it sends "learn start" with the step length from the height', learnMsg && learnMsg.action === 'start' && Math.abs(learnMsg.step_m - 0.726) < 0.001, JSON.stringify(learnMsg));
  for (let i = 0; i < 25; i++) sb.listeners.devicemotion({ accelerationIncludingGravity: { x: 0.1 * i, y: 0.2, z: 9.8 } });
  sb.els.v.videoWidth = 640;
  run(sb, 'flushMotion()');
  const motionMsgs = ws.sent.map(x => { try { return JSON.parse(x); } catch (e) { return null; } }).filter(m => m && m.type === 'motion');
  const total = motionMsgs.reduce((n, m) => n + m.s.length, 0);
  check('motion samples are sent in batches while learning', motionMsgs.length >= 2 && total === 25 && motionMsgs[0].s[0].length === 4, `${motionMsgs.length} batches, ${total} samples`);
  check('the page is not in pocket mode while learning', sb.els.pocket.style.display !== 'flex');
  ws.onmessage({ data: JSON.stringify({ state: { status: 'tracking' }, events: [{ text: 'Scale saved.', speak: true, learn: 'done' }] }) });
  await startP;
  await sleep(100);
  check('after "Scale saved." it asks you to put the phone in the pocket and enters pocket mode', sb.els.pocket.style.display === 'flex' && sb.spoken.some(t => /front pocket/.test(t)));
  check('no motion samples are collected after learning ends', (() => { const n = ws.sent.length; sb.listeners.devicemotion && sb.listeners.devicemotion({ accelerationIncludingGravity: { x: 1, y: 1, z: 9 } }); run(sb, 'flushMotion()'); return ws.sent.length === n; })());

  // D. learning fails -> the page still carries on to pocket mode
  let sb3 = makeSandbox({ iosPermission: 'granted' });
  const p3 = sb3.els.start.onclick(); await sleep(300);
  const w3 = sb3.sockets[0];
  w3.onmessage({ data: JSON.stringify({ state: { status: 'tracking' }, events: [] }) }); await sleep(200);
  w3.onmessage({ data: JSON.stringify({ state: { status: 'tracking' }, events: [{ text: 'Learning failed. Say calibrate to try again.', speak: true, learn: 'failed' }] }) });
  await p3; await sleep(100);
  check('a failed learning still lets the session start', sb3.els.pocket.style.display === 'flex');

  // E. permission refused / no sensors -> old flow with a spoken explanation
  let sb4 = makeSandbox({ iosPermission: 'denied' });
  const p4 = sb4.els.start.onclick(); await p4; await sleep(100);
  check('permission refused: learning is off, the normal start still works', sb4.spoken.some(t => /Motion sensors are not available/.test(t)) && sb4.els.pocket.style.display === 'flex' && sb4.sockets.length === 1);

  // F. voice command "calibrate" starts learning, "learning mode" too
  let sb5 = makeSandbox({ iosPermission: 'granted' });
  const p5 = sb5.els.start.onclick(); await sleep(300);
  const w5 = sb5.sockets[0];
  w5.onmessage({ data: JSON.stringify({ state: { status: 'tracking' }, events: [] }) }); await sleep(200);
  w5.onmessage({ data: JSON.stringify({ events: [{ text: 'Scale saved.', speak: true, learn: 'done' }] }) }); await p5; await sleep(100);
  const before = w5.sent.filter(x => /"learn"/.test(x)).length;
  run(sb5, "ask('please calibrate')"); await sleep(100);
  const after = w5.sent.filter(x => /"learn"/.test(x)).length;
  check('saying "calibrate" starts learning again', after === before + 1);
  check('"calibrate" and "learning mode" pass the spoken-command filter', run(sb5, "COMMAND.test('calibrate now') && COMMAND.test('learning mode')"));

  // G. round trip time is shown
  run(sb5, 'sentAt = performance.now() - 120 * 50');
  w5.onmessage({ data: JSON.stringify({ state: { status: 'tracking' }, events: [] }) });
  check('the round trip time of a frame is shown', /round trip \d+ ms/.test(sb5.els.rtt.textContent), sb5.els.rtt.textContent);

  console.log(results.every(Boolean) ? 'ALL OK' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
