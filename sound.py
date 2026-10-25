"""Sound: speak answers and warnings, and listen to spoken queries.

No extra packages are needed on Windows: it uses the speech engine built into Windows through
PowerShell. On macOS it uses `say`, on Linux `espeak` or `spd-say` if installed.
Every function fails quietly (returns False/None) when the computer has no speech support,
so nothing else breaks.

Speech can never block the program for long: every call has a time limit that grows with the
length of the text, the speech process (and anything it started) is killed when the limit is
reached, and after repeated time-outs speech is paused for a minute so warnings keep flowing.
"""
import os
import platform
import queue
import shutil
import signal
import subprocess
import threading
import time

SYSTEM = platform.system()

MIN_SPEAK_SECONDS = 4.0        # even a short line gets this long (the speech engine needs a moment to start)
SECONDS_PER_WORD = 0.6         # normal speech is about 2.5 words a second, so this is generous
MAX_SPEAK_SECONDS = 20.0       # no line may block longer than this
LISTEN_SECONDS = 20.0          # the Windows recognizer listens for 8 s; the rest is start-up time
BREAKER_TIMEOUTS = 3           # this many time-outs within BREAKER_WINDOW ...
BREAKER_WINDOW = 60.0
BREAKER_PAUSE = 60.0           # ... pause speech for this long

_timeouts = []                 # times (time.monotonic) of recent time-outs
_paused_until = 0.0
_lock = threading.Lock()
last_status = None             # 'ok', 'timeout' or 'failed' for the most recent speak_text call


def speech_timeout(text):
    """Seconds a line may take: a few seconds plus a little per word, never more than the maximum."""
    words = len(str(text).split())
    return min(MAX_SPEAK_SECONDS, MIN_SPEAK_SECONDS + SECONDS_PER_WORD * words)


def _kill_tree(proc):
    """Kill a process and everything it started (PowerShell starts helper processes)."""
    try:
        if SYSTEM == 'Windows':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True, timeout=5)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run(cmd, timeout, env=None, capture=False):
    """Run a command with a hard time limit. Returns (succeeded, timed_out, stdout)."""
    kwargs = {'stdin': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'env': env,
              'stdout': subprocess.PIPE if capture else subprocess.DEVNULL}
    if capture:
        kwargs['text'] = True
    if SYSTEM == 'Windows':
        kwargs['creationflags'] = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    else:
        kwargs['start_new_session'] = True       # its own process group, so the whole group can be killed
    proc = subprocess.Popen(cmd, **kwargs)
    deadline = time.monotonic() + timeout
    try:
        while True:                               # wait in short slices so Ctrl+C is noticed quickly (also on Windows)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                out, _ = proc.communicate(timeout=min(0.2, remaining))
                return proc.returncode == 0, False, (out or '')
            except subprocess.TimeoutExpired:
                continue
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=3)
        except (subprocess.SubprocessError, OSError):
            pass
        return False, True, ''
    except BaseException:                         # Ctrl+C or any other interruption: never leave it running
        _kill_tree(proc)
        raise


def _speak_command(text):
    """(command, environment) that speaks the text on this system, or (None, None)."""
    if SYSTEM == 'Windows':
        env = dict(os.environ, SPEAK_TEXT=text)           # passed as data, never as code
        script = ('Add-Type -AssemblyName System.Speech; '
                  '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:SPEAK_TEXT)')
        return ['powershell', '-NoProfile', '-Command', script], env
    if SYSTEM == 'Darwin' and shutil.which('say'):
        return ['say', text], None
    for tool in ('espeak', 'spd-say'):
        if shutil.which(tool):
            return [tool, text], None
    return None, None


def speech_paused():
    return time.monotonic() < _paused_until


def _note_timeout():
    """Remember a time-out; after several in a row, pause speech for a minute."""
    global _paused_until
    now = time.monotonic()
    with _lock:
        _timeouts[:] = [t for t in _timeouts if now - t < BREAKER_WINDOW] + [now]
        if len(_timeouts) >= BREAKER_TIMEOUTS:
            _paused_until = now + BREAKER_PAUSE
            _timeouts.clear()
            print(f'Sound: speech kept timing out, so it is paused for {BREAKER_PAUSE:.0f} seconds. '
                  'Messages are still printed.')


def speak_text(text, timeout=None):
    """Say text out loud and wait until it is finished, but never longer than `timeout` seconds
    (default: a few seconds plus a little per word). Returns True if it could speak."""
    global last_status
    text = ' '.join(str(text).split())
    last_status = 'failed'
    if not text or speech_paused():
        return False
    cmd, env = _speak_command(text)
    if cmd is None:
        return False
    try:
        ok, timed_out, _ = _run(cmd, timeout or speech_timeout(text), env)
    except OSError:
        return False
    if timed_out:
        last_status = 'timeout'
        print('Sound: speech took too long and was stopped.')
        _note_timeout()
    elif ok:
        last_status = 'ok'
    return ok


class Speaker:
    """Speaks in a background thread so the camera loop is never blocked.

    say(text, key, gap): the same key is not repeated within `gap` seconds, and old messages
    are dropped if the queue backs up, so warnings stay current.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._queue = queue.Queue()
        self._last = {}
        self.works = None            # becomes True/False after the first attempt
        if enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text, key=None, gap=8.0):
        if not self.enabled:
            return
        key = key or text
        now = time.time()
        if now - self._last.get(key, 0) < gap:
            return
        self._last[key] = now
        while self._queue.qsize() >= 2:          # keep only the freshest messages
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def _worker(self):
        while True:
            text = self._queue.get()
            ok = speak_text(text)
            if self.works is None and last_status != 'timeout':      # a time-out says nothing about the engine
                self.works = ok
                if not ok:
                    print('Sound: this computer could not speak (no speech engine found). '
                          'Messages are still printed.')


WINDOWS_LISTEN = r'''
Add-Type -AssemblyName System.Speech
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$engine.SetInputToDefaultAudioDevice()
$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$result = $engine.Recognize([TimeSpan]::FromSeconds(8))
if ($result) { Write-Output $result.Text }
'''


def listen_once():
    """Listen on the microphone for up to about 8 seconds. Returns the text, or None.

    Uses the Windows built-in recognizer (needs a microphone and Windows speech recognition
    for your language). Elsewhere it uses the SpeechRecognition package if installed.
    """
    try:
        if SYSTEM == 'Windows':
            ok, timed_out, out = _run(['powershell', '-NoProfile', '-Command', WINDOWS_LISTEN],
                                      LISTEN_SECONDS, capture=True)
            text = out.strip()
            return text or None
        import speech_recognition as sr             # optional, other systems only
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            audio = recognizer.listen(mic, timeout=8, phrase_time_limit=8)
        return recognizer.recognize_google(audio)
    except Exception:                               # no microphone, no engine, nothing heard
        return None
