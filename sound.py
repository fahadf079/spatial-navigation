"""Sound: speak answers and warnings, and listen to spoken queries.

No extra packages are needed on Windows: it uses the speech engine built into Windows through
PowerShell. On macOS it uses `say`, on Linux `espeak` or `spd-say` if installed.
Every function fails quietly (returns False/None) when the computer has no speech support,
so nothing else breaks.
"""
import os
import platform
import queue
import shutil
import subprocess
import threading
import time

SYSTEM = platform.system()


def speak_text(text):
    """Say text out loud and wait until it is finished. Returns True if it could speak."""
    text = ' '.join(str(text).split())
    if not text:
        return False
    try:
        if SYSTEM == 'Windows':
            env = dict(os.environ, SPEAK_TEXT=text)      # passed as data, never as code
            script = ('Add-Type -AssemblyName System.Speech; '
                      '(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:SPEAK_TEXT)')
            done = subprocess.run(['powershell', '-NoProfile', '-Command', script], env=env,
                                  capture_output=True, timeout=60)
            return done.returncode == 0
        if SYSTEM == 'Darwin' and shutil.which('say'):
            return subprocess.run(['say', text], timeout=60).returncode == 0
        for tool in ('espeak', 'spd-say'):
            if shutil.which(tool):
                return subprocess.run([tool, text], timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        pass
    return False


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
            if self.works is None:
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
            done = subprocess.run(['powershell', '-NoProfile', '-Command', WINDOWS_LISTEN],
                                  capture_output=True, text=True, timeout=30)
            text = done.stdout.strip()
            return text or None
        import speech_recognition as sr             # optional, other systems only
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            audio = recognizer.listen(mic, timeout=8, phrase_time_limit=8)
        return recognizer.recognize_google(audio)
    except Exception:                               # no microphone, no engine, nothing heard
        return None
