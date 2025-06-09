#!/usr/bin/env python3
import speech_recognition as sr, threading, queue, time, pyautogui, re

# ── tuning ────────────────────────────────────────────
TOL        = 0.02  # reduced for more precise gesture detection
MAX_SPEECH = 15    # increased max speech duration
SPEECH_CD  = 0.3   # reduced cooldown between speech sessions
AMBIENT_DURATION = 0.1  # reduced ambient noise calibration time

# Voice commands mapping
VOICE_COMMANDS = {
    'open chrome': lambda: pyautogui.hotkey('command', 'space') and pyautogui.write('chrome') and pyautogui.press('enter'),
    'scroll down': lambda: pyautogui.scroll(-100),
    'scroll up': lambda: pyautogui.scroll(100),
    'click': lambda: pyautogui.click(),
    'double click': lambda: pyautogui.doubleClick(),
    'right click': lambda: pyautogui.rightClick(),
    'copy': lambda: pyautogui.hotkey('command', 'c'),
    'paste': lambda: pyautogui.hotkey('command', 'v'),
    'cut': lambda: pyautogui.hotkey('command', 'x'),
    'select all': lambda: pyautogui.hotkey('command', 'a'),
    'undo': lambda: pyautogui.hotkey('command', 'z'),
    'redo': lambda: pyautogui.hotkey('command', 'shift', 'z'),
    'new tab': lambda: pyautogui.hotkey('command', 't'),
    'close tab': lambda: pyautogui.hotkey('command', 'w'),
    'switch tab': lambda: pyautogui.hotkey('command', 'tab'),
    'refresh': lambda: pyautogui.hotkey('command', 'r'),
    'go back': lambda: pyautogui.hotkey('command', '['),
    'go forward': lambda: pyautogui.hotkey('command', ']'),
    'zoom in': lambda: pyautogui.hotkey('command', '+'),
    'zoom out': lambda: pyautogui.hotkey('command', '-'),
    'full screen': lambda: pyautogui.hotkey('command', 'f'),
    'minimize': lambda: pyautogui.hotkey('command', 'm'),
    'quit': lambda: pyautogui.hotkey('command', 'q'),
}

def ext(lm, tip, pip):  return lm[tip].y < lm[pip].y - TOL
def cur(lm, tip, pip):  return lm[tip].y > lm[pip].y + TOL

class SpeechDictation:
    """Thumb+pinky = speech; closed hand = Enter; pinky‑only = F3."""
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300  # increased for better noise rejection
        self.rec.dynamic_energy_threshold = True
        self.rec.pause_threshold = 0.5  # increased for better phrase detection
        self.rec.phrase_threshold = 0.3  # increased for better phrase detection
        self.state = "idle"
        self.stop_flag = False
        self.last_speech = 0
        self.audio_q = queue.Queue()
        self.enter_t0 = None; self.last_enter = 0
        self.f3_t0    = None; self.last_f3 = 0
        self.is_recording = False
        self.recording_thread = None
        self.mic = None
        threading.Thread(target=self._worker, daemon=True).start()

    def update(self, lm):
        now = time.time()
        t_ext = ext(lm,4,2); t_cur = cur(lm,4,2)
        idx_ext = ext(lm,8,6); idx_cur = cur(lm,8,6)
        mid_ext = ext(lm,12,10); mid_cur = cur(lm,12,10)
        ring_ext= ext(lm,16,14); ring_cur= cur(lm,16,14)
        pink_ext= ext(lm,20,18); pink_cur= cur(lm,20,18)

        # Speech gesture: thumb and pinky extended (Y shape), others curled
        speech_g = t_ext and pink_ext and idx_cur and mid_cur and ring_cur

        # Start recording when gesture is detected
        if speech_g and not self.is_recording and now-self.last_speech>SPEECH_CD:
            self.is_recording = True
            self.state = "recording"
            self.stop_flag = False
            if self.recording_thread is None or not self.recording_thread.is_alive():
                self.recording_thread = threading.Thread(target=self._record, daemon=True)
                self.recording_thread.start()
        
        # Stop recording when gesture is released
        elif not speech_g and self.is_recording:
            self.is_recording = False
            self.stop_flag = True
            self.state = "idle"
            self.last_speech = now
            if self.mic:
                try:
                    self.mic.stop()
                except:
                    pass

        # ENTER gesture: all fingers curled (relaxed fist)
        if idx_cur and mid_cur and ring_cur and pink_cur and t_cur:
            if self.enter_t0 is None: self.enter_t0=now
            if now-self.enter_t0>=0.3 and now-self.last_enter>1.5:
                pyautogui.press("enter")
                self.last_enter=now; self.enter_t0=None
        else:
            self.enter_t0=None

        # F3 gesture removed as it's not part of the spec

    def _record(self):
        try:
            self.mic = sr.Microphone()
            with self.mic as source:
                self.rec.adjust_for_ambient_noise(source, duration=AMBIENT_DURATION)
                while self.is_recording and not self.stop_flag:
                    try:
                        audio = self.rec.listen(source, phrase_time_limit=MAX_SPEECH)
                        if not self.stop_flag:
                            self.audio_q.put(audio)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error during recording: {e}")
                        break
        except Exception as e:
            print(f"Speech recognition error: {e}")

    def _worker(self):
        while True:
            try:
                audio = self.audio_q.get()
                text = self.rec.recognize_google(audio).lower()
                
                # Check for voice commands first
                for cmd, action in VOICE_COMMANDS.items():
                    if cmd in text:
                        action()
                        return
                
                # If no command matched, treat as dictation
                pyautogui.typewrite(text + " ")
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results: {e}")
            except Exception as e:
                print(f"Error in speech worker: {e}")
