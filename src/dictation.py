#!/usr/bin/env python3
import speech_recognition as sr
import threading
import time
import pyautogui
import queue
import traceback

# ── tuning ────────────────────────────────────────────
TOL          = 0.03   # curl / extend tolerance (unused here)
MAX_SPEECH   = 10     # sec
SPEECH_CD    = 2      # cooldown between dictations

def ext(lm, tip, pip):
    return lm[tip].y < lm[pip].y - TOL

def cur(lm, tip, pip):
    return lm[tip].y > lm[pip].y + TOL

class SpeechDictation:
    """Thumb+pinky = speech; closed hand = Enter; pinky-only = F3."""
    def __init__(self):
        print("Initializing speech dictation...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.adjust_for_ambient_noise = True
        self.state = "idle"
        self.stop_flag = False
        self.last_speech = 0
        self.enter_t0 = None; self.last_enter = 0
        self.f3_t0    = None; self.last_f3 = 0
        
        # Adjust recognizer settings for better performance
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.pause_threshold = 0.5  # Shorter pause threshold
        self.recognizer.phrase_threshold = 0.3  # Shorter phrase threshold
        self.recognizer.non_speaking_duration = 0.3  # Shorter non-speaking duration
        
        threading.Thread(target=self._process_audio, daemon=True).start()
        
        # Performance optimization
        self.gesture_history = []
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3  # Reduced cooldown for faster response
        
        print("Speech dictation initialized successfully")
        
    def start(self):
        """Start dictation in a separate thread."""
        if not self.is_listening:
            self.is_listening = True
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            # Start listening in a separate thread
            threading.Thread(target=self._listen_loop, daemon=True).start()
            
    def stop(self):
        """Stop dictation."""
        self.is_listening = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
    def _listen_loop(self):
        """Continuously listen for audio input."""
        try:
            with self.microphone as source:
                if self.adjust_for_ambient_noise:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    self.adjust_for_ambient_noise = False
                
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5.0)
                        self.audio_queue.put(audio)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error in listen loop: {str(e)}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"Error in microphone setup: {str(e)}")
            
    def _process_audio(self):
        """Process audio from the queue."""
        while self.is_listening:
            try:
                audio = self.audio_queue.get(timeout=1.0)
                try:
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        print(f"Dictated: {text}")
                        # Type out the text
                        pyautogui.typewrite(text + " ")
                except sr.UnknownValueError:
                    pass  # Ignore unrecognized speech
                except sr.RequestError as e:
                    print(f"Could not request results: {str(e)}")
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)

    def update(self, landmarks):
        """Update dictation state based on hand landmarks"""
        if landmarks is None:
            return
            
        try:
            now = time.time()
            t_ext = ext(landmarks,4,2); t_cur = cur(landmarks,4,2)
            idx_ext = ext(landmarks,8,6); idx_cur = cur(landmarks,8,6)
            mid_ext = ext(landmarks,12,10); mid_cur = cur(landmarks,12,10)
            ring_ext= ext(landmarks,16,14); ring_cur= cur(landmarks,16,14)
            pink_ext= ext(landmarks,20,18); pink_cur= cur(landmarks,20,18)

            speech_g = t_ext and pink_ext and idx_cur and mid_cur and ring_cur
            closed_g = idx_cur and mid_cur and ring_cur and pink_cur
            pinky_g  = pink_ext and t_cur and idx_cur and mid_cur and ring_cur

            if speech_g and self.state=="idle" and now-self.last_speech>SPEECH_CD:
                self.state="recording"; self.stop_flag=False
                threading.Thread(target=self._record, daemon=True).start()
            elif not speech_g and self.state=="recording":
                self.stop_flag=True
                self.state="idle"
                self.last_speech=now

            # ENTER
            if closed_g:
                if self.enter_t0 is None: self.enter_t0=now
                if now-self.enter_t0>=0.3 and now-self.last_enter>1.5:
                    pyautogui.press("enter")
                    self.last_enter=now; self.enter_t0=None
            else:
                self.enter_t0=None

            # F3
            if pinky_g:
                if self.f3_t0 is None: self.f3_t0=now
                if now-self.f3_t0>=0.2 and now-self.last_f3>2.0:
                    pyautogui.press("f3")
                    self.last_f3=now; self.f3_t0=None
            else:
                self.f3_t0=None

            # Update gesture history
            if now - self.last_gesture_time >= self.gesture_cooldown:
                self.gesture_history.append(speech_g)
                if len(self.gesture_history) > 3:
                    self.gesture_history.pop(0)
                
                # Check for consistent gesture
                if len(self.gesture_history) >= 2:
                    if all(self.gesture_history) and self.state=="idle":
                        self.state="recording"
                    elif not any(self.gesture_history) and self.state=="recording":
                        self.state="idle"
                
                self.last_gesture_time = now
                
        except Exception as e:
            print(f"Error in dictation update: {str(e)}")
            traceback.print_exc()
            
    def _record(self):
        with self.microphone as m:
            audio = self.recognizer.listen(m, phrase_time_limit=MAX_SPEECH)
            if not self.stop_flag:
                self.audio_queue.put(audio)
