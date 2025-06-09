import speech_recognition as sr
import threading

class SpeechDictation:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.last_text = ""
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        while True:
            with self.microphone as source:
                audio = self.recognizer.listen(source, phrase_time_limit=3)
            try:
                text = self.recognizer.recognize_google(audio)
                self.last_text = text
                print(f"Recognized: {text}")
            except Exception:
                pass

    def update(self, landmarks):
        pass 