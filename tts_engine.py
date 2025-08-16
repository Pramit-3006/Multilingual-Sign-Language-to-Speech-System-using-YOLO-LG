import tempfile, os

class TTSEngine:
    def __init__(self, backend="pyttsx3", rate=170, volume=1.0):
        self.backend = backend
        if backend == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
        elif backend == "gtts":
            from gtts import gTTS
            self.gTTS = gTTS
        else:
            raise ValueError("Unsupported TTS backend: %s" % backend)

    def speak(self, text: str, lang: str = "en"):
        if not text.strip():
            return
        if self.backend == "pyttsx3":
            # pyttsx3 is offline but may not support all languages flawlessly
            self.engine.say(text)
            self.engine.runAndWait()
        elif self.backend == "gtts":
            # gTTS requires internet connection
            tts = self.gTTS(text=text, lang=lang)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            path = tmp.name
            tmp.close()
            tts.save(path)
            # Try to play the audio; if not possible, just save file
            try:
                import simpleaudio as sa
                wave_obj = sa.WaveObject.from_wave_file(path)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            except Exception:
                pass
            return path
