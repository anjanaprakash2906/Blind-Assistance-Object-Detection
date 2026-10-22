"""
Threaded Text-to-Speech engine.

Speech synthesis (pyttsx3) is blocking, so running it directly inside the main
video loop would freeze the camera feed while a sentence is being spoken.
This module runs a dedicated worker thread with a queue: the main loop simply
pushes strings onto the queue and continues processing frames immediately.
"""

import queue
import sys
import threading
import time

import pyttsx3


class TTSEngine:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._startup_error = None
        self._rate = rate
        self._volume = volume

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Could not initialize the text-to-speech engine.") from self._startup_error

    def _worker(self):
        """Runs in a background thread, continuously speaking queued phrases."""
        try:
            # On Windows, pyttsx3's SAPI driver must be initialized and used on
            # the same thread.
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            engine.startLoop(False)
        except Exception as error:
            self._startup_error = error
            self._ready.set()
            return

        self._ready.set()
        try:
            while True:
                phrase = self._queue.get()
                try:
                    if phrase is None:
                        return
                    if self._stop_event.is_set():
                        continue
                    print(f"[AUDIO] Speaking: {phrase}", flush=True)
                    engine.say(phrase)
                    while engine.isBusy():
                        if self._stop_event.is_set():
                            engine.stop()
                            break
                        engine.iterate()
                        time.sleep(0.01)
                except Exception as error:
                    print(f"[ERROR] Text-to-speech playback failed: {error}", file=sys.stderr)
                finally:
                    self._queue.task_done()
        finally:
            engine.endLoop()

    def speak(self, phrase: str):
        """Queue a phrase to be spoken. Non-blocking."""
        if phrase:
            self._queue.put(phrase)

    def clear_pending(self):
        """Drop any phrases waiting to be spoken (keeps the currently spoken one)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def wait_until_idle(self):
        """Wait until all phrases queued so far have finished playing."""
        self._queue.join()

    def stop(self):
        """Stop current and queued speech, then shut down the TTS thread."""
        self._stop_event.set()
        self.clear_pending()
        self._queue.put(None)
        self._thread.join()
