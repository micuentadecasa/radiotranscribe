import requests
import whisper
import torch
from pydub import AudioSegment
from io import BytesIO
import queue
import datetime
import os
import logging
import threading

class ProcessStream:
    def __init__(self):
        self.stream_url = "http://stream.zeno.fm/y9y2bhvzs4zuv"  # Your stream URL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language_whisper = "english"
        self.model_fp32 = whisper.load_model(name="small", device=self.device)
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.transcription_thread = threading.Thread(target=self.transcribe_audio)
        self.transcription_thread.start()

    def transcribe_audio(self):
        while True:
            try:
                audio_file = self.audio_queue.get()
                self.results_queue.put("Transcription start")
                result = self.model_fp32.transcribe(audio_file)
                self.results_queue.put("Transcription complete")
                self.results_queue.put("Transcription complete of " + audio_file)
                self.audio_queue.task_done()
            except Exception as e:
                logging.error(f"Error in transcription: {e}")
                continue

    def run(self):
        try:
            response = requests.get(self.stream_url, stream=True)
            buffer = BytesIO()
            self.results_queue.put("Starting stream processing")
            for block in response.iter_content(1024):
                buffer.write(block)
                if buffer.tell() >= 10 * 1024 * 10:
                    self.results_queue.put("Saving audio file")
                    now = datetime.datetime.now()
                    os.makedirs('static/audio', exist_ok=True)
                    filename = f'static/audio/stream_{now.strftime("%Y%m%d_%H%M%S")}.wav'
                    with open(filename, 'wb') as f:
                        f.write(buffer.getvalue())
                    self.audio_queue.put(filename)
                    self.results_queue.put("Created audio file " + filename)
                    buffer.seek(0)
                    buffer.truncate()
        except Exception as e:
            logging.error(f"Error in run method: {e}")

    def get_latest_result(self):
        try:
            while True:
                result = self.results_queue.get_nowait()
        except queue.Empty:
            result = "No transcription available"
        return result

    def get_latest_audio(self):
        try:
            audio_file = self.audio_queue.queue[-1]
        except IndexError:
            audio_file = None
        return audio_file
