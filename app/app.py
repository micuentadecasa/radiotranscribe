from flask import Flask, render_template
from multiprocessing import Process, Manager
import requests
import whisper
import torch
from pydub import AudioSegment
from io import BytesIO
import queue
import datetime
import os
import logging

app = Flask(__name__)

# Create a Manager object to manage shared state
manager = Manager()

# Create queues in the manager's namespace
audio_queue = manager.Queue()
results_queue = manager.Queue()

# Shared variable to keep track of the latest audio file
latest_audio = manager.Value('s', "")
latest_transcript = manager.Value('s', "")

# Function to transcribe audio files
def transcribe_audio(audio_queue, results_queue, latest_transcript):
    # Set up the ASR model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fp32 = whisper.load_model(name="small", device=device)

    while True:
        try:
            audio_file = audio_queue.get()
            results_queue.put("Transcription start")
            result_dict = model_fp32.transcribe(audio_file)
            result = result_dict['text'] if 'text' in result_dict else "No transcription available"
            latest_transcript.value = result
            results_queue.put("Transcription complete: " + result)
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            continue

# Start the transcription process
transcription_process = Process(target=transcribe_audio, args=(audio_queue, results_queue, latest_transcript))
transcription_process.start()

@app.route("/")
def home():
    return "Stream transcription app is running. Check /results for the latest result."

@app.route("/results")
def results():
    # Retrieve and return the latest result
    latest_result = latest_transcript.value
    latest_audio_path = latest_audio.value  # Retrieve the latest audio file from the shared variable
    return render_template('results.html', transcript=latest_result, audio_file=latest_audio_path)



def process_stream():
    stream_url = "http://stream.zeno.fm/y9y2bhvzs4zuv"  # Your stream URL

    try:
        response = requests.get(stream_url, stream=True)
        buffer = BytesIO()
        results_queue.put("Starting stream processing")

        for block in response.iter_content(1024):
            buffer.write(block)
            if buffer.tell() >= 10 * 1024 * 100:
                results_queue.put("Saving audio file")
                now = datetime.datetime.now()
                os.makedirs('static/audio', exist_ok=True)
                filename = f'static/audio/stream_{now.strftime("%Y%m%d_%H%M%S")}.wav'

                with open(filename, 'wb') as f:
                    f.write(buffer.getvalue())

                audio_queue.put(filename)
                results_queue.put("Created audio file " + filename)

                # Store the filename in the shared variable
                latest_audio.value = filename
                
                buffer.seek(0)
                buffer.truncate()
    except Exception as e:
        logging.error(f"Error in process_stream method: {e}")

if __name__ == "app":
    # Start the stream processing process
    stream_process = Process(target=process_stream)
    stream_process.start()

    app.run(debug=True)
