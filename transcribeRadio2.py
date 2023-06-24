import requests
import whisper
import torch
from pydub import AudioSegment
from io import BytesIO
import io
import queue
import threading
import datetime
import os
import gradio as gr


import matplotlib as mpl
font_cache_path = mpl.get_cachedir() + '/fontList.cache'
print(font_cache_path)
#%rm $font_cache_path

# Step 1: Get the radio stream
stream_url = "http://stream.zeno.fm/y9y2bhvzs4zuv"  # Your stream URL

# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set language
language_whisper = "english"

model_fp32 = whisper.load_model(
    name="small",
    device=device)

# Create a queue to store the audio files
audio_queue = queue.Queue()

# Create placeholders for the audio player, log text area, and transcript text area
log_text_area = gr.outputs.Textbox(label="Log")
transcript_text_area = gr.outputs.Textbox(label="Transcript")
audio_player = gr.outputs.Audio(label="Audio")


# Function to transcribe audio files
def transcribe_audio():
    while True:
        # Get the next audio file from the queue
        audio_file = audio_queue.get()
        print(audio_file)
        print("start transcribe")

        # Log the start of transcription
        gr.set_output(log_text_area="transcription start")

        # Transcribe the audio
        result = model_fp32.transcribe(audio_file)

        # Update the audio player and the text box
        audio_player.audio(audio_file)
        gr.set_output(transcript_text_area=result["text"])
        gr.set_output(log_text_area="transcription complete")

        
        # Log the end of transcription
        gr.set_output(log_text_area="transcription complete of " + audio_file)
        # Mark the task as done
        audio_queue.task_done()

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.start()

# Download the stream
response = requests.get(stream_url, stream=True)

# Create a buffer for the incoming audio data
buffer = BytesIO()

gr.set_output(log_text_area="Starting stream processing")

# Process the stream in chunks of 10 seconds
for block in response.iter_content(1024):
    # Write the audio data to our buffer
    buffer.write(block)
    # If we have 10 seconds of audio data
    if buffer.tell() >= 10 * 1024 * 100:
        gr.set_output(log_text_area="Saving audio file")
        # Get the current datetime
        now = datetime.datetime.now()

        # Create a directory for the audio files
        os.makedirs('audio', exist_ok=True)

        # Create a filename with the datetime and directory path
        filename = f'audio/stream_{now.strftime("%Y%m%d_%H%M%S")}.wav'

        # Save the buffer to a file
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())

        # Add the filename to the queue
        audio_queue.put(filename)

        # Log the creation of the audio file
        gr.set_output(log_text_area="Created audio file " + filename)

        # Clear the buffer
        buffer.seek(0)
        buffer.truncate()
