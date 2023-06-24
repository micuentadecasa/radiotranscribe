import requests
import whisper
import torch
from pydub import AudioSegment
from io import BytesIO
import io

# Step 1: Get the radio stream
stream_url = "http://stream.zeno.fm/y9y2bhvzs4zuv"  # Your stream URL
#stream_url = "http://rs1.radiostreamer.com:8000/listen.pls"  # Your stream URL
# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set language
language_whisper = "english"

model_fp32 = whisper.load_model(
    name="small",
    device=device)

# Download the stream
response = requests.get(stream_url, stream=True)

# Create a buffer for the incoming audio data
buffer = BytesIO()

# Process the stream in chunks of 10 seconds
for block in response.iter_content(1024):
    # Write the audio data to our buffer
    buffer.write(block)

    # If we have 10 seconds of audio data
    if buffer.tell() >= 10 * 1024 *100:
        # saves the buffer to a file .wav
        with open('stream.wav', 'wb') as f:
            f.write(buffer.getvalue())
      
        # Transcribe the audio
        result = model_fp32.transcribe("stream.wav")
        print("----------------------------------")
        print(result["text"])
        print("----------------------------------")

        # Clear the buffer
        buffer.seek(0)
        buffer.truncate()
