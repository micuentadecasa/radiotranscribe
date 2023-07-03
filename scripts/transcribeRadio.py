from pydub import AudioSegment
import requests
import whisper
import torch

# Step 1: Get the radio stream
stream_url = "http://stream.zeno.fm/y9y2bhvzs4zuv"  # Your stream URL

# Download the stream
response = requests.get(stream_url, stream=True)
# Save the stream to a .wav file
with open('stream.wav', 'wb') as f:
    for block in response.iter_content(1024):
        f.write(block)


# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set language (korean)#
language_whisper = "english"


model_fp32 = whisper.load_model(
    name="small",
    device=device)

#response  = model_fp32.asr( audio.raw_data, language=language_whisper)
result = model_fp32.transcribe("stream.wav")
print(result["text"])

#print(response['choices'][0]['transcript'])
