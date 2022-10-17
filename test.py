import torch
import whisper

# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Set language (korean)#
language_whisper = "english"


model_fp32 = whisper.load_model(
    name="small",
    device="cuda")

result = model_fp32.transcribe("./files/audio.wav")
print(result["text"])

result = model_fp32.transcribe("./files/radioswahili.mp3")


model = whisper.load_model("tiny")



# test 1
audio = whisper.load_audio("./files/audio.wav")
mel   = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)


# test 2
result = model.transcribe("./files/radiolingala.mp3")
print(result["text"])