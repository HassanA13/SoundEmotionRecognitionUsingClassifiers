import pyaudio
import wave

# Define the audio recording parameters
# FORMAT = pyaudio.paInt16
FORMAT = pyaudio.paFloat32
CHANNELS = 1
# RATE = 44100
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 2

# Create a new PyAudio object
audio = pyaudio.PyAudio()

# Open a new stream for recording audio data
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Start recording audio data
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop recording audio data and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded audio data to a WAV file
with wave.open("output_file.wav", "wb") as wav_file:
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(audio.get_sample_size(FORMAT))
    wav_file.setframerate(RATE)
    wav_file.writeframes(b"".join(frames))