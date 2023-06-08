
import os
import numpy as np
import time
import random
import subprocess
import sys
import librosa





# # Function to convert ogg to wav
# def ogg2wav(src_filename, dest_filename):
#     process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
#     print(src_filename, dest_filename)

#     if process.returncode != 0:
#         raise Exception("Error converting file")

def ogg2wav(src_filename, dest_filename):
    import soundfile as sf
    y, sr = sf.read(src_filename)
    # Save the audio file as WAV using soundfile
    sf.write(dest_filename, y, sr)

def getWavDetail(path):
    import wave

    # Open the WAV file
    with wave.open(path, "rb") as wav_file:
        # Get the file metadata
        # num_channels = wav_file()
        getparams = wav_file.getparams()
        compname = wav_file.getcompname()
        comptype = wav_file.getcomptype()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / float(sample_rate)

    # Print the metadata to the console
    print("getparams:", getparams)
    print("compname:", compname)
    print("comptype:", comptype)
    print("Number of channels:", num_channels)
    print("Sample width (bytes):", sample_width)
    print("Sample rate (Hz):", sample_rate)
    print("Number of frames:", num_frames)
    print("Duration (seconds):", duration)

def scale_amplitude(src_filename, dest_filename,ratio=0.2):
    import soundfile as sf
    y, sr = sf.read(src_filename)
    print(np.max(y))
    print(np.min(y))
    y = y*((ratio*2)/(np.max(y)-np.min(y)))
    print(np.max(y))
    print(np.min(y))
    # Save the audio file as WAV using soundfile
    sf.write(dest_filename, y, sr)

def print_signal_array(path):
    (signal, sample_rate) = librosa.load(path)
    return signal

def reduce_noise(path,after_path):
    from scipy.io import wavfile
    import noisereduce as nr
    # load data
    rate, data = wavfile.read(path)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(after_path, rate, reduced_noise)
