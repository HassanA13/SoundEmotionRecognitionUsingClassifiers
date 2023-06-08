# from IPython.display import IFrame
# IFrame('https://tonejs.github.io/examples/envelope.html', width=700, height=350)

import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
# "E:\Development\My_graduation_project\SER-emotion-recognition-using-speech\temp\1685908351_9901456388.wav"
# "E:\Development\My_graduation_project\SER-emotion-recognition-using-speech\temp\1685908351_after_reduce_noise_2288118154.wav"
# "E:\Development\My_graduation_project\SER-emotion-recognition-using-speech\temp\1685908843_6688493281.wav"
# "E:\Development\My_graduation_project\SER-emotion-recognition-using-speech\temp\1685908843_after_reduce_noise_1631162905.wav"
# path = 'temp\\1685907474_7141767664.wav'
# path = 'temp\\1685908351_9901456388.wav'
# path = 'temp\\1685908843_6688493281.wav'
path = 'temp\\1685908843_after_reduce_noise_1631162905.wav'
x, sr = librosa.load(path)
# x=x*0.2
# print(x.shape)
# print(sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.xlabel("Time (s)")
plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
ipd.Audio(path) # load a local WAV file