import numpy as np
import pandas as pd
from MFCC import MFCC
from SSC import SSC
from Chroma import Chroma
from MelSpectrogram import MelSpectrogram
from Contrast import Contrast
from Tonnetz import Tonnetz
from F0 import F0
import os
import sys
import os
import librosa
sys.path.insert(0, os.getcwd())
import parameters as para


class Feature:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        (self.signal, self.sample_rate) = librosa.load(self.file_name)
        if 'Chroma' in para.features or 'Contrast' in para.features or 'Tonnetz' in para.features:
            self.stft = np.abs(librosa.stft(self.signal))

    def extract_feature(self):
        result = []

        if 'MFCC' in para.features:
            # Compute MFCC
            result.append(MFCC(file_name=self.file_name, signal=self.signal,
                          sample_rate=self.sample_rate).extract())
        if 'SSC' in para.features:
            # Compute SSC
            result.append(SSC(file_name=self.file_name, signal=self.signal,
                          sample_rate=self.sample_rate).extract())
        if 'Chroma' in para.features:
            # Compute chroma feature
            result.append(Chroma(file_name=self.file_name, signal=self.signal,
                          stft=self.stft, sample_rate=self.sample_rate).extract())
        if 'MelSpectrogram' in para.features:
            # Compute MEL spectrogram feature
            result.append(MelSpectrogram(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate).extract())
        if 'Contrast' in para.features:
            # Compute spectral contrast feature
            result.append(Contrast(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate,stft=self.stft).extract())
        if 'Tonnetz' in para.features:
            # Compute tonnetz feature
            result.append(Tonnetz(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate,stft=self.stft).extract())
        if 'F0' in para.features:
            # Compute F0 feature
            result.append(F0(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate).extract())

        return np.concatenate(result)

