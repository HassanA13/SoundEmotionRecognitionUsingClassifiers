from python_speech_features import mfcc
from scipy.stats import skew, kurtosis
import numpy as np
import librosa
import pandas as pd
import tqdm
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
import modules.FeaturesManagement.helper as featureHelper

class MFCC:
    def __init__(self, sample_rate, signal, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate

    def extract(self):
        mfcc_feat = np.array(mfcc(self.signal, self.sample_rate, numcep=40, nfilt=40, nfft=1024))

        result = np.concatenate((
            mfcc_feat.mean(axis=0),
            mfcc_feat.min(axis=0),
            mfcc_feat.max(axis=0),
            mfcc_feat.var(axis=0),
            np.median(mfcc_feat, axis=0),
            np.ptp(mfcc_feat, axis=0),
            skew(mfcc_feat, axis=0, bias=True),
            kurtosis(mfcc_feat, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(
            mfcc_feat, [75, 50, 25], axis=0))
        return result


    def extract_feature(file_name):
        signal, sample_rate = librosa.load(file_name)

        result = []
        if 'MFCC' in para.features:
            # Compute MFCC
            result.append(MFCC(file_name=file_name, signal=signal,sample_rate=sample_rate).extract())

        return np.concatenate(result)


    def extract_feature_emotion_X_y_array():
        feature_emotion_X_y_array_name = featureHelper.get_name_datasets_feature_emotions(feature='MFCC')

        if os.path.isfile(feature_emotion_X_y_array_name):
            # if file already exists, just load then
            if para.verbose:
                print("[+] Feature file already exists, loading...")
            feature_emotion_X_y_array = pd.read_csv(feature_emotion_X_y_array_name)
            feature_emotion_X_y_array = feature_emotion_X_y_array.iloc[:,1:].values

        else:
            path_emotion_array = pd.DataFrame({'path': [], 'emotion': []})
            for dataset_name in para.datasets:
                file_name = f"{dataset_name}.csv"
                path_emotion_array = pd.concat(
                    (path_emotion_array, pd.read_csv(file_name)), sort=False)

            feature_emotion_X_y_array = pd.DataFrame()
            pbar = tqdm.tqdm(total=path_emotion_array['path'].size)
            for path, emotion in zip(path_emotion_array['path'], path_emotion_array['emotion']):
                if not (emotion in para.emotions):
                    continue
                feature_emotion_X_y_array = feature_emotion_X_y_array.append(pd.Series(np.append(MFCC.extract_feature(path), para.emotions.index(emotion))), ignore_index=True)
                pbar.update(1)
            pbar.close()
            feature_emotion_X_y_array.to_csv(feature_emotion_X_y_array_name)
            feature_emotion_X_y_array = feature_emotion_X_y_array.values

        X = feature_emotion_X_y_array[:,:-1]

        return {
            "X": X,
            "y": np.concatenate(feature_emotion_X_y_array[:,-1:]),
        }
    
if __name__ == '__main__':
    MFCC.extract_feature_emotion_X_y_array()
