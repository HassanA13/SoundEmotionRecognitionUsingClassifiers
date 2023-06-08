import numpy as np
from MFCC import MFCC
from SSC import SSC
from Chroma import Chroma
from MelSpectrogram import MelSpectrogram
from Contrast import Contrast
from Tonnetz import Tonnetz
from F0 import F0
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para


def get_name_datasets_features_emotions(prefix='', extension='.csv'):
    name = prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    for feature in para.features:
        name = name + \
            f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    name = os.path.join(
        para.feature_emotion_X_Y_array_folder_path_name, f"{name}{extension}")
    return name

def extract_feature_emotion_X_y_array(filter=True):
    result = []

    if 'MFCC' in para.features:
        feature_emotion_X_Y_array = MFCC.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'SSC' in para.features:
        feature_emotion_X_Y_array = SSC.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'Chroma' in para.features:
        feature_emotion_X_Y_array = Chroma.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'MelSpectrogram' in para.features:
        feature_emotion_X_Y_array = MelSpectrogram.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'Contrast' in para.features:
        feature_emotion_X_Y_array = Contrast.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'Tonnetz' in para.features:
        feature_emotion_X_Y_array = Tonnetz.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'F0' in para.features:
        feature_emotion_X_Y_array = F0.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)

    result = np.concatenate(result,axis=1)
    if filter == True:
        sys.path.insert(0, os.getcwd()+"\modules\FeatureSelectionManagement")
        from  modules.FeatureSelectionManagement.CatBoostFeatureSelector import CatBoostFeatureSelector 
        result = CatBoostFeatureSelector.filter_features(result)
    print(result.shape)

    return {
        "X": result,
        "y": y,
    }


if __name__ == '__main__':

    extract_feature_emotion_X_y_array()
