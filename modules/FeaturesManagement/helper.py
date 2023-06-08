import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para

def get_name_datasets_feature_emotions(prefix='',feature='', extension='.csv'):
    name = prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    # for feature in para.features:
    #     name = name + \
    #         f"_{feature}"
    name = name + \
        f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    name = os.path.join(
        para.feature_emotion_X_Y_array_folder_path_name, f"{name}{extension}")
    return name
