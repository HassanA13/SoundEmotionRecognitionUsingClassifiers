import pandas as pd
import glob
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para


def write_csv(emotions, dataset_name, verbose=0):
    X = []
    y = []
    for emotion in emotions:
        target = {"path": [], "emotion": []}
        for file in glob.glob(f"dataset/{dataset_name}/{emotion}/*.wav"):
            target['emotion'].append(emotion)
            target['path'].append(file)

        n_samples = len(target['path'])
        if verbose:
            print(f"[{dataset_name}] Total files to write:({emotion}) ", n_samples)

        if verbose:
            print(f"[{dataset_name}] Number of samples:", n_samples)
            
        X.extend(target['path'])
        y.extend(target['emotion'])

    file_name = f"{dataset_name}.csv"
    pd.DataFrame({"path": X, "emotion": y}).to_csv(file_name)

if __name__ ==  '__main__':
    for dataset in para.datasets:
        write_csv(emotions=para.emotions, dataset_name=dataset, verbose=para.verbose)
