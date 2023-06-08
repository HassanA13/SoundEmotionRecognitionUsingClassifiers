# SER-Speech-Emotion-Recognition
## Introduction
- The purpose of this repository is to develop and train a system for Speech Emotion Recognition, which involves building and testing machine learning and deep learning algorithms capable of detecting human emotions from speech.

- The potential applications of this tool are numerous and span across various industries, including affective computing, product recommendations, and more.
### Emotions available
There are 9 emotions available: "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise) and "boredom".
## Features
### Telegram
integration with telegram: you can record your voice using telegram bot and the result of detected emotion will be sent by the bot; keeping in mind that the code must be running.
### Classifiers
I used a Confusion Matrix to determine the best classifier.
- MLPClassifier
- GradientBoostingClassifier
- SVC
- RandomForestClassifier
- KNeighborsClassifier
- BaggingClassifier
- AdaBoostClassifier
### Hyperparameter
I used RandomizedSearchCV Algorithm to select the best hyperparameter for each Classifiers
### Features Selection
After extraction for features and blend them together in this case the input for the classifier become big so I use CatBoost algorithm to chose the most important features.

## Requirements
- **Python 3.6+**
### Python Packages
- **scikit-learn**
- **librosa**
- **numpy**
- **pandas**
- **soundfile**
- **tqdm**
- **matplotlib**
- **pyaudio**
- **telebot**
- **pickled**
- **[ffmpeg](https://ffmpeg.org/)**: used if you want to add more sample audio by converting to 16000Hz sample rate and mono channel which is provided in ``convert_wavs.py``

### Dataset
This repository used 4 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).
- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.
- **Custom** : Some unbalanced noisy dataset that is located in `data/train-custom` for training and `data/test-custom` for testing in which you can add/remove recording samples easily by converting the raw audio to 16000 sample rate, mono channel (this is provided in `create_wavs.py` script in ``convert_audio(audio_path)`` method which requires [ffmpeg](https://ffmpeg.org/) to be installed and in *PATH*) and adding the emotion to the end of audio file name separated with '_' (e.g "20190616_125714_happy.wav" will be parsed automatically as happy)


## Feature Extraction
Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

In this repository, we have used the most-used features that are available in [librosa](https://github.com/librosa/librosa) library, including:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- SSC
- Chroma 
- MelSpectrogram
- Chromagram 
- Contrast
- Tonnetz (tonal centroid features)
- F0
