#!/usr/bin/env python3
# Author: Jan Cuhel
# Date: 2.5.2021

import os

import gtts
import librosa
import numpy as np
import pickle
from pydub import AudioSegment
from pydub.playback import play

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import scipy
import speech_recognition as sr

# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import warnings
warnings.filterwarnings('ignore')

from extract_audio_features import extract_audio_features


# Audio constants
DURATION_RAVDESS = 3
DURATION_IEMOCAP = 11
SAMPLING_RATE = 16000
input_length_iemocap = SAMPLING_RATE * DURATION_IEMOCAP
input_length_ravdess = SAMPLING_RATE * DURATION_RAVDESS
DEFAULT_FILE = 'microphone-results.wav'
# TRILL models
SER_TRILL_MODEL_IEMOCAP = '/content/mer-thesis-app/result_models/ser_trill_lstm_iemocap_model.h5'
SER_TRILL_MODEL_RAVDESS = '/content/mer-thesis-app/result_models/ser_trill_lstm_ravdess_model.h5'
MER_ELECTRA_TRILL = '/content/mer-thesis-app/result_models/mer_trill_electra_small_model.h5'
# Yamnet models
SER_YAMNET_MODEL_IEMOCAP = '/content/mer-thesis-app/result_models/ser_yamnet_iemocap_model.h5'
SER_YAMNET_MODEL_RAVDESS = '/content/mer-thesis-app/result_models/ser_yamnet_ravdess_model.h5'
MER_ELECTRA_YAMNET = '/content/mer-thesis-app/result_models/mer_electra_yamnet_iemocap_model.h5'
# TER Electra
TER_ELECTRA_IEMOCAP = '/content/mer-thesis-app/result_models/ter_electra_iemocap_model.h5'
TER_ELECTRA_PSYCHEXP = '/content/mer-thesis-app/result_models/ter_electra_model_psychexp.h5'
# Emotion available in datasets
emotions_iemocap = ['neutral', 'happy', 'sad', 'angry']
emotions_ravdess = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emotion_psychexp = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
# Language of the models
LANG='en'
# URL addresses for the audio embeddings
TRILL_URL = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3'
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'


class DeepLearningModel():
    """ Definition of a class for DeepLearning Emotion Recognition model """
    def __init__(self, model_filename, emotions=emotions_iemocap):
        self.model_filename = model_filename
        self.emotions = emotions
        self.model = self.load_model()
    
    def load_model(self):
        """ Loads the model from TF Hub """
        return tf.keras.models.load_model(
            self.model_filename, custom_objects={'KerasLayer':hub.KerasLayer})


class TERModel(DeepLearningModel):
    """ Definition of a class for Text Emotion Recognition model (TER) """
    def __init__(self, model_filename, emotions=emotions_iemocap):
        super().__init__(model_filename, emotions)

    def predict_emotion(self, text):
        """ Predicts an emotion of the given text """
        X_text = np.array([text])

        # Make prediction    
        pred_id = tf.argmax(self.model.predict(X_text), 1).numpy()[0]

        return self.emotions[pred_id]


class SERModel(DeepLearningModel):
    """ Definition of a class for Speech Emotion Recognition model (SER) """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, emotions)
        self.input_length = input_length
        self.embedding = hub.load(embedding_url)
        self.sample_rate = sample_rate

    def load_model(self):
        """ Loads the model """
        return tf.keras.models.load_model(self.model_filename)
    
    def predict_emotion(self, audio_file):
        """ Predicts an emotion of the given audio file """
        y, _ = librosa.load(audio_file, sr=self.sample_rate)
        # y,_ = librosa.effects.trim(y, top_db = 25)
        # https://en.wikipedia.org/wiki/Wiener_filter
        # https://cs.wikipedia.org/wiki/Wiener%C5%AFv_filtr
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html
        y = scipy.signal.wiener(y)

        if len(y) > self.input_length:
            # Cut to the same length 
            y = y[0:self.input_length]
        elif self.input_length > len(y):
            # Pad the sequence
            max_offset = self.input_length - len(y)  
            y = np.pad(y, (0, max_offset), "constant")

        X_audio = self.get_audio_embedding(y)

        # Make prediction
        pred_id = tf.argmax(self.model.predict(X_audio), 1).numpy()[0]

        return self.emotions[pred_id]

    def get_audio_embedding(self, audio):
        return np.array([audio])


class TRILLSERModel(SERModel):
    """ 
    Definition of a class for Speech Emotion Recognition model (SER) that
    uses TRILL Embedding
    """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, embedding_url, emotions, input_length, sample_rate)

    def get_audio_embedding(self, audio):
        return np.array([self.embedding(samples=audio, sample_rate=self.sample_rate)['embedding'].numpy()])


class YAMNetSERModel(SERModel):
    """ 
    Definition of a class for Speech Emotion Recognition model (SER) that
    uses YAMNet as an Embedding
    """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, embedding_url, emotions, input_length, sample_rate)

    def get_audio_embedding(self, audio):
        # Get the embedding from the yamnet
        _, embeddings, _ = self.embedding(audio)
        return np.array([embeddings.numpy()])


class MERModel(DeepLearningModel):
    """ Definition of a class for Multimodal Emotion Recognition model (MER) """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, emotions)
        self.input_length = input_length
        self.embedding = hub.load(embedding_url)
        self.input_length = input_length
        self.sample_rate = sample_rate
    
    def predict_emotion(self, text, audio_file):
        """ Predicts an emotion of the given text and audio file """
        y, _ = librosa.load(audio_file, sr=self.sample_rate)
        # y,_ = librosa.effects.trim(y, top_db = 25)
        # https://en.wikipedia.org/wiki/Wiener_filter
        # https://cs.wikipedia.org/wiki/Wiener%C5%AFv_filtr
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html
        y = scipy.signal.wiener(y)

        if len(y) > self.input_length:
            # Cut to the same length 
            y = y[0:self.input_length]
        elif self.input_length > len(y):
            # Pad the sequence
            max_offset = self.input_length - len(y)  
            y = np.pad(y, (0, max_offset), "constant")

        X_audio = self.get_audio_embedding(y)

        X_text = np.array([text])

        # Make prediction
        pred_id = tf.argmax(self.model.predict([X_text, X_audio]), 1).numpy()[0]

        return self.emotions[pred_id]
    
    def get_audio_embedding(self, audio):
        return np.array([audio])


class ElectraTRILLMERModel(MERModel):
    """ 
    Definition of a class for Multimodal Emotion Recognition model (MER) that
    uses TRILL Embedding
    """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, embedding_url, emotions, input_length, sample_rate)

    def get_audio_embedding(self, audio):
        return np.array([self.embedding(samples=audio, sample_rate=self.sample_rate)['embedding'].numpy()])


class ElectraYAMNetMERModel(MERModel):
    """ 
    Definition of a class for Multimodal Emotion Recognition model (MER) that
    uses YAMNet as an Embedding
    """
    def __init__(self, model_filename, embedding_url, emotions=emotions_iemocap, input_length=input_length_iemocap, sample_rate=SAMPLING_RATE):
        super().__init__(model_filename, embedding_url, emotions, input_length, sample_rate)

    def get_audio_embedding(self, audio):
        # Get the embedding from the yamnet
        _, embeddings, _ = self.embedding(audio)
        return np.array([embeddings.numpy()])        


def record_speech(lang=LANG, dur=DURATION_IEMOCAP, filepath=DEFAULT_FILE):
    """ 
    This function records a speech from a microphone and get the text. 

    params:
    - lang: the language of the recorded speach
    - dur: how long in seconds should the function record
    - filepath: path to the file where should be the audio recording saved
    returns:
    - text: transcript of the audio recording
    - filepath: where was the audio recording saved
    """
    # initialize the recognizer
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print(f'Starting recording for the next {dur}s.\nPlease speak...')
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=dur)
            print("Recording ended.\nRecognizing...")
            # convert speech to text
            text = r.recognize_google(audio_data, language=lang)
            print('Done.')

            print(f'\nYou\'ve said {text}.\n')

            # write audio to a WAV file
            with open(filepath, "wb") as f:
                f.write(audio_data.get_wav_data())

            print('Done.')

            return text, filepath
    except:
        print('Something went wrong... Try to speak again')
    
    return None, None
