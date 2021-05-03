import numpy as np
import librosa # To extract speech features
import glob
import os


# Extract feature function
def extract_audio_features(file_name, should_augment=False, **kwargs):
  """
  Extract feature from audio file `file_name`
    Features supported:
     - MFCC (mfcc)
     - Chroma (chroma)
     - MEL Spectrogram Frequency (mel)
    e.g:
      `features = extract_audio_features(path, mel=True, mfcc=True)`
  """
  mfcc = kwargs.get("mfcc")
  chroma = kwargs.get("chroma")
  mel = kwargs.get("mel")

  # https://stackoverflow.com/questions/9458480/read-mp3-in-python-3
  # https://librosa.org/doc/latest/tutorial.html#quickstart
  # https://github.com/librosa/librosa/issues/1015
  X, sample_rate = librosa.load(file_name)
  if chroma:
    stft = np.abs(librosa.stft(X))
  result = np.array([])
  if mfcc:
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    # print('mfccs shape', mfccs.shape)
  if mel:
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result = np.hstack((result, mel))
    # print('mel shape', mel.shape)
  if chroma:
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result = np.hstack((result, chroma))
    # print('chroma shape', chroma.shape)
  return result

