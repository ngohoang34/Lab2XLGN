import sys
import os
import queue
import math
import threading
from pynput import keyboard

from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk

import librosa
import sounddevice as sd
import soundfile as sf

CLASS_LABELS = {"la", "nguoi", "cua", "giadinh", "co"}

# Controls
STOP_RECORD_CMD = "/s"
RECORD_CMD = "/r"

inputQueue = queue.Queue()

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix


def read_kb_input(inputQueue):
    while True:
        input_str = input()
        inputQueue.put(input_str)

# Indexing functions
def writeIndex(file, sentence, file_name):
    file.write(file_name + "\n")
    file.write(sentence  + "\n")

# Recording functions
SAMPLE_RATE = 22050
CHANNELS = 2

q = queue.Queue()

def callback( indata, frames, time, status):
    #This is called (from a separate thread) for each audio block.
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def record(file_name):
    try:
        #Open a new soundfile and attempt recording
        with sf.SoundFile(file_name, mode='x', samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_24") as file:
            with sd.InputStream(samplerate=SAMPLE_RATE, device=sd.default.device, channels=CHANNELS, callback=callback):
                print("Recording ... ('{}' to stop recording)".format(STOP_RECORD_CMD))
            
                while True:
                    file.write(q.get())

                    if (inputQueue.qsize() > 0):
                        input_str = inputQueue.get()
                        if (input_str == STOP_RECORD_CMD):
                            break

                print("Saved to: {}\n".format(file_name))

    except Exception as e:
        print(e)

# Kmeans
def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    return kmeans  

if __name__ == "__main__":
    models = {}
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "rb") as file: models[label] = pk.load(file)

    input("Press any key to start recording")

    inputThread = threading.Thread(target=read_kb_input, args=(inputQueue,), daemon=True)
    inputThread.start()

    record("live_recording.wav")

    sound_mfcc = get_mfcc("live_recording.wav")

    os.remove("live_recording.wav")

    kmeans = clustering(sound_mfcc)
    sound_mfcc = kmeans.predict(sound_mfcc).reshape(-1,1)

    evals = {cname : model.score(sound_mfcc, [len(sound_mfcc)]) for cname, model in models.items()}
    cmax = max(evals.keys(), key=(lambda k: evals[k]))
    print(evals)
    print("Conclusion: " + cmax)