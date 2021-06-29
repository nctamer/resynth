import os
from scipy.io import wavfile
import numpy as np
import sys
sys.path.append(os.path.abspath('../windowed/'))
import crepe

if __name__ == '__main__':
    raw_file_path = "data/original"
    pitch_track_path = "data/pitch_tracks"
    model = "crepe_viterbi"
    for file in sorted(os.listdir(raw_file_path)):
        sr, audio = wavfile.read(os.path.join(raw_file_path, file))
        print(file, sr, len(audio))
        if model == "crepe":
            time, frequency, confidence, activation = crepe.predict(audio, sr, step_size=10, verbose=1)
            data = np.vstack([time, frequency, confidence]).T
        if model == "crepe_viterbi":
            time, frequency, confidence, activation = crepe.predict(audio, sr, step_size=10, verbose=1, viterbi=True)
            data = np.vstack([time, frequency, confidence]).T
        elif model == "dilated":
            data = audio
        elif model == "dilated_viterbi":
            MODEL = "/home/nazif/PycharmProjects/models/dilated2048_Jun23_16/dilated2048"

            # cr = CREPE().cuda() #no cuda for debug
            cr = CREPE().cpu()
            cr.load_weight(MODEL)
        else:
            data = audio

        np.save(os.path.join(pitch_track_path, file[:-3]+model+".npy"), data)
