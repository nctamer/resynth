import sys
import os
import numpy as np
from pathlib import Path
from scipy.signal import get_window
from scipy.io import wavfile
# Append sms tools model folder to sys.path
if os.path.join(Path().absolute(), 'sms_tools', 'models') not in sys.path:
    sys.path.append(os.path.join(Path().absolute(), 'sms_tools', 'models'))
from sms_tools.models import hpsModel as HPS
from sms_tools.models import hprModel as HPR
from sms_tools.models import sineModel as SM
from sms_tools.models.utilFunctions import refinef0Twm
import librosa
import soundfile as sf
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import pickle as pkl


np.set_printoptions(threshold=sys.maxsize)
class Synthesizer(object):
    def __init__(self, model='hpr', window='hanning', M=1001, N=4096, Ns=512, H=128, t=-90,
                 minSineDur=0.001, nH=30, maxf0=1760, minf0=55, f0et=5.0, harmDevSlope=0.001, stocf=0.1):

        # Model to use
        self.model = model

        # Synthesis parameters
        self.window = window
        self.M = M
        self.N = N
        self.Ns = Ns
        self.H = H
        self.t = t
        self.minSineDur = minSineDur
        self.nH = nH
        self.maxf0 = maxf0
        self.minf0 = minf0
        self.f0et = f0et
        self.harmDevSlope = harmDevSlope
        self.stocf = stocf
        self.sample_rate = 44100  # The standard sampling frequency for Saraga audios

    def get_parameters(self):
        return {
            'window': self.window,
            'M': self.M,
            'N': self.N,
            'Ns': self.Ns,
            'H': self.H,
            't': self.t,
            'minSineDur': self.minSineDur,
            'nH': self.nH,
            'minf0': self.minf0,
            'maxf0': self.maxf0,
            'f0et': self.f0et,
            'harmDevSlope': self.harmDevSlope,
            'stocf': self.stocf,
        }

    def analyze(self, filtered_audio, pitch_track):

        # Get window for the stft
        w = get_window(self.window, self.M, fftbins=True)

        # Get harmonic content from audio using extracted pitch as reference
        hfreq, hmag, hphase, xr, len_f0 = HPR.hprModelAnal(
            x=filtered_audio,
            f0=pitch_track,
            fs=self.sample_rate,
            w=w,
            N=self.N,
            H=self.H,
            t=self.t,
            minSineDur=self.minSineDur,
            nH=self.nH,
            minf0=self.minf0,
            maxf0=self.maxf0,
            f0et=self.f0et,
            harmDevSlope=self.harmDevSlope,
        )

        new_pitch = np.zeros_like(pitch_track)
        f0error = np.inf * np.ones_like(pitch_track)
        for ind, f0c in enumerate(pitch_track):
            if f0c > 0:
                pfreq = hfreq[ind]
                pmag = hmag[ind]
                new_pitch[ind], f0error[ind] = refinef0Twm(pfreq, pmag, f0c=f0c, refinement_range_cents=20)
        return {"f0": {"old": pitch_track, "new": new_pitch, "error": f0error},
                "harmonic": {"freq": hfreq, "mag": hmag, "phase": hphase},
                "residual": xr}


if __name__ == '__main__':
    raw_file_path = "/home/nazif/PycharmProjects/data/Kreutzer-violin-etudes/audio"
    save_file_path = "/home/nazif/PycharmProjects/data/Kreutzer-violin-etudes/analyzed"
    pitch_track_path = "/home/nazif/PycharmProjects/data/Kreutzer-violin-etudes/annotation"

    hop_size = 128
    sr = 44100

    for file in sorted(os.listdir(raw_file_path)):
        audio = librosa.load(os.path.join(raw_file_path, file), sr=sr, mono=True)[0]
        f0 = pd.read_csv(os.path.join(pitch_track_path, file[:-3]+"f0.csv"))

        time = np.array(range(len(audio)//hop_size)) * (hop_size/sr)
        f0 = np.interp(time, f0.time.to_numpy(), f0.frequency.to_numpy())
        f0[f0 < 10] = 0  # interpolation might introduce odd frequencies
        # Get freq limits to compute minf0
        tmp_est_freq = [x for x in f0 if x > 20]
        if len(tmp_est_freq) > 0:
            minf0 = min(tmp_est_freq) - 20
        else:
            minf0 = 0

        # Synthesize vocal track
        synthesizer = Synthesizer(
            model='hpr',
            minf0=minf0,
            maxf0=max(f0) + 50
        )
        analyzed = synthesizer.analyze(filtered_audio=audio, pitch_track=f0)
        with open(os.path.join(save_file_path, file[:-3]+"analyzed.pickle"), "wb") as f:
            pkl.dump(analyzed, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(file)
