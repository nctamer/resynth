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
import librosa
import soundfile as sf


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

    def synthesize(self, filtered_audio, pitch_track, pitch_shift_cents):

        # Get window for the stft
        w = get_window(self.window, self.M, fftbins=True)

        # todo: modify using the w here!
        if self.model == 'hps':
            # Get harmonic content from audio using extracted pitch as reference
            hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(
                x=filtered_audio,
                f0=pitch_track,
                fs=self.sample_rate,
                w=w,
                N=self.N,
                H=self.H,
                t=self.t,
                nH=self.nH,
                minf0=self.minf0,
                maxf0=self.maxf0,
                f0et=self.f0et,
                harmDevSlope=self.harmDevSlope,
                minSineDur=self.minSineDur,
                Ns=self.Ns,
                stocf=self.stocf
            )

            # Synthesize audio with generated harmonic content
            y, _, _ = HPS.hpsModelSynth(
                hfreq,
                hmag,
                hphase,
                stocEnv,
                self.Ns,
                self.H,
                self.sample_rate
            )

            return y, pitch_track

        if self.model == 'hpr':
            # Get harmonic content from audio using extracted pitch as reference
            hfreq, hmag, hphase, xr = HPR.hprModelAnal(
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
            # Synthesize audio with generated harmonic content
            harmonic = SM.sineModelSynth(hfreq, hmag, hphase, self.Ns, self.H, self.sample_rate)

            # Synthesize audio with the shifted harmonic content
            alt_hfreq = hfreq * pow(2, (pitch_shift_cents/1200))
            alt_harmonic = SM.sineModelSynth(alt_hfreq, hmag, hphase, self.Ns, self.H, self.sample_rate)

            sz = min(harmonic.size, alt_harmonic.size, xr.size)
            return np.array(harmonic[:sz],
                            dtype='float64'), np.array(alt_harmonic[:sz],
                                                       dtype='float64'), np.array(xr[:sz],
                                                                                  dtype='float64'), pitch_track


if __name__ == '__main__':
    raw_file_path = "data/original"
    save_file_path = "data/synthesized"
    pitch_track_path = "data/pitch_tracks"
    model = "crepe_viterbi"
    for file in sorted(os.listdir(raw_file_path))[::-1]:
        audio = librosa.load(os.path.join(raw_file_path, file), sr=44100, mono=True)[0]
        pitch = np.load(os.path.join(pitch_track_path, file[:-3]+model+".npy"))
        time_processed = range(int(np.floor(pitch[1, 0] / (128/44100))), 100+int(np.ceil(pitch[-1, 0] / (128/44100))))
        time_processed = np.array(time_processed) * (128/44100)
        pitch_processed = np.interp(time_processed, pitch[:, 0], pitch[:, 1])
        # Get freq limits to compute minf0
        tmp_est_freq = [x for x in pitch_processed if x > 20]
        if len(tmp_est_freq) > 0:
            minf0 = min(tmp_est_freq) - 20
        else:
            minf0 = 0

        # Synthesize vocal track
        synthesizer = Synthesizer(
            model='hpr',
            minf0=minf0,
            maxf0=max(pitch_processed) + 50
        )
        harmonic_audio, shifted_audio, residual_audio, new_pitch = synthesizer.synthesize(
            filtered_audio=audio,
            pitch_track=pitch_processed,
            pitch_shift_cents=400
        )
        sf.write(os.path.join(save_file_path, "shift_" + model + "_" + file), shifted_audio, 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "major3_" + model + "_" + file), shifted_audio+harmonic_audio, 44100,
                 'PCM_24')
        sf.write(os.path.join(save_file_path, "shiftPlusRes_" + model + "_" + file), shifted_audio+residual_audio,
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "synth_" + model + "_" + file), harmonic_audio+residual_audio,
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "suppressed_" + model + "_" + file), harmonic_audio+(0.2*residual_audio),
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "harm_" + model + "_" + file), harmonic_audio, 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "res_" + model + "_" + file), residual_audio, 44100, 'PCM_24')
        with open(os.path.join(save_file_path, "harm_" + model + "_" + file[:-3] + "txt"), "w") as annotation:
            for i in range(len(new_pitch)):
                annotation.write(str(time_processed[i]) + " " + str(new_pitch[i]) + "\n")
        print(file)
