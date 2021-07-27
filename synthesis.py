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

            """
            hfreq, hmag, hphase, xr, len_f0 = HPR.hprModel(
                x=filtered_audio,
                fs=self.sample_rate,
                w=w,
                N=self.N,
                t=self.t,
                nH=self.nH,
                minf0=self.minf0,
                maxf0=self.maxf0,
                f0et=self.f0et,
            )"""


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
            f0error = np.inf*np.ones_like(pitch_track)
            for ind, f0c in enumerate(pitch_track):
                if f0c > 0:
                    pfreq = hfreq[ind]
                    pmag = hmag[ind]
                    new_pitch[ind], f0error[ind] = refinef0Twm(pfreq, pmag, f0c=f0c, refinement_range_cents=20)

            # refine the f0 prediction
            valid_min, valid_out_of, average_over = 6, 10, 30
            hvalid = hfreq[:, :valid_out_of].astype(bool).sum(axis=1)
            hvalid = hvalid >= valid_min

            hmag_ptr = np.copy(hmag)[:, :valid_out_of]+100
            hmag_ptr = np.divide(hmag_ptr, hmag_ptr.sum(axis=1)[:, None], where=hvalid[:, None])
            instrument_mu = hmag_ptr[hvalid, :].sum(axis=0) / sum(hvalid)
            instrument_dist = euclidean_distances(hmag_ptr, [instrument_mu]).reshape(-1)
            hvalid_instrument = instrument_dist < 1.5*np.median(instrument_dist)
            hvalid = np.logical_and(hvalid, hvalid_instrument)

            # accept valid segments with the rule: m out of n harmonic peaks are seen at the first nf0 harmonics
            # todo: include a refinement step for hvalid so that we have segments with some min length
            hfreq_ptr = np.copy(hfreq[:, :average_over])  # only consider the first n harmonics for the f0 estimation
            hfreq_ptr[~hvalid, :] = 0

            # and average over the specified number of harmonics to detect the f0
            refined_pitch = hfreq_ptr * 1/np.array(range(1, hfreq_ptr.shape[1]+1))  # harmonic numbers 1, ..., avg_over
            hnum = refined_pitch.astype(bool).sum(axis=1)
            refined_pitch = np.divide(refined_pitch.sum(axis=1), hnum, where=hnum != 0)

            deviation_rate = pitch_track[hvalid]/refined_pitch[hvalid]
            deviation_cents = np.log2(deviation_rate)*1200
            print("deviation mu: {:.3f}c std: {:.3f}c, coverage: {:.2f}%".format(deviation_cents.mean(),
                                                                                 deviation_cents.std(),
                                                                                 100*sum(hvalid)/len(hvalid)))

            # cancel out the *invalid* harmonic content
            hfreq[~hvalid, :] = 0
            hmag[~hvalid, :] = 0
            hphase[~hvalid, :] = 0

            # Synthesize audio with the generated harmonic content
            harmonic = SM.sineModelSynth(hfreq, hmag, hphase, self.Ns, self.H, self.sample_rate)

            # Synthesize audio with the shifted harmonic content
            alt_hfreq = hfreq * pow(2, (pitch_shift_cents/1200))
            alt_harmonic = SM.sineModelSynth(alt_hfreq, hmag, np.array([]), self.Ns, self.H, self.sample_rate)

            sz = min(harmonic.size, alt_harmonic.size, xr.size)
            return np.array(harmonic[:sz], dtype='float64'), \
                   np.array(alt_harmonic[:sz], dtype='float64'), \
                   np.array(xr[:sz], dtype='float64'), pitch_track, refined_pitch


if __name__ == '__main__':
    raw_file_path = "data/original"
    save_file_path = "data/synthesized"
    pitch_track_path = "data/pitch_tracks"
    model = "crepe_viterbi"

    hop_size = 128
    sr = 44100

    for file in sorted(os.listdir(raw_file_path))[::-1]:
        audio = librosa.load(os.path.join(raw_file_path, file), sr=sr, mono=True)[0]
        f0 = np.load(os.path.join(pitch_track_path, file[:-3]+model+".npy"))

        time = np.array(range(len(audio)//hop_size)) * (hop_size/sr)
        f0 = np.interp(time, f0[:, 0], f0[:, 1])
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
        harmonic_audio, shifted_audio, residual_audio, old_pitch, new_pitch = synthesizer.synthesize(
            filtered_audio=audio,
            pitch_track=f0,
            pitch_shift_cents=300
        )
        old_pitch[old_pitch < 10] = 0
        new_pitch[new_pitch < 10] = 0
        sf.write(os.path.join(save_file_path, "shift_" + model + "_" + file), shifted_audio, 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "m3_" + model + "_" + file), shifted_audio+harmonic_audio, 44100,
                 'PCM_24')
        sf.write(os.path.join(save_file_path, "shiftPlusRes_" + model + "_" + file), shifted_audio+residual_audio,
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "synth_" + model + "_" + file), harmonic_audio+residual_audio,
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "suppressed_" + model + "_" + file), harmonic_audio+(0.2*residual_audio),
                 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "harm_" + model + "_" + file), harmonic_audio, 44100, 'PCM_24')
        sf.write(os.path.join(save_file_path, "res_" + model + "_" + file), residual_audio, 44100, 'PCM_24')
        with open(os.path.join(save_file_path, "old_" + model + "_" + file[:-3] + "txt"), "w") as annotation:
            for i in range(len(old_pitch)):
                annotation.write(str(time[i]) + " " + str(new_pitch[i]) + "\n")
        with open(os.path.join(save_file_path, "new_" + model + "_" + file[:-3] + "txt"), "w") as annotation:
            for i in range(len(new_pitch)):
                annotation.write(str(time[i]) + " " + str(new_pitch[i]) + "\n")
        print(file)
