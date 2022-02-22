import sys
import os
import numpy as np
from pathlib import Path
from scipy.signal import get_window, medfilt
from scipy import interpolate
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
from pitchfilter import PitchFilter

np.set_printoptions(threshold=sys.maxsize)


class Synthesizer(object):
    def __init__(self, model='hpr', window='hanning', M=1001, N=4096, Ns=512, H=128, t=-90,
                 minSineDur=0.001, nH=30, maxf0=3600, minf0=180, f0et=5.0, harmDevSlope=0.001, stocf=0.1):

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
        max_len = len(hfreq)
        for ind, f0c in enumerate(pitch_track):
            if (f0c > 0) & (ind < max_len):
                pfreq = hfreq[ind]
                pmag = hmag[ind]
                new_pitch[ind], f0error[ind] = refinef0Twm(pfreq, pmag, f0c=f0c, refinement_range_cents=50)
        return {"f0": {"old": pitch_track, "new": new_pitch, "error": f0error},
                "harmonic": {"freq": hfreq, "mag": hmag, "phase": hphase},
                "residual": xr}

    def synthesize(self, pitch_track, freq, mag, phase, pitch_shift_cents=None, apply_valid_check=False, mu=None, dist=0.8852633350795827):

        pitch_track = pitch_track[:len(freq)]
        freq[pitch_track == 0, :] = 0
        mag[pitch_track == 0, :] = 0
        phase[pitch_track == 0, :] = 0

        if apply_valid_check:

            # refine the f0 prediction
            valid_min, valid_out_of = 7, 9
            hvalid = freq[:, :valid_out_of].astype(bool).sum(axis=1)
            hvalid = hvalid >= valid_min
            freq[~hvalid, :] = 0
            mag[~hvalid, :] = 0
            phase[~hvalid, :] = 0
            pitch_track = np.array(pitch_track[:len(hvalid)])
            pitch_track[~hvalid] = 0

            """
            if not mu:
                mu = np.array([1.,         0.91288472, 0.85301497, 0.81154242, 0.7546334,  0.69182445,
                               0.66007972, 0.60962449, 0.56290305, 0.52123842, 0.4787446,  0.44326154,
                               0.4091061,  0.37623347, 0.33984164])
            # cancel out the *invalid* harmonic content
            hmag_ptr = np.copy(mag)[:, :15] + 100
            hvalid = hmag_ptr[:, 0] > 0
            hmag_ptr = np.divide(hmag_ptr, hmag_ptr[:, 0][:, None], where=hvalid[:, None])
            instrument_dist = euclidean_distances(hmag_ptr, [mu]).reshape(-1)
            hvalid_instrument = instrument_dist < dist
            hvalid = np.logical_and(hvalid, hvalid_instrument)
            freq[~hvalid, :] = 0
            mag[~hvalid, :] = 0
            phase[~hvalid, :] = 0
            pitch_track = np.array(pitch_track[:len(hvalid)])
            pitch_track[~hvalid] = 0
            """


        if pitch_shift_cents:
            alt_pitch_track = pitch_track * pow(2, (pitch_shift_cents/1200))

            alt_hfreq = freq * pow(2, (pitch_shift_cents/1200))
            alt_harmonic = SM.sineModelSynth(alt_hfreq, mag, np.array([]), self.Ns, self.H, self.sample_rate)

            return np.array(alt_harmonic, dtype='float64'), alt_pitch_track

        else:
            # Synthesize audio with the generated harmonic content
            harmonic = SM.sineModelSynth(freq, mag, phase, self.Ns, self.H, self.sample_rate)

            return np.array(harmonic, dtype='float64'), pitch_track


if __name__ == '__main__':

    parent = "/run/user/1000/gvfs/sftp:host=hpc.s.upf.edu/homedtic/ntamer/violindataset/singlevoice"

    names = ["MazasOp36-BernardChevalier",
             "DanclaOp84-BernardChevalier",
             "DanclaOp84-GiovanniMantovani",
             "KayserOp20-AlexandrosIakovou",
             "KayserOp20-FabricioValvasori",
             "KayserOp20-PatrickRafferty",
             "Kreutzer42-BochanKang",
             "Kreutzer42-CihatAskin",
             "WohlfahrtOp45-BernardChevalier",
             "WohlfahrtOp45-PatrickRafferty"]


    #pitch_filter = PitchFilter(min_chunk_size=40)
    for name in names:
        main_path = os.path.join(parent, name)

        raw_file_path = os.path.join(main_path, "audio")
        anal_file_path = os.path.join(main_path, "analyzed")
        synth_file_path = os.path.join(main_path, "synthesized")
        #synth_file_path = "synth"
        pitch_track_path = os.path.join(main_path, "annotation")

        hop_size = 128
        sr = 44100

        for file in sorted(os.listdir(raw_file_path)):
            audio = librosa.load(os.path.join(raw_file_path, file), sr=sr, mono=True)[0]
            f0 = pd.read_csv(os.path.join(pitch_track_path, file[:-3]+"f0.csv")).to_numpy()
            # apply  a 75ms median filter to remove discrepancies
            f0[:, 1] = medfilt(f0[:, 1], kernel_size=45)
            # apply pitch filter to remove octave errors
            #try:
            #    f0 = np.array(pitch_filter.filter(f0))
            #except:
            #    print("pitch filter error in ", file)
            # cubic interpolation for the change of time interval from 1ms to 2.9ms
            f = interpolate.interp1d(f0[:, 0], f0[:, 1], kind="cubic", fill_value="extrapolate")
            time = np.array(range(int(np.ceil(len(audio)/hop_size)))) * (hop_size/sr)
            f0 = f(time)
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
                maxf0=max(f0) + 50,
                H=hop_size,
                N=2048,
            )
            analyzed = synthesizer.analyze(filtered_audio=audio, pitch_track=f0)
            with open(os.path.join(anal_file_path, file[:-3]+"analyzed.pickle"), "wb") as f:
                pkl.dump(analyzed, f, protocol=pkl.HIGHEST_PROTOCOL)
            print(file, "analysis complete")
            try:
                error = analyzed["f0"]["error"]
                valid_bool = (error < 5) * (analyzed["f0"]["old"] > 0) * (analyzed["f0"]["new"] > 0)
                f0 = analyzed["f0"]["new"]
                f0[~valid_bool] = 0
                harmonic_audio, pitch_track = synthesizer.synthesize(f0, **analyzed["harmonic"])
                sf.write(os.path.join(synth_file_path, file), harmonic_audio, 44100, 'PCM_24')
                np.savetxt(os.path.join(synth_file_path, file[:-3] + "txt"), np.stack((time[:len(pitch_track)], 
                                                                                       pitch_track, 
                                                                                       error[:len(pitch_track)]), axis=0).T)
                print(file, "synth complete")
            except:
                print("ERROR IN SYNTH: ", file)

