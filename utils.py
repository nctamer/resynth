import os
from scipy.io import wavfile
import numpy as np
import librosa
import pandas as pd
from scipy import interpolate
import crepe
from scipy.signal import get_window, medfilt
from pitchfilter import PitchFilter
import sys
from pathlib import Path
if os.path.join(Path().absolute(), 'sms_tools', 'models') not in sys.path:
    sys.path.append(os.path.join(Path().absolute(), 'sms_tools', 'models'))
from sms_tools.models import hprModel as HPR
from sms_tools.models import sineModel as SM
from sms_tools.models.utilFunctions import refinef0Twm
import soundfile as sf

HOP_SIZE = 128
SAMPLING_RATE = 44100

def silence_unvoiced_segments(pitch_track_csv, confidence_threshold=0.08, min_voiced_segment_ms=12):
    """
    Accepts crepe output in the csv format and removes unvoiced segments with confidence and accepted voiced segment duration
    :param pitch_track_csv: csv with [ºtimeº, ºfrequencyº, ºconfidenceº] fields
    :param confidence_threshold: confidence threshold in range (0,1)
    :param min_voiced_segment_ms: voiced segments shorter than the specified lenght are discarded
    :return: input csv file with the silenced segments
    """
    annotation_interval_ms = 1000*pitch_track_csv.loc[:1, "time"].diff()[1]
    voiced_th = int(np.ceil(min_voiced_segment_ms)/annotation_interval_ms)
    conf_bool = np.array(pitch_track_csv["confidence"]>confidence_threshold).reshape(-1)
    absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    segment_durs = np.diff(ranges,axis=1)
    valid_segments = ranges[np.repeat(segment_durs>voiced_th, repeats=2, axis=1)].reshape(-1, 2)
    voiced = np.zeros(len(pitch_track_csv), dtype=bool)
    for segment in valid_segments:
        voiced[segment[0]:segment[1]] = True
    pitch_track_csv.loc[~voiced, "frequency"] = 0
    return pitch_track_csv

def interpolate_f0_to_sr(pitch_track_csv, audio, sr=SAMPLING_RATE, hop_size=HOP_SIZE):
    f = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["frequency"],
                             kind="cubic", fill_value="extrapolate")
    time = np.array(range(int(np.ceil(len(audio)/hop_size)))) * (hop_size/sr)
    pitch_track_np = f(time)
    pitch_track_np[pitch_track_np < 10] = 0  # interpolation might introduce odd frequencies
    return pitch_track_np

def hpr_anal(audio, f0, hop_size=HOP_SIZE, sr=SAMPLING_RATE):
    # Get harmonic content from audio using extracted pitch as reference
    # Get freq limits to compute minf0
    tmp_est_freq = [x for x in f0 if x > 20]
    if len(tmp_est_freq) > 0:
        minf0 = min(tmp_est_freq) - 20
    else:
        minf0 = 0

    w = get_window('hanning', 1001, fftbins=True)
    f0et = 10.0
    f0_refinement_range_cents = 10
    hfreq, hmag, hphase, xr, len_f0 = HPR.hprModelAnal(
        x=audio,
        f0=f0,
        fs=sr,
        w=w,
        minf0=minf0,
        maxf0=max(f0) + 50,
        H=hop_size,
        N=2048,
        f0et=f0et,
        t=-90,
        nH=30,
        harmDevSlope=0.001,
        minSineDur=0.001
    )
    return hfreq, hmag, hphase, xr

def refine_harmonics_twm(hfreq, hmag, f0, f0et=10.0, f0_refinement_range_cents=10):
    """
    Refine the f0 estimate with the help of two-way mismatch algorithm and change the harmonic components
    to the exact multiples of the refined f0 estimate
    :param hfreq: analyzed harmonic frequencies
    :param hmag: analyzed magnitudes
    :param f0: f0 in Hz before TWM
    :param f0et: error threshold for the TWM
    :param f0_refinement_range_cents: the range to be explored in TWM
    :return: new synthesis parameters
    """
    for frame, f0_frame in enumerate(f0):
        if f0_frame > 0: # for the valid frequencies
            pfreq = hfreq[frame]
            pmag = hmag[frame]
            f0_twm, f0err_twm = refinef0Twm(pfreq, pmag, f0_frame, refinement_range_cents=f0_refinement_range_cents)
            if f0err_twm < f0et:
                hfreq[frame] = f0_twm * np.round(pfreq/f0_twm)
                f0[frame] = f0_twm
            else:
                f0[frame] = 0
                hfreq[frame] = 0
                hmag[frame] = 0
    return hfreq, hmag, f0

def apply_pitch_filter(pitch_track_csv, min_chunk_size=20):
    pitch_filter = PitchFilter(min_chunk_size=min_chunk_size)
    pitch_track_np = pitch_filter.filter(np.array(pitch_track_csv))
    return pd.DataFrame(data=pitch_track_np, columns=pitch_track_csv.columns)

if __name__ == '__main__':
    names = ["Kreutzer42-CihatAskin"]
    dataset_folder = "/home/nazif/Documents/violindataset/singlevoice"
    for name in names:
        main_path = os.path.join(dataset_folder, name)
        raw_file_path = os.path.join(main_path, "audio")
        save_file_path = os.path.join(main_path, "synth_v2")
        pitch_track_path = os.path.join(main_path, "annotation")
        for file in sorted(os.listdir(raw_file_path)):
            try:
                audio = librosa.load(os.path.join(raw_file_path, file), sr=SAMPLING_RATE, mono=True)[0]
                f0s = pd.read_csv(os.path.join(pitch_track_path, file[:-3] + "f0.csv"))
                f0s = silence_unvoiced_segments(f0s, confidence_threshold=0.08, min_voiced_segment_ms=12)
                # f0s = apply_pitch_filter(f0s, min_chunk_size=6)
                f0s = interpolate_f0_to_sr(f0s, audio)
                hfreqs, hmags, hphases, xr = hpr_anal(audio, f0s)
                hfreqs, hmags, f0s = refine_harmonics_twm(hfreqs, hmags, f0s, f0et=10.0, f0_refinement_range_cents=10)
                print("analysis:", file)
                harmonic_audio = SM.sineModelSynth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
                sf.write(os.path.join(save_file_path, file), harmonic_audio, 44100, 'PCM_24')
                np.savetxt(os.path.join(save_file_path, file + "f0.txt"), f0s)
                print("synthesis:", file)
            except:
                print("ERROR IN FILE: ", file)

