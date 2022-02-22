import sys
import os
import numpy as np
from pathlib import Path
from scipy.signal import get_window
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
from anal import Synthesizer


names = ["DanclaOp84-BernardChevalier",
         "KayserOp20-FabricioValvasori",
         "KayserOp20-PatrickRafferty",
         "KayserOp20-AlexandrosIakovou"]
         #"Kreutzer42-CihatAskin",
         #"MazasOp36-BernardChevalier",
         #"WohlfahrtOp45-PatrickRafferty"]

store_patterns = np.zeros((0, 15))
n_total = 0
n_valid = 0

parent_folder = "/home/nazif/PycharmProjects/data"
synthesizer = Synthesizer(model='hpr')

for name in names:
    main_path = os.path.join(parent_folder, name)

    read_file_path = os.path.join(main_path, "analyzed")
    save_file_path = os.path.join(main_path, "synthesized")

    for file in sorted(os.listdir(read_file_path)):
        with open(os.path.join(read_file_path, file), "rb") as f:
            try:
                analyzed = pkl.load(f)
                error = analyzed["f0"]["error"]
                valid_bool = (error < 0.9) * (analyzed["f0"]["old"] > 0) * (analyzed["f0"]["new"] > 0)
                f0 = analyzed["f0"]["new"]
                f0[~valid_bool] = 0
                harmonic_audio, pitch_track = synthesizer.synthesize(f0, **analyzed["harmonic"])
                sf.write(os.path.join(save_file_path, file[:-16] + ".wav"), harmonic_audio, 44100, 'PCM_24')
                np.savetxt(os.path.join(save_file_path, file[:-16] + ".txt"), pitch_track)
                print(file)
            except:
                print("ERROR IN FILE: ", file)

