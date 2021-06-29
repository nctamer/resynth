import os
import re
import glob
import random
import shutil
import numpy as np
import essentia.standard as estd
from scipy.io.wavfile import write
from tqdm import tqdm
from pathlib import Path

from scipy.signal import get_window

from PredominantMelodyMakam import PredominantMelodyMakam
from synthesis import Synthesizer
from pitch_track_processing import PitchProcessor

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

#from pysndfx import AudioEffectsChain

np.random.seed(280490)

class CarnaticMelodySynthesis(object):
    def __init__(self, hop_size=128, frame_size=2048, mixing_weights=None,
                 data_path=os.path.join(Path().absolute(), 'resources', 'tmp_clean_dataset'),
                 output_dataset_path=os.path.join(Path().absolute(), 'resources', 'Saraga-Synth-Dataset', 'experiments'),
                 output_home=os.path.join(Path().absolute(), 'resources', 'output')):

        self.hop_size = hop_size  # default hopSize of PredominantMelody
        self.frame_size = frame_size  # default frameSize of PredominantMelody
        
        self.mixing_weights = mixing_weights

        self.data_path = data_path  # Path where clean chunks are
        self.output_dataset_path = output_dataset_path  # Path to store the saraga synth dataset
        self.output_home = output_home  # Output folder for random outputs

        self.sample_rate = 44100  # The standard sampling frequency for Saraga audio
        
    def get_dataset(self, pitch_preproc=True, voicing=False):
        # Clean dataset folder
        if not os.path.exists(self.output_dataset_path):
            # Create dataset folder
            os.mkdir(self.output_dataset_path)
            os.mkdir(os.path.join(self.output_dataset_path, 'audio'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'melody'))
            os.mkdir(os.path.join(self.output_dataset_path, 'annotations', 'activations'))

        # Get vocal audio paths
        total_vocal_tracks = glob.glob(os.path.join(self.data_path, '*.wav'))
        print(len(total_vocal_tracks))
        total_vocal_tracks.sort(key=natural_keys)
        total_ids = [x.split('/')[-1].replace('.wav', '') for x in total_vocal_tracks]
        computed_vocal_paths = glob.glob(os.path.join(self.output_dataset_path, 'audio', 'synth_mix_*'))
        computed_ids = [x.split('/')[-1].replace('synth_mix_', '').replace('.wav', '') for x in computed_vocal_paths]
        print(len(computed_vocal_paths))
        print(len(computed_ids))
        
        #if not computed_vocal_paths:
        #    last_track_computed = 0
        #else:
        #    computed_vocal_paths.sort(key=natural_keys)
        #    last_track_computed = int(computed_vocal_paths[-1].split('/')[-1].replace('synth_mix_', '').replace('.wav', ''))
        #computed_ids = [x.split('/')[-1].replace('synth_mix_', '').replace('.wav', '') for x in computed_vocal_paths]

        remaining_ids = [x for x in total_ids if x not in computed_ids]
        remaining_vocal_paths = [
            self.data_path + '/' + x + '.wav' for x in remaining_ids
        ]
        
        print(len(remaining_vocal_paths))
        
        voice_cleaner = Separator('spleeter:2stems')
        
        # Iterate over remaining tracks to synthesize
        for track in tqdm(remaining_vocal_paths[:-1]):
            _, _, _ = carnatic_synthesizer.generate_synthesized_mix(
                filename=track,
                separator=voice_cleaner,
                pitch_preproc=pitch_preproc,
                voicing=voicing
            )
        
    def generate_synthesized_mix(self, filename, separator, pitch_preproc, voicing):
        # Get file id from filename
        file_id = filename.split('/')[-1].replace('.wav', '')
        
        # Load audio with Spleeter's AudioAdapter
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(
            filename,
            sample_rate=self.sample_rate
        )

        # Run vocal separation on vocal audio
        #prediction = separator.separate(waveform)
        #audio = prediction['vocals']
        audio = waveform
        
        # To mono, energy filering and apply EqualLoudness for a better pitch extraction
        audio_mono = audio.sum(axis=1) / 2
        audio_mono_filt = self.filter_audio(audio=audio_mono, coef=0.00125)  # Energy filter to remove background noise
        audio_mono_eqloud = estd.EqualLoudness(sampleRate=self.sample_rate)(audio_mono_filt)
        
        # Extract pitch using PredominantMelodyMakam algorithm
        est_time, est_freq = self.extract_pitch_pmm(audio=audio_mono_eqloud)
        pitch = [[x, y] for x, y in zip(est_time, est_freq)]

        # Preprocessing analyzed audio and pitch
        preprocessor = PitchProcessor(
            pitch_preproc=pitch_preproc,
            voicing=voicing,
            gap_len=25,
        )
        audio, pitch_processed, time_stamps_processed = preprocessor.pre_processing(
            audio=audio_mono,
            extracted_pitch=pitch,
        )
        
        # Get freq limits to compute minf0
        tmp_est_freq = [x for x in est_freq if x > 20]
        if len(tmp_est_freq) > 0:
            minf0 = min(tmp_est_freq) - 20
        else:
            minf0 = 0
            
        # Synthesize vocal track
        synthesizer = Synthesizer(
            model='hpr',
            minf0=minf0,
            maxf0=max(pitch_processed) + 50,
        )
        synthesized_audio, pitch_track = synthesizer.synthesize(
            filtered_audio=audio,
            pitch_track=pitch_processed,
        )

        # Equalize voice
        #fx = (AudioEffectsChain().equalizer(200))
        #synthesized_audio = fx(synthesized_audio)

        # Get synthesized mix
        synthesized_audio_mix = self.mix(
            filename=filename,
            synthesized_voice=synthesized_audio
        )
        
        # Get vocal activations
        start_times, end_times = self.get_activations(time_stamps_processed, pitch_track)
        
        if len(start_times) > 2:
            # Write synthesized audio to file
            tmp_wav = 'audio/synth_mix_' + file_id + '.wav'
            self.save_audio_to_dataset(tmp_wav, synthesized_audio_mix)
    
            # Write csv melody annotation to file
            tmp_txt = 'annotations/melody/synth_mix_' + file_id + '.csv'
            self.save_pitch_track_to_dataset(tmp_txt, time_stamps_processed, pitch_track)
            
            # Write lab activations to file
            tmp_lab = 'annotations/activations/synth_mix_' + file_id + '.lab'
            self.save_activation_to_dataset(tmp_lab, start_times, end_times)
    
            return synthesized_audio_mix, pitch_track, time_stamps_processed
        else:
            print('UNVOICED TRACK! Skipping...')
            return [], [], []
        
    def mix(self, filename, synthesized_voice):
        # Get instrument lineup
        filename_violin = filename.replace("vocal.wav", "violin.wav")
        filename_mridangam_right = filename.replace("vocal.wav", "mridangam_right.wav")
        filename_mridangam_left = filename.replace("vocal.wav", "mridangam_left.wav")
        filename_tanpura = filename.replace("vocal.wav", "tanpura.wav")

        # Load audios and trim to synthesized voice length
        violin_mono = estd.MonoLoader(filename=filename_violin)()
        violin_mono_processed = np.array(violin_mono[:len(synthesized_voice) + 1], dtype='float64')
        violin_mono_processed_filt = self.filter_audio(audio=violin_mono_processed, coef=0.00075)
        mridangam_right_mono = estd.MonoLoader(filename=filename_mridangam_right)()
        mridangam_right_mono_processed = np.array(mridangam_right_mono[:len(synthesized_voice) + 1], dtype='float64')
        mridangam_right_mono_processed_filt = self.filter_audio(audio=mridangam_right_mono_processed, coef=0.001)
        mridangam_left_mono = estd.MonoLoader(filename=filename_mridangam_left)()
        mridangam_left_mono_processed = np.array(mridangam_left_mono[:len(synthesized_voice) + 1], dtype='float64')
        mridangam_left_mono_processed_filt = self.filter_audio(audio=mridangam_left_mono_processed, coef=0.001)
        tanpura_mono = estd.MonoLoader(filename=filename_tanpura)()
        tanpura_mono_processed = np.array(tanpura_mono[:len(synthesized_voice) + 1], dtype='float64')
        
        # Assign weights
        if self.mixing_weights:
            weight_voice = self.mixing_weights['voice']
            weight_violin = self.mixing_weights['violin']
            weight_mridangam_right = self.mixing_weights['mridangam_right']
            weight_mridangam_left = self.mixing_weights['mridangam_left']
            weight_tanpura = self.mixing_weights['tanpura']
        else:
            # Predefined weights in case no weight dict is provided
            weight_voice = 5.25
            weight_violin = 4
            weight_mridangam_right = 1
            weight_mridangam_left = 1
            weight_tanpura = 33.5

        # Get mix
        synthesized_audio_mix = [
            x*weight_voice +
            y*weight_violin +
            z*weight_mridangam_right +
            w*weight_mridangam_left +
            t*weight_tanpura for x, y, z, w, t in zip(
                synthesized_voice,
                violin_mono_processed_filt,
                mridangam_right_mono_processed_filt,
                mridangam_left_mono_processed_filt,
                tanpura_mono_processed
            )
        ]
        
        return synthesized_audio_mix
        
    def save_pitch_track_to_dataset(self, filename, est_time, est_freq):
        """
        Function to write txt annotation to file
        """
        pitchtrack_to_save = os.path.join(self.output_dataset_path, filename)
        with open(pitchtrack_to_save, 'w') as f:
            for i, j in zip(est_time, est_freq):
                f.write("{}, {}\n".format(i, j))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))
        
    def save_activation_to_dataset(self, filename, start_times, end_times):
        """
        Function to write lab activation annotation to file
        """
        activation_to_save = os.path.join(self.output_dataset_path, filename)
        with open(activation_to_save, 'w') as f:
            for i, j in zip(start_times, end_times):
                f.write("{}, {}, singer\n".format(i, j))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))

    def save_audio_to_dataset(self, filename, audio):
        """
        Function to write wav audio to file
        """
        audio_to_save = os.path.join(self.output_dataset_path, filename)
        write(audio_to_save, self.sample_rate, np.array(audio))
        print('{} saved with exit to {}'.format(filename, self.output_dataset_path))
        
    def filter_audio(self, audio, coef):
        """
        Code taken from Baris Bozkurt's MIR teaching notebooks
        """
        audio_modif = audio.copy()
        start_indexes = np.arange(0, audio.size - self.frame_size, self.hop_size, dtype=int)
        num_windows = start_indexes.size
        w = get_window('blackman', self.frame_size)
        energy = []
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + self.frame_size] * w
            energy.append(np.sum(np.power(x_win, 2)))
            
        for k in range(num_windows):
            x_win = audio[start_indexes[k]:start_indexes[k] + self.frame_size] * w
            energy_frame = np.sum(np.power(x_win, 2))
            if energy_frame < np.max(energy) * coef:
                audio_modif[start_indexes[k]:start_indexes[k] + self.frame_size] = np.zeros(self.frame_size)
                
        return audio_modif

    def extract_pitch_melodia(self, audio):
        # Running melody extraction with MELODIA
        pitch_extractor = estd.PredominantPitchMelodia(frameSize=self.frame_size, hopSize=self.hop_size)
        est_freq, _ = pitch_extractor(audio)
        est_freq = np.append(est_freq, 0.0)
        est_time = np.linspace(0.0, len(audio) / self.sample_rate, len(est_freq))

        return est_time, est_freq

    def extract_pitch_pmm(self, audio):
        # Running melody extraction with PMM
        pmm = PredominantMelodyMakam(hop_size=self.hop_size, frame_size=self.frame_size)
        output = pmm.run(audio=audio)

        # Organizing the output
        pitch_annotation = output['pitch']
        est_time = [x[0] for x in pitch_annotation]
        est_freq = [x[1] for x in pitch_annotation]

        return est_time, est_freq
    
    @staticmethod
    def get_activations(time_stamps, pitch_track):
        silent_zone_on = True
        start_times = []
        end_times = []
        for idx, value in enumerate(pitch_track):
            if value == 0:
                if not silent_zone_on:
                    end_times.append(time_stamps[idx-1])
                    silent_zone_on = True
            else:
                if silent_zone_on:
                    start_times.append(time_stamps[idx])
                    silent_zone_on = False
                    
        return start_times, end_times
    
    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Run python3 core.py to perform synthesis over input data
if __name__ == '__main__':

    mixing_weights = {
        'voice': 5.3,
        'violin': 3.75,
        'mridangam_right': 0.9,
        'mridangam_left': 0.9,
        'tanpura': 33.75,
    }
    
    carnatic_synthesizer = CarnaticMelodySynthesis(
        mixing_weights=mixing_weights,
        data_path="../data/original",
        output_dataset_path="../data/output_dataset_path",
        output_home="../data/output_home"
    )
    carnatic_synthesizer.get_dataset(
        pitch_preproc=True,
        voicing=False,
    )
