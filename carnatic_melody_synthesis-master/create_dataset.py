import os
import glob
import mirdata
import shutil
import essentia.standard as estd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io.wavfile import write

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

np.random.seed(280490)


class DatasetCreator(object):
    def __init__(self,
                 dataset_path=None,
                 tanpura_dataset_path=os.path.join(Path().absolute(), 'resources', 'synth_tanpura_dataset'),
                 chunks_path=os.path.join(Path().absolute(), 'resources', 'tmp_dataset'),
                 output_dataset_path=os.path.join(Path().absolute(), 'resources', 'tmp_clean_dataset')):
        
        self.dataset_path = dataset_path
        self.tanpura_dataset_path = tanpura_dataset_path
        self.chunks_path = chunks_path
        self.output_dataset_path = output_dataset_path
    
    def split_dataset_in_chunks(self):
        # Create output folder if it does not exist
        if not os.path.exists(self.chunks_path):
            os.mkdir(self.chunks_path)
        
        # Initialize Saraga Carnatic dataset and get list of multitrack audio subset
        saraga_carnatic = mirdata.initialize('saraga_carnatic', data_home=self.dataset_path)
        track_ids = saraga_carnatic.track_ids
        saraga_data = saraga_carnatic.load_tracks()
        concerts_to_ignore = ['Akkarai', 'Sundar']
        multitrack_list = self.get_multitrack_ids(track_ids, saraga_data, concerts_to_ignore)
        
        split_count = 0
        for track_id in tqdm(multitrack_list):
            
            # Get track to format
            track = saraga_data[track_id]
            
            # Get tonic rounded at 4 decimals
            tonic = round(self.get_tonic(track), 4)
            
            # Get tanpura audio from the synthesized tanpura dataset
            tanpura_filename = os.path.join(self.tanpura_dataset_path, 'tanpura_' + str(tonic) + '.wav')
            audio_tanpura = estd.MonoLoader(
                filename=tanpura_filename
            )()
            # Get voice
            audio_vocal = estd.MonoLoader(
                filename=track.audio_vocal_path
            )()
            # Get violin
            audio_violin = estd.MonoLoader(
                filename=track.audio_violin_path
            )()
            # Get mridangam right
            audio_mridangam_right = estd.MonoLoader(
                filename=track.audio_mridangam_right_path
            )()
            # Get mridangam left
            audio_mridangam_left = estd.MonoLoader(
                filename=track.audio_mridangam_left_path
            )()
            
            # Get splits
            split_mridangam_left = self.split_into_chunks(audio_mridangam_left, len(audio_tanpura))
            split_mridangam_right = self.split_into_chunks(audio_mridangam_right, len(audio_tanpura))
            split_violin = self.split_into_chunks(audio_violin, len(audio_tanpura))
            split_vocal = self.split_into_chunks(audio_vocal, len(audio_tanpura))
            split_tanpura = [audio_tanpura] * len(split_vocal)
            
            number_of_chunks = 0
            for split_id, (tanpura, vocal, violin, mri_right, mri_left) in enumerate(
                    zip(split_tanpura, split_vocal, split_violin, split_mridangam_right, split_mridangam_left)):
                write(
                    filename=os.path.join(self.chunks_path, str(split_id + split_count) + '_tanpura.wav'),
                    rate=44100,
                    data=np.array(tanpura)
                )
                write(
                    filename=os.path.join(self.chunks_path, str(split_id + split_count) + '_vocal.wav'),
                    rate=44100,
                    data=np.array(vocal)
                )
                write(
                    filename=os.path.join(self.chunks_path, str(split_id + split_count) + '_violin.wav'),
                    rate=44100,
                    data=np.array(violin)
                )
                write(
                    filename=os.path.join(self.chunks_path,
                                          str(split_id + split_count) + '_mridangam_right.wav'),
                    rate=44100,
                    data=np.array(mri_right)
                )
                write(
                    filename=os.path.join(self.chunks_path,
                                          str(split_id + split_count) + '_mridangam_left.wav'),
                    rate=44100,
                    data=np.array(mri_left)
                )
                number_of_chunks = split_id
            
            split_count = split_count + number_of_chunks
    
    def create_tmp_dataset(self, separator2, separator4):
        # Create output folder if it does not exist
        if not os.path.exists(self.output_dataset_path):
            os.mkdir(self.output_dataset_path)
        
        # Move tanpura audios to output clean chunks dataset folder
        self.move_tanpura()
        self.move_mridangam()
        self.move_violin()
        
        tmp_audios = glob.glob(os.path.join(self.chunks_path, '*vocal.wav'))
        
        # Iterate over tracks to clean
        for track in tqdm(tmp_audios):
            
            if 'vocal' in track:
                audio_id = track.split('/')[-1].split('_')[0]
                # Get vocal prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=track,
                    source='vocals'
                )
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=os.path.join(self.output_dataset_path, str(audio_id) + '_vocal.wav'),
                    rate=44100,
                    data=np.array(audio_clean)
                )
            
            '''
            if 'violin' in track:
                audio_id = track.split('_')[0]
                # Get violin prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator4,
                    track_path=os.path.join(self.chunks_path, track),
                    source='other'
                )
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=os.path.join(self.output_dataset_path, str(audio_id) + '_violin.wav'),
                    rate=44100,
                    data=np.array(audio_clean)
                )
            
            if 'mridangam_right' in track:
                audio_id = track.split('_')[0]
                # Get mridantam right prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=os.path.join(self.chunks_path, track),
                    source='accompaniment'
                )
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=os.path.join(self.output_dataset_path, str(audio_id) + '_mridangam_right.wav'),
                    rate=44100,
                    data=np.array(audio_clean)
                )
            
            if 'mridangam_left' in track:
                audio_id = track.split('_')[0]
                # Get mridangam left prediction
                audio_clean = self.get_spleeter_prediction(
                    separator=separator2,
                    track_path=os.path.join(self.chunks_path, track),
                    source='accompaniment'
                )
                audio_clean = audio_clean.sum(axis=1) / 2
                write(
                    filename=os.path.join(self.output_dataset_path, str(audio_id) + '_mridangam_left.wav'),
                    rate=44100,
                    data=np.array(audio_clean)
                )
            '''
        
        # Remove temporary dataset folder
        #shutil.rmtree(self.chunks_path, ignore_errors=True)
    
    def move_tanpura(self):
        # Move tanpura tracks to final clean chunks dataset folder
        tanpura_tracks = glob.glob(os.path.join(self.chunks_path, '*tanpura.wav'))
        for i in tanpura_tracks:
            shutil.move(i, os.path.join(self.output_dataset_path, i.split('/')[-1]))

    def move_violin(self):
        # Move tanpura tracks to final clean chunks dataset folder
        tanpura_tracks = glob.glob(os.path.join(self.chunks_path, '*violin.wav'))
        for i in tanpura_tracks:
            shutil.move(i, os.path.join(self.output_dataset_path, i.split('/')[-1]))

    def move_mridangam(self):
        # Move tanpura tracks to final clean chunks dataset folder
        tanpura_tracks = glob.glob(os.path.join(self.chunks_path, '*mridangam_left.wav'))
        for i in tanpura_tracks:
            shutil.move(i, os.path.join(self.output_dataset_path, i.split('/')[-1]))
            
        # Move tanpura tracks to final clean chunks dataset folder
        tanpura_tracks = glob.glob(os.path.join(self.chunks_path, '*mridangam_right.wav'))
        for i in tanpura_tracks:
            shutil.move(i, os.path.join(self.output_dataset_path, i.split('/')[-1]))
    
    @staticmethod
    def split_into_chunks(track, length):
        # Split audio stream into chunks of certain length
        split_track = [
            track[i * length:(i + 1) * length]
            for i in range((len(track) + length - 1) // length)
        ]
        
        return split_track
    
    @staticmethod
    def get_tonic(track):
        # Get tonic from mirdata Saraga Carnatic track
        tonic = track.tonic
        
        # If no tonic annotation available we extract it from first minute of mix track
        if tonic is None:
            tonic_extractor = estd.TonicIndianArtMusic()
            audio_mix = estd.MonoLoader(
                filename=os.path.join(track.audio_path)
            )()
            computed_tonic = tonic_extractor(audio_mix[:44100 * 60])
            return computed_tonic
        else:
            return tonic
    
    @staticmethod
    def get_multitrack_ids(track_ids, data, concerts_to_ignore):
        # Get list of multitrack audios from Saraga Carnatic dataset
        multitrack_list = []
        for track_id in track_ids:
            if data[track_id].audio_vocal_path is not None:
                if not any(concert in data[track_id].audio_vocal_path for concert in concerts_to_ignore):
                    multitrack_list.append(track_id)
                else:
                    print('Ignored track: ', track_id)
        
        return multitrack_list
    
    @staticmethod
    def get_spleeter_prediction(separator, track_path, source=None):
        # Get Spleeter prediction taking model and source to obtain as input
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(
            track_path,
            sample_rate=44100
        )
        if source == 'other':
            prediction = separator.separate(waveform)
            return prediction[source]
        
        else:
            prediction = separator.separate(waveform)
            return prediction[source]


if __name__ == '__main__':
    # Initialize DatasetCreator instance
    dataset_creator = DatasetCreator(
        dataset_path=os.path.join(Path().absolute(), 'resources')
    )
    
    # Split Saraga Carnatic tracks with multitrack audio into chunks
    dataset_creator.split_dataset_in_chunks()
    
    # Initialize Spleeter separator models
    separator2 = Separator('spleeter:2stems')
    separator4 = Separator('spleeter:4stems')
    
    # Create clean chunk dataset removing the intermediate chunk folder at the end
    dataset_creator.create_tmp_dataset(
        separator2=separator2,
        separator4=separator4,
    )