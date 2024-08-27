import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
import librosa

import warnings
warnings.filterwarnings('ignore')

import gc
import  tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import minmax_scale

random.seed(7)

def clear_memory(variables):
    for var in variables:
        del var
    gc.collect()

class CreamData:

    def __init__(self,
                 path,
                 female,
                 male,
                 path_to_standardize_audio_data = None,
                 spec_augmentation = True,
                 standardize_audio = True,
                 window_len = 512,
                 hop_length = 64,
                 n_fft = 1024,
                 n_mels = 128,
                 audio_duration = 4,
                 train_size=0.8,
                 validation_size=0.1,
                 test_size=0.1,
                 emotion_dict={
                     "SAD": "sad",
                     "ANG": "angry",
                     "DIS": "disgust",
                     "FEA": "fear",
                     "HAP": "happy",
                     "NEU": "neutral"
                 },
                 batch_size = 32
                 ):
        self.path = path
        self.path_to_standardize_audio_data = path_to_standardize_audio_data
        self.female = female
        self.male = male
        self.audio_duration = audio_duration
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.emotion_dict = emotion_dict
        self.window_length = window_len
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mask_time_SpecAugmentation = 1
        self.n_mask_freq_SpecAugmentation = 1
        self.max_time_warp_SpecAugmentation = 80
        self.T_SpecAugmentation = 80
        self.F_SpecAugmentation = 15
        self.data = None
        self.processed_data = None
        self.train_set = None
        self.test_set = None
        self.validation_set  = None
        self.BATCH_SIZE = batch_size
        self.want_augmentation = spec_augmentation
        self.standardize_audio = standardize_audio

    def get_emotion(self, filename):
        filename = filename.split("_")
        id = filename[0]
        emotion1 = self.emotion_dict[filename[2]]
        if int(filename[0]) in self.female:
            emotion2 = "_female"
        else:
            emotion2 = "_male"
        emotion = emotion1 + emotion2
        return (id,emotion, emotion1, emotion2[1:])
    
    def standardize_audio_duration(self):

        input_dir = self.path
        output_dir = self.path_to_standardize_audio_data
        if(output_dir == None):
            return
        
        os.makedirs(output_dir,exist_ok=True)
        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                input_file = os.path.join(input_dir,filename)
                output_file = os.path.join(output_dir,filename)
                
                y,sr =  librosa.load(input_file)

                target_samples = int(self.audio_duration * sr)

                if len(y) < target_samples:
                    y_padded = librosa.util.pad_center(y,size = target_samples)
                else:
                    y_padded = y[:target_samples]
                
                sf.write(output_file,y_padded,sr)
            print(f"Standard {output_file}")
            
    def make_dataset(self):

        dir_list = os.listdir(self.path_to_standardize_audio_data)
        dir_list.sort()

        ids = []
        emotions_lable = []
        emotions = []
        gender = []
        paths = []

        for i in dir_list:
            ids.append(self.get_emotion(i)[0])
            emotions_lable.append(self.get_emotion(i)[1])
            emotions.append(self.get_emotion(i)[2])
            gender.append(self.get_emotion(i)[3])
            paths.append(self.path_to_standardize_audio_data + '/' + i)

        df = pd.DataFrame.from_dict({
                'id': ids,
                'gender': gender,
                'emotion': emotions,
                'lable':emotions_lable,
                'path': paths
            })
        self.data = df

    def compute_mel_spectrogram(self,y,sr):
        stft = librosa.stft(y, n_fft= self.n_fft, hop_length=self.hop_length,win_length= self.window_length)
        magnitude = np.abs(stft) ** 2
        mel_spec = librosa.feature.melspectrogram(S=magnitude,sr = sr,n_fft= self.n_fft, hop_length= self.hop_length, n_mels = self.n_mels)

        log_mel = librosa.power_to_db(mel_spec, ref = np.max)

        return log_mel
    
    def apply_all_time_augmentations(self,y):

        def augmentation_noise(sound, noise_val):

            noise_amp = noise_val * np.random.uniform() * np.random.normal(size= sound.shape[0])
            sound = sound.astype('float64') + noise_amp * np.random.normal(size = sound.shape[0])
            return sound

        def augmentation_shift(sound):
            
            shift_range = int(np.random.uniform(low = -5, high = 5) * 1000)
            return np.roll(sound, shift_range)

        def augmentation_speed_and_pitch(sound):

            length_change = np.random.uniform(low = 0.8, high = 1)
            speed_frac = 1.2 / length_change

            tmp = np.interp(np.arange(0, len(sound), speed_frac), np.arange(0, len(sound)), sound)
            min_len = min(sound.shape[0], tmp.shape[0])
            sound *= 0
            sound[0: min_len] = tmp[0:min_len]
            return sound
    
        y = augmentation_noise(y,noise_val = 0.05)
        y = augmentation_shift(y)
        y = augmentation_speed_and_pitch(y)
        return y
  
    def spec_augment(self,spectrogram, num_time_masks=2, num_freq_masks=2, max_time_warp=80, T=100, F=20):

        spec_tensor = tf.convert_to_tensor(spectrogram[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        n_freq = spectrogram.shape[0]
        n_time = spectrogram.shape[1]

        source_control_point_locations = tf.random.uniform((1, 2, 2), minval=0, maxval=max_time_warp, dtype=tf.float32)
        dest_control_point_locations = tf.random.uniform((1, 2, 2), minval=0, maxval=max_time_warp, dtype=tf.float32)
        # Apply sparse image warp
        warped_spec_tensor, _= tfa.image.sparse_image_warp(
                spec_tensor,
                source_control_point_locations=source_control_point_locations,
                dest_control_point_locations=dest_control_point_locations,
               num_boundary_points=2
            )

        augmented_spec = tf.squeeze(warped_spec_tensor,axis = 0)[:,:,0].numpy()
        
        
        for _ in range(num_time_masks):
            mask_duration = np.random.randint(0, T)
            mask_start = np.random.randint(0, n_time - mask_duration - 1)
            augmented_spec[:, mask_start:mask_start + mask_duration] = 0

        for _ in range(num_freq_masks):
            mask_width = np.random.randint(0, F)
            mask_start = np.random.randint(0, n_freq - mask_width - 1)
            augmented_spec[mask_start:mask_start + mask_width, :] = 0

        clear_memory([warped_spec_tensor])
        tf.keras.backend.clear_session()
        return augmented_spec

    def convert_to_final_spec(self, sounds_path):
        y,sr = librosa.load(sounds_path)
        y = self.apply_all_time_augmentations(y)
        mel_spec = self.compute_mel_spectrogram(y,sr)
        
        if self.want_augmentation:
            final_spec = self.spec_augment(
                    mel_spec,
                    num_time_masks = self.n_mask_time_SpecAugmentation,
                    num_freq_masks = self.n_mask_freq_SpecAugmentation,
                    max_time_warp= self.max_time_warp_SpecAugmentation,
                    T = self.T_SpecAugmentation,
                    F = self.F_SpecAugmentation
                )
        else:
            return minmax_scale(mel_spec.flatten()).reshape(mel_spec.shape)
        # Min/Max scaling 
        final_spec = minmax_scale(final_spec.flatten()).reshape(final_spec.shape)
        return final_spec

    def extract_features_with_labels(self, batch_data, output_path):
        process_data = []
        lables = []
        for path, label in zip(batch_data['path'],batch_data['emotion']):
            
            d = self.convert_to_final_spec(path)
            process_data.append(d)
            lables.append(label)

        np.savez(output_path, features = np.array(process_data), labels= np.array(lables))
        clear_memory([process_data,lables])

    def process_and_save_features(self, data_sets, batch_size, output_dir):
        num_batches = len(data_sets) // batch_size + (1 if len(data_sets) % batch_size != 0 else 0)

        if  not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            batch_data = data_sets[start:end]
            output_path = os.path.join(output_dir, f'batch_{batch}')
            print(f'{output_path}')
            self.extract_features_with_labels(batch_data, output_path)
        clear_memory([])

    def train_test_split(self):
        
        female_size = len(self.female)
        male_size = len(self.male)
        data = self.data.copy()
        test_female = random.sample(self.female, int(self.test_size * female_size))
        remaining = list(set(self.female) - set(test_female))
        validation_female = random.sample(remaining,int(self.validation_size * female_size))
        train_female = list(set(remaining)- set(validation_female))

        test_male = random.sample(self.male, int(self.test_size * male_size))
        remaining = list(set(self.male) - set(test_male))
        validation_male = random.sample(remaining,int(self.validation_size * male_size))
        train_male = list(set(remaining)- set(validation_male))

        train_indices = data['id'].astype(int).isin(train_female + train_male)
        train = data[train_indices]
        validation_indices = data['id'].astype(int).isin(validation_female + validation_male)
        validation = data[validation_indices]
        test_indices = data['id'].astype(int).isin(test_female + test_male)
        test = data[test_indices]
        self.test_set = test.copy()
        self.train_set = train.copy()
        self.validation_set = validation.copy()
        clear_memory([test,train,validation])
    
    def process_data(self):

        if self.standardize_audio:
            self.standardize_audio_duration()
        
        print('Making Dataframe')
        self.make_dataset()
        print('Spliting data')
        self.train_test_split()

        print(f'Want augmentation: {self.want_augmentation}')
        if self.want_augmentation:
            print('Processing training data')
            self.process_and_save_features(self.train_set,self.BATCH_SIZE,'batches_augment/train')
            print('Processing validation data')
            self.process_and_save_features(self.validation_set, self.BATCH_SIZE, 'batches_augment/validation')
            print('Processing test data')
            self.process_and_save_features(self.test_set, self.BATCH_SIZE, 'batches_augment/test')
        else:
            print('Processing training data')
            self.process_and_save_features(self.train_set,self.BATCH_SIZE,'batches/train')
            print('Processing validation data')
            self.process_and_save_features(self.validation_set, self.BATCH_SIZE, 'batches/validation')
            print('Processing test data')
            self.process_and_save_features(self.test_set, self.BATCH_SIZE, 'batches/test')


        



    