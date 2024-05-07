import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
import librosa
from matplotlib import pyplot as plt
import seaborn as sns
import  tensorflow as tf
import tensorflow_addons as tfa


class CreamData:

    def __init__(self,
                 path,
                 female,
                 male,
                 path_to_standardize_audio_data = None,
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
                 }
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
            

    def make_dataset(self):
        dir_list = os.listdir(self.path_to_standardize_audio_data)
        print(dir_list)
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
            paths.append(self.path_to_standardize_audio_data + i)

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

        # mozda normalizacija

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
            # Get the shape of the spectrogram
        n_freq = spectrogram.shape[0]
        n_time = spectrogram.shape[1]

            # Generate random warp parameters
        source_control_point_locations = tf.random.uniform((1, 4, 2), minval=0, maxval=max_time_warp, dtype=tf.float32)
        dest_control_point_locations = tf.random.uniform((1, 4, 2), minval=0, maxval=max_time_warp, dtype=tf.float32)
            # Apply sparse image warp
        warped_spec_tensor = tfa.image.sparse_image_warp(
                            spec_tensor,
                            source_control_point_locations=source_control_point_locations,
                            dest_control_point_locations=dest_control_point_locations,
                            num_boundary_points=2)

        image, _ = warped_spec_tensor
        warped_mel = tf.squeeze(image,axis = 0)[:,:,0].numpy()

        augmented_spec = warped_mel
            # Apply time masks
        for _ in range(num_time_masks):
            mask_duration = np.random.randint(0, T)
            mask_start = np.random.randint(0, n_time - mask_duration - 1)
            augmented_spec[:, mask_start:mask_start + mask_duration] = 0

            # Apply frequency masks
        for _ in range(num_freq_masks):
            mask_width = np.random.randint(0, F)
            mask_start = np.random.randint(0, n_freq - mask_width - 1)
            augmented_spec[mask_start:mask_start + mask_width, :] = 0

        return augmented_spec


    def extract_features(self):
        unprocess_data = self.data.copy()

        process_data = []
        for sound in unprocess_data['path']:
            y,sr = librosa.load(sound)
            y = self.apply_all_time_augmentations(y)
            mel_spec = self.compute_mel_spectrogram(y,sr)
            final_spec = self.spec_augment(
                mel_spec,
                num_time_masks = self.n_mask_time_SpecAugmentation,
                num_freq_masks = self.n_mask_freq_SpecAugmentation,
                max_time_warp= self.max_time_warp_SpecAugmentation,
                T = self.T_SpecAugmentation,
                F = self.F_SpecAugmentation
            )
            process_data.append(final_spec)
        unprocess_data['X'] = process_data
        self.processed_data =  unprocess_data


    def train_test_split(self):
        '''
            For now we'll have random choice for now
        ''' 
        self.standardize_audio_duration()
        self.make_dataset()
        self.extract_features()
        female_size = len(self.female)
        male_size = len(self.male)
        data = self.processed_data.copy()
        test_female = random.sample(self.female, int(self.test_size * female_size))
        remaining = list(set(self.female) - set(test_female))
        validation_female = random.sample(remaining,int(self.validation_size * female_size))
        train_female = list(set(remaining)- set(validation_female))

        test_male = random.sample(self.male, int(self.test_size * male_size))
        remaining = list(set(self.male) - set(test_male))
        validation_male = random.sample(remaining,int(self.validation_size * male_size))
        train_male = list(set(remaining)- set(validation_male))

        train = data[data['id'].astype(int).isin(train_female + train_male)]
        validation = data[data['id'].astype(int).isin(validation_female + validation_male)]
        test = data[data['id'].astype(int).isin(test_female + test_male)]
        self.test_set = test.copy()
        self.train_set = train.copy()
        self.validation_set = validation.copy()
        



        
    def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize = 10):

        df_cm = pd.DataFrame(
            confusion_matrix, index =class_names, columns = class_names
        )
        fig = plt.figure(figsize=figsize)

        try:
            heatmap = sns.heatmap(df_cm, annt = True, fmt = "d")
        except ValueError:
            raise ValueError("Confusion matrix must be right")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        

