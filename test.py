import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import random
import tensorflow as tf


import preprocessing_CREMA as prep


female_ids = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,
              1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,1052,1053,1054,
              1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,
              1082,1084,1089,1091]
male_ids =list(set(list(range(1001,1092))) - set(female_ids))

creamData = prep.CreamData(
    path = 'data/CREAM-D_wav/AudioWAV/',
    female = female_ids,
    male = male_ids,
    path_to_standardize_audio_data='ProcessedData',
)

data = pd.read_csv('extracted_features.csv')
loaded_matrices = np.load('extracted_features_matrices.npy')
print(loaded_matrices.shape)

data.drop('X', axis=1, inplace=True)
data_map = {path : matrix for path in data['path'] for matrix in loaded_matrices}
print(len(data_map))

train, validation, test = creamData.train_test_split(data)
print(len(train), len(validation), len(test))

