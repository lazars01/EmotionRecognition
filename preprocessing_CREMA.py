import os
import pandas as pd
import numpy as np
import random

class CreamData:

    def __init__(self,
                 path,
                 female,
                 male,
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
        self.female = female
        self.male = male
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.emotion_dict = emotion_dict
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

    def make_dataset(self):
        dir_list = os.listdir(self.path)
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
            paths.append(self.path + i)

        df = pd.DataFrame.from_dict({
                'id': ids,
                'gender': gender,
                'emotion': emotions,
                'lable':emotions_lable,
                'path': paths
            })

        return df


    '''
    For now we'll have random choice for now
    '''
    def train_test_split(self):

        data = self.make_dataset()
        female_size = len(self.female)
        male_size = len(self.male)

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
        



        

