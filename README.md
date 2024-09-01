# Emotion Recognition

## Authors
 - Lazar Stanojević (1013/2023)
 - Lazar Kračunović (36/2019)

## Overview
This project focuses on predicting emotions in speech using Convolutional Neural Networks (CNNs). The goal is to accurately classify different emotions such as Anger, Disgust, Fear, Happy, Neutral, and Sad based on audio data. This project was developed as part of the Machine Learning course at the Faculty of Mathematics, University of Belgrade.


## Dataset
The dataset was obtained from the Kaggle platform and can be found at the following [link](https://www.kaggle.com/datasets/ejlok1/cremad.)

### Data description
CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).
We were only predicting emotions, we didn't use emotion levels in our prediction procedure.

## Project Setup
First of all, clone the repository:
```shell
git clone https://github.com/lazars01/EmotionRecognition
cd EmotionRecognition
```
Then, install Poetry, which you can achieve with:
```shell
pip install poetry
```
 Poetry simplifies the process of managing project dependencies, ensuring that the required packages and versions are installed and maintained.

```shell
poetry install
```

After you set up poetry, run **extract.py** script, which will start preprocessing and saving data.
```shell
poetry run extract
```
Make sure that you have atleast 20GB available disk space. \
You can run training script, which trains different models and saves results to files, using poetry script
```shell
poetry run train
```
All the necessary functions used in jupyter notebooks can be found in **utils.py** script. \
For preprocessing, we are using features of CreamData class, defined in **preprocessing_CREMA.py**. \
Our custom torch Dataset is defined in **custom_dataset.py** file, and is allowing us to work with big datasets. \
We are using CNN model defined in **model.py** for classification. \
After all, our jupyter notebooks summarize all that have been done.

## Literature
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)
- [SepTr: Separable Transformer for Audio Spectrogram Processing](https://www.researchgate.net/publication/363646767_SepTr_Separable_Transformer_for_Audio_Spectrogram_Processing)
- [Speech Emotion Recognition with deep learning](https://ieeexplore.ieee.org/document/8049931)
- [A Comparison on Data Augmentation Methods Based on Deep Learning for Audio Classification](https://iopscience.iop.org/article/10.1088/1742-6596/1453/1/012085/meta)
